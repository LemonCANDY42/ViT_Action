# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms 

import torchshow as ts
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange
from PIL import Image

def tensor_memory_size(tensor):
    tensor_memory_size = tensor.element_size() * tensor.numel()
    
    units = ['B', 'KB', 'MB', 'GB']
    unit_index = 0

    while tensor_memory_size > 1024 and unit_index < len(units) - 1:
        tensor_memory_size /= 1024.0
        unit_index += 1

    return f"{tensor_memory_size:.2f} {units[unit_index]}"



# %%
torch.cuda.is_available()

# %%
z = np.array([[1,2,3,4],[0,1,2,3],[0,0,1,2],[0,0,0,1]])
z_new = z.copy()
z_new[z_new > 0] -= 1
z_new,z
z - z_new

# %%
image_size = 224
patch_size = 16
in_channels = 3
num_frames = 16

num_patches = (image_size // patch_size) ** 2
patch_dim = in_channels * patch_size ** 2
patch_dim

# %%
shape_size = 8
a = torch.linspace(0, shape_size**2-1, shape_size**2,dtype=int).reshape(shape_size,shape_size)
a = rearrange(a,'h w -> 1 h w')

a = a.repeat((3,1,1)).unsqueeze(0)

# a = repeat(a,'b h w -> b c h w',c=3)
a[:,1] *= 10
a[:,2] *= 100
a.shape
a_after = rearrange(a,'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = 4, p2 = 4)
print(a,a.shape,"\n",a_after,a_after.shape)

# %%
a.shape,tensor_memory_size(a)

# %%
image = Image.open("/Users/kennymccormick/Pictures/01f55c5dcf95b5a8012129e205b0bd.jpg@1280w_1l_2o_100sh.jpg")
image = image.resize((224,224))
transform = transforms.Compose([ 
    transforms.PILToTensor() 
])
img_tensor = transform(image) 
x = rearrange(img_tensor,'c h w -> 1 1 c h w')
x.shape

# %%
x = torch.squeeze(x,0,)
print(x.shape)
# img_tensor = torch.narrow(img_tensor,0,0,3)
ts.show(x)


# %%
x = torch.rand([1,num_frames,in_channels,image_size,image_size])

# %%
# _ = rearrange(img_tensor,'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)

_ = rearrange(x,'b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = patch_size, p2 = patch_size)
print(_.shape)

# %%
dim = 192
to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

# %%
x = to_patch_embedding(x)

print(x.shape)

# %%
dim_head = 64
heads = 8
dim = 192

inner_dim = dim_head *  heads

to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
inner_dim,to_qkv

space_token = nn.Parameter(torch.randn(1, 1, dim))
inner_dim,to_qkv,space_token.shape

# %%
b, t, n, _ = x.shape

cls_space_tokens = repeat(space_token, '() n d -> b t n d', b = b, t=t)
x.shape,cls_space_tokens.shape

# %%

# %%
x1 = torch.cat((cls_space_tokens, x), dim=2)
x1.shape

# %%
# Aittention

# %%
dim = 192
heads = 3
dim_head = 64

inner_dim = dim_head *  heads
project_out = not (heads == 1 and dim_head == dim)

scale = dim_head ** -0.5

inner_dim,project_out,scale

# %%
x = torch.rand([16,197,192])
to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

b, n, _, h = *x.shape, heads
b, n, _, h,to_qkv.parameters

# %%
tqkv = to_qkv(x)
qkv = to_qkv(x).chunk(3, dim = -1)
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
x.shape,tqkv.shape,q.shape,k.shape,v.shape

# %%

# %%
# y = torch.rand([1,num_frames,in_channels,image_size,image_size])
y = torch.rand([64,num_frames,in_channels,512,256])
# y = rearrange(y,'b t c h w -> b c t h w')
y.shape

# %%
y.shape,tensor_memory_size(y)

# %%
conv1 =  nn.Conv2d(in_channels,16, kernel_size=1)
conv1.weight

# %%
conv2 = nn.Conv2d(3,2,kernel_size=(16,1,16),stride=1)

# %%
y = conv2(y)
y.shape

# %%
conv1.parameters
