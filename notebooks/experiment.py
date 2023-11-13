# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn
import torchvision.transforms as transforms 

import torchshow as ts
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange
from PIL import Image

# %%
image_size = 224
patch_size = 16
in_channels = 3
num_frames = 16

num_patches = (image_size // patch_size) ** 2
patch_dim = in_channels * patch_size ** 2

# %%
x = torch.rand([1,num_frames,in_channels,image_size,image_size])

# %%
shape_size = 8
a = torch.linspace(0, shape_size**2-1, shape_size**2,dtype=int).reshape(shape_size,shape_size)
a = rearrange(a,'h w -> 1 h w')

a = repeat(a,'b h w -> b c h w',c=3)
b = a[:,2][0]
a[:,2] += 1


print(a[:,2][0])
# a_after = rearrange(a,'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = 4, p2 = 4)
# print(a,a.shape,"\n",a_after,a_after.shape)

# %%
image = Image.open("/Users/kennymccormick/Pictures/01f55c5dcf95b5a8012129e205b0bd.jpg@1280w_1l_2o_100sh.jpg")
image = image.resize((224,224))
transform = transforms.Compose([ 
    transforms.PILToTensor() 
])
img_tensor = transform(image) 
img_tensor = rearrange(img_tensor,'c h w -> 1 1 c h w')
img_tensor.shape

# img_tensor = torch.squeeze(img_tensor,0,)
# print(img_tensor.shape)
# # img_tensor = torch.narrow(img_tensor,0,0,3)
# ts.show(img_tensor)


# %%
# _ = rearrange(img_tensor,'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)

img_tensor = torch.squeeze(img_tensor,0)
_ = rearrange(img_tensor,'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size)
print(_.shape)

# %%
# Rearrange?

# %%
dim = 192
to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

# %%
print(to_patch_embedding[0].Parameter)

# %%
dim_head = 64
heads = 8
dim = 192

inner_dim = dim_head *  heads

to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
inner_dim,to_qkv

# %%
b, n, _, h = *x.shape, self.heads

# %%
qkv = to_qkv(x).chunk(3, dim = -1)

# %%
