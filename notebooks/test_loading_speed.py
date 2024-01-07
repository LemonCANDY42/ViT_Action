# -*- coding: utf-8 -*-
# %% [markdown]
# @Time    : 2023/11/23 17:57
# @Author  : Kenny Zhou
# @FileName: test_loading_speed.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

# %%
import cv2
from PIL import Image
from pathlib import Path
import time
import numpy as np

# %%
def test_loading(path,times:int,module:str,describe:str):
    assert module in ["cv2","PIL","npy","npz"]

    # if type(path) is Path:
    #     path = str(path)

    t0= time.time()
    for i in range(times):
        if module == "cv2":
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif module == "npy":
            img = np.load(path)
            type(img)
        elif module == "npz":
            data_from_npz = np.load(path)
            img = data_from_npz['img']
            type(img)
        else:
            img = np.asarray(Image.open(path), dtype=np.uint8)

    print(f"{module}读取{describe}数据{times}次用时：{time.time()-t0:.5f}s")

# %%
if __name__ == '__main__':
    # 比较PIL和opencv读取jpg图片到numpy.ndarray的速度
    small_jpg = Path("/Users/kennymccormick/tempForMe/2023-11-23/small.jpg")
    large_jpg = Path("/Users/kennymccormick/tempForMe/2023-11-23/large.jpeg")
    middle_jpg = Path("/Users/kennymccormick/tempForMe/2023-11-23/middle.jpg")
    large_png = Path("/Users/kennymccormick/tempForMe/2023-11-23/large.png")

    test_loading(small_jpg,100,"cv2","small.jpg")
    test_loading(small_jpg, 100, "PIL", "small.jpg")

    test_loading(middle_jpg, 100, "cv2", "middle_jpg")
    test_loading(middle_jpg, 100, "PIL", "middle_jpg")

    test_loading(large_jpg, 100, "cv2", "large_jpg")
    test_loading(large_jpg, 100, "PIL", "large_jpg")

    test_loading(large_png, 100, "cv2", "large_png")
    test_loading(large_png, 100, "PIL", "large_png")


# %%
def resize_keep_aspect_ratio(image_path, target_width=256):
    # 读取图像
    img = cv2.imread(image_path)

    # 获取原始图像的宽度和高度
    height, width = img.shape[:2]

    # 计算缩放比例
    ratio = target_width / width

    # 计算新的高度
    target_height = int(height * ratio)

    # 使用cv2.resize进行缩放
    resized_img = cv2.resize(img, (target_width, target_height))

    return resized_img

# 示例使用
resized_image = resize_keep_aspect_ratio(str(middle_jpg))
print(str(middle_jpg.parent) + "/" + (middle_jpg.stem) + "_resize_256.jpg")
cv2.imwrite(str(middle_jpg.parent) + "/" + (middle_jpg.stem) + "_resize_256.jpg", resized_image)

# %%
repetition = 12
video_counts = 40

test_path = str(middle_jpg.parent) + "/" + (middle_jpg.stem) + "_resize_256.jpg"
npy_path = str(middle_jpg.parent) + "/" + (middle_jpg.stem) + f"_resize_256_t{repetition}.npy"
npz_path = str(middle_jpg.parent) + "/" + (middle_jpg.stem) + f"_resize_256_t{repetition}_nums_{video_counts}.npz"

img = cv2.imread(test_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

video_array = np.expand_dims(img,0).repeat(repetition,axis=0)
print(video_array.shape)

assert video_counts>=1
video_label_pairs = {str(name):video_array for name in range(video_counts-1)}
np.save(npy_path, video_array)
np.savez_compressed(npz_path, img=video_array,**video_label_pairs,compress='lzma')


# %%
def windows_to_wsl_path(windows_path):
    # Check if the path starts with a drive letter and a colon
    if len(windows_path) > 1 and windows_path[1] == ':':
        drive_letter = windows_path[0].lower()  # Convert drive letter to lowercase
        wsl_path = "/mnt/" + drive_letter + windows_path[2:].replace('\\', '/')
        return wsl_path
    else:
        return "The path does not appear to be a valid Windows path."


# %%
from einops import rearrange,reduce,repeat

# %%
img = cv2.imread(windows_to_wsl_path(r"C:\Users\LWRF4\Pictures\Screenshots\Snipaste_2022-02-09_15-45-17.png"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = rearrange(img,'h w c-> 1 c h w')

img1 = np.concatenate((img,img),axis = 0)
img1 = np.concatenate((img1,img),axis = 0)

print(img.shape)
print(img1.shape)
print(img1)

# %%
test_loading(test_path, 1000*repetition, "cv2", f"middle_resize_256_{repetition}")
test_loading(npy_path,1000,"npy",f"middle_resize_256_{repetition}")
test_loading(npz_path,1000,"npz",f"middle_resize_256_{repetition}")

# %%
a = np.load(windows_to_wsl_path(r"G:\Kinetics-400\raw-part\compress\train_256_frames_12fps_np\abseiling\-3B32lodo2M_000059_000069.npy"))
b = np.load(windows_to_wsl_path(r"G:\Kinetics-400\raw-part\compress\train_256_frames_12fps_np\abseiling\-7kbO0v4hag_000107_000117.npy"))
c = np.load(windows_to_wsl_path(r"G:\Kinetics-400\raw-part\compress\train_256_frames_12fps_np\abseiling\-bwYZwnwb8E_000013_000023.npy"))

# %%
a == b
a == c
b == c

# %%

# %%
