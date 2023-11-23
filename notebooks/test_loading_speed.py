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
    assert module in ["cv2","PIL"]

    # if type(path) is Path:
    #     path = str(path)

    t0= time.time()
    for i in range(times):
        if module == "cv2":
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            img = np.asarray(Image.open(path), dtype=np.uint8)

    print(f"{module}读取{describe}图片100次用时：{time.time()-t0:.5f}s")

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
