# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 22:09
# @FileName: dataset_h5.py

import h5py
import numpy as np
from pathlib import Path

from utils.wsl_tools import windows_to_wsl_path

# 假设你有一个函数来生成你的数据，这里用一个简化的例子代替
def generate_data(T, W):
    return np.random.randint(0, 256, size=(T, 3, 256, W), dtype=np.uint8)

# 定义数据集的一些参数
num_classes = 4
samples_per_class = 1
max_T = 20*12
max_W = 256*4

def generate_dataset(path):
    # 创建HDF5文件
    with h5py.File(windows_to_wsl_path(path), 'w', libver='latest') as f: #, swmr=True
        f.swmr_mode = True
        # 创建一个数据集来存储所有数据
        dataset = f.create_dataset('data', shape=(num_classes * samples_per_class, 1, 3, 256, 1), maxshape=(None, None, 3, 256, None),
                                   chunks=(1, max_T, 3, 256, max_W), dtype=h5py.vlen_dtype(np.dtype('uint8')), compression='lzf') # uint8
    
        # 创建一个数据集来存储类别信息
        class_names_dataset = f.create_dataset('class_names', shape=(num_classes,), dtype=h5py.string_dtype())
    
        # 创建类别到数据集位置的索引
        class_index = {}
    
        for class_id in range(num_classes):
            start_idx = class_id * samples_per_class
            end_idx = start_idx + samples_per_class
            
    
            # 生成每个类别的数据并写入数据集
            for idx in range(start_idx, end_idx):
                T = np.random.randint(1, max_T + 1)
                W = np.random.randint(1, max_W + 1)
                data = generate_data(T, W)
                
                print(data.shape)
                # 确保数据集可以容纳新的数据形状
                if idx >= dataset.shape[0]:
                    dataset.resize((idx + 1, T, 3, 256, W))
                
                dataset[idx] = data
                
            # 记录每个类别在数据集中的位置范围
            class_index[class_id] = (start_idx, end_idx)
    
            # 将类别名称写入类别信息数据集
            class_names_dataset[class_id] = f'class_{class_id}'
    
        # 保存类别索引
        f.create_dataset('class_index', data=np.array(list(class_index.items())))

def read_category(h5_file, data_index):
    with h5py.File(h5_file, 'r') as f:
        # 读取类别索引
        class_index = dict(f['class_index'])

        # 找到数据点所属的类别
        category_id = None
        for class_id, (start_idx, end_idx) in class_index.items():
            if start_idx <= data_index < end_idx:
                category_id = class_id
                break

        # 读取类别信息
        if category_id is not None:
            class_names_dataset = f['class_names']
            category_name = class_names_dataset[category_id]
            return category_id, category_name
        else:
            return None

if __name__ == "__main__":
    generate_dataset(r'G:\test\large_dataset.h5')

#
#     # 示例：读取数据点索引为10的类别信息
#     data_index_to_read = 10
#     category_info = read_category('large_dataset.h5', data_index_to_read)
#
#     if category_info is not None:
#         category_id, category_name = category_info
#         print(f'Data point {data_index_to_read} belongs to category {category_name} (ID: {category_id})')
#     else:
#         print(f'Error: Could not find category for data point {data_index_to_read}')