# -*- coding: utf-8 -*-
# @Time    : 2023/11/24 15:31
# @Author  : Kenny Zhou
# @FileName: build_np_dataset.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com


import numpy as np
import subprocess
import concurrent.futures
from pathlib import Path
import cv2
from einops import rearrange,reduce,repeat

import json
import os
import signal
from tqdm import tqdm

from utils.wsl_tools import wsl_to_windows_path
from utils.GLOBAL_CONSTANTS import VIDEO_EXTENSIONS

STATE_FILE = './resume_frames_state.txt'

def save_state(state_data):
    with open(STATE_FILE, 'w') as f:
        json.dump(state_data, f)


def frames_2_npy(frames_classes_folder, output_root):
	"""
	Extract frames from a video using FFmpeg with NVDEC GPU-acceleration.
	If the video is corrupted or the process fails, it returns False.

	Args:
		frames_classes_folder (Path): The path to the video file.
		output_root (Path): The root path to store the extracted frames.

	Returns:
		bool: True if frames were extracted successfully, False otherwise.
	"""

	output_path = output_root / frames_classes_folder.name
	output_path.mkdir(parents=True, exist_ok=True)
	

	subfolders = [folder for folder in frames_classes_folder.iterdir() if folder.is_dir()]
	
	file_path_str = None
	try:
		for frames_path in subfolders:
			
			
			a = 0
			
			jpg_files = sorted(frames_path.glob('*.jpg'))
			if len(jpg_files):
				out_np_path = str(output_path) + "\\" + str(frames_path.name) + ".npz"
				
				video_arrays = None
				img = None
				for file_path in jpg_files:
					file_path_str = str(file_path)
					assert int(file_path_str[-7:-4]) - a == 1 , f"Wrong!{frames_classes_folder},{frames_path}"
					a += 1
					
					img = cv2.imread(file_path_str)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					img = rearrange(img, 'h w c-> 1 c h w')
					
					if video_arrays is None:
						video_arrays = img
					else:
						video_arrays = np.concatenate((video_arrays, img), axis=0)

				if video_arrays is not None:
					
					np.savez_compressed(out_np_path, video=video_arrays,compress='lzma')
					
		return True, None
		
	except Exception as e:
		print(e)
		return False,file_path_str

def process_all_frames(root_folder, output_root):
	"""
	Process all video files in a directory using multithreading.
	Prints out corrupted videos and skips them.

	Args:
		frames_classes_folder (str): The path to the folder containing video files.
		output_root (str): The root path to store the extracted frames.
	"""
	root_folder = Path(root_folder)

	output_root = Path(output_root)
	print(output_root)
	
	subfolders = [folder for folder in root_folder.iterdir() if folder.is_dir()]
	
	# sub_class_folders = []
	#
	# for sub_class_folder_item in subfolders:
	#
	# 	sub_class_folders += [folder for folder in sub_class_folder_item.iterdir() if folder.is_dir()]
	corrupted_path = []
	# Use ThreadPoolExecutor to process video files in parallel
	with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
		futures = {executor.submit(frames_2_npy, frames_classes_folder,output_root): frames_classes_folder for frames_classes_folder in
				   tqdm(subfolders, desc="Processing Videos")}

		for future in concurrent.futures.as_completed(futures):
			frames_classes_folder = futures[future]
			print(f"Processing {str(frames_classes_folder)}")
			
			success,path = future.result()
			
			if not success:
				corrupted_path.append(path)
			
				
	save_state(corrupted_path)




if __name__ == "__main__":
	"""
		The tree of files and folders before processing looks like this:

			-out_{fps}fps
				-A
					-video_1
						-video_1_00001.jpg
						-video_1_00002.jpg
						...
					-video_2
						-video_2_00001.jpg
						-video_2_00002.jpg
						...
				-B
					-video_3
						-video_3_00001.jpg
						-video_3_00002.jpg
						...
					-video_4
						-video_4_00001.jpg
						-video_4_00002.jpg
						...


		after:
		
			-train_XXX_npz
				-A
					-video_1.npz
					-video_2.npz
					...
				-B
					-video_3.npz
					-video_4.npz
					...

	"""

	input_videos_path = r'G:\Kinetics-400\raw-part\compress\train_256_frames_12fps'
	output_frames_path = r'G:\Kinetics-400\raw-part\compress\train_256_frames_12fps_np'

	# wsl = False
	# if wsl:
	# input_videos_path = wsl_to_windows_path(input_videos_path)
	# output_frames_path = wsl_to_windows_path(output_frames_path)


	process_all_frames(input_videos_path, output_frames_path)
