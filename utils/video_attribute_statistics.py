# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 12:14
# @FileName: video_attribute_statistics.py

import cv2
import subprocess
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from fractions import Fraction  # Import Fraction for handling frame rate

from utils.time_tools import timeit
from utils.GLOBAL_CONSTANTS import VIDEO_EXTENSIONS


def get_video_info(video_path,use):
    try:
        if use == "cv2":
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = num_frames / fps
            cap.release()
        else:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,duration", "-of", "csv=p=0", video_path],
                capture_output=True,
                text=True
            )
            width, height, frame_rate, duration = map(str, result.stdout.strip().split(',')) # because frame_rate just like 30000/1001、 30/1 etc.
            width, height, duration = map(float,(width, height, duration))
            # Parse frame rate as a fraction
            fps_fraction = Fraction(frame_rate)
            
            # Calculate the FPS as float
            fps = float(fps_fraction.numerator) / float(fps_fraction.denominator)
        
        return Path(video_path).name, width, height, fps, duration
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return Path(video_path).name,None,None,None,None

def process_folder(folder,use):
    # video_files = glob.glob(os.path.join(folder, "*.mp4"))
    video_files = [p.resolve() for p in Path(folder).glob("**/*") if p.suffix.lower() in VIDEO_EXTENSIONS or p.suffix.upper() in VIDEO_EXTENSIONS ]
    
    videos_stats = []

    with tqdm(total=len(video_files), desc=f"Processing videos in {folder.name}") as pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(get_video_info, video_files,use))
            pbar.update(len(video_files))

    for result in results:
        if result[1]:
            name,width, height, fps, duration = result
            videos_stats.append({
                "name":name,
                "width": width,
                "height":height,
                "fps": fps,
                "duration": duration,
            })
        else:
            name, width, height, fps, duration = result
            videos_stats.append({
                "name":name,
                "width": None,
                "height":None,
                "fps": None,
                "duration": None,
            })
    return videos_stats
@timeit
def main(root_folder, output_csv,use = "cv2"):
    
    assert use in ["cv2","ffmpeg"]
    
    folder_stats = {"train":{},"val":{}}

    root_path = Path(root_folder)
    for subset_folder in root_path.iterdir():
        if subset_folder.is_dir():
            for class_folder in subset_folder.iterdir():
                if class_folder.is_dir():
                    print(f"Processing folder: {class_folder}")
                    folder_stats[subset_folder.name[:-4]][class_folder.name] = process_folder(class_folder,use) # delete _256

    print("\nCreating DataFrame...")
    df_list = []
    for dataset_name, class_data in folder_stats.items():
        for class_name, videos in class_data.items():
            for stats in videos:
                name = stats["name"]
                fps = stats["fps"]
                width = stats["width"]
                height = stats["height"]
                duration = stats['duration']
    
                df_list.append({
                    'dataset_name':dataset_name,
                    'class': class_name,
                    'video_name': name,
                    "width": width,
                    "height":height,
                    "fps": fps,
                    "duration": duration,
                })

    df = pd.DataFrame(df_list)
    df.to_csv(output_csv, index=False)
    print(f"\nDataFrame saved to {output_csv}")

if __name__ == "__main__":
    from utils.wsl_tools import windows_to_wsl_path
    
    root_folder = windows_to_wsl_path(r"G:\Kinetics-400\raw-part\compress")
    output_csv = windows_to_wsl_path(r"G:\Kinetics-400\label\video_stats.csv")
    main(root_folder, output_csv,use = "cv2") # ffmpeg has some bugs.
    
