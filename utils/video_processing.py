# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 9:15
# @FileName: video_processing.py

import subprocess
import concurrent.futures
from pathlib import Path
import json
import os
import signal
from tqdm import tqdm

from utils.wsl_tools import wsl_to_windows_path
from utils.GLOBAL_CONSTANTS import VIDEO_EXTENSIONS

STATE_FILE = './resume_state.txt'
STOP_REQUESTED = False
DELETE_STATE_FILE = True

def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Signal received, stopping...")

def save_state(state_data):
    with open(STATE_FILE, 'w') as f:
        json.dump(state_data, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return []

def extract_frames(video_path, output_root,fps:int=1,platform=False):
    """
    Extract frames from a video using FFmpeg with NVDEC GPU-acceleration.
    If the video is corrupted or the process fails, it returns False.

    Args:
        video_path (Path): The path to the video file.
        output_root (Path): The root path to store the extracted frames.

    Returns:
        bool: True if frames were extracted successfully, False otherwise.
    """
    output_path = output_root /video_path.parent.name/ (video_path.stem)
    output_path.mkdir(parents=True, exist_ok=True)

    # print(output_root)
    # print(video_path.parent.name)
    # print(video_path.stem)
    # print(output_path)
    
    if platform == "cuda":
        # FFmpeg command using NVDEC for GPU acceleration
        ffmpeg_path = '/mnt/d/ffmpeg-6.0-full_build/bin/ffmpeg.exe'
        hwaccel_parameter = ['-hwaccel', 'nvdec',  # Use CUDA for GPU acceleration
                                    '-hwaccel_device', '0']
    elif platform == "mps":
        ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
        hwaccel_parameter = ["-hwaccel","videotoolbox"]
    else:
        ffmpeg_path = '/usr/local/bin/ffmpeg'
        hwaccel_parameter = [""]

    cmd = [
        ffmpeg_path,
        *hwaccel_parameter,
        # '-hwaccel_output_format', 'cuda',  # Set the hardware acceleration output format to CUDA
        # '-c:v', 'h264_nvenc',  # Use the h264_nvenc decoder
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        str(output_path / f'{video_path.stem}_%05d.jpg')
    ]
    
    # Execute the FFmpeg command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Frames extracted for {video_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else 'An unknown error occurred'
        print(f"Error processing {video_path}: {error_message}")
        return False

def process_all_videos(video_folder, output_root,fps:int = 1 ,platform="cuda" ,use_last_state=True):
    global DELETE_STATE_FILE
    """
    Process all video files in a directory using multithreading.
    Prints out corrupted videos and skips them.

    Args:
        video_folder (str): The path to the folder containing video files.
        output_root (str): The root path to store the extracted frames.
    """
    video_folder = Path(video_folder)

    output_root = Path(output_root)

    if use_last_state:
        processed_videos = load_state()
    else:
        processed_videos = []
    
    video_files = [video for video in video_folder.glob('**/*') if video.suffix.lower() in VIDEO_EXTENSIONS and str(video) not in processed_videos]

    corrupted_videos = []

    try:

        # Use ThreadPoolExecutor to process video files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(extract_frames, video_file, output_root, fps, platform): video_file for video_file in
                       tqdm(video_files, desc="Processing Videos")}
            
            for future in concurrent.futures.as_completed(futures):
                if STOP_REQUESTED:
                    break
                video_file = futures[future]
                print(f"Processing {str(video_file)}")
                success = future.result()
                if success:
                    processed_videos.append(str(video_file))
                    save_state(processed_videos)
                else:
                    corrupted_videos.append(video_file)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving state...")
        save_state(processed_videos)
        return
    
    # Print out corrupted videos
    if corrupted_videos:
        DELETE_STATE_FILE = False
        print("Corrupted videos:")
        for video in corrupted_videos:
            print(video)

    # If all videos are processed, remove the state file
    if not STOP_REQUESTED and DELETE_STATE_FILE:
        try:
            print(f"Delete {STATE_FILE}")
            os.unlink(STATE_FILE)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    """
        The tree of files and folders before processing looks like this:
        
        -train_XXX
            -A
                -video_1.mp4
                -video_2.mp4
            -B
                -video_3.mp4
                -video_4.mp4
    
        after:
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
    """
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    fps = 12

    wsl = False

    platform = "mps"
    input_videos_path = '/Users/kennymccormick/tempForMe/2023-11-20/test'
    output_frames_path = f'/Users/kennymccormick/tempForMe/2023-11-20/out_{fps}fps'

    if wsl:
        input_videos_path = wsl_to_windows_path(input_videos_path)
        output_frames_path = wsl_to_windows_path(output_frames_path)

    use_last_state = True
    
    process_all_videos(input_videos_path, output_frames_path,fps,platform,use_last_state)
