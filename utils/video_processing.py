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

STATE_FILE = './resume_state.txt'
stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, stopping...")

def save_state(state_data):
    with open(STATE_FILE, 'w') as f:
        json.dump(state_data, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return []

def extract_frames(video_path, output_root,fps:int=1,wsl=False):
    """
    Extract frames from a video using FFmpeg with NVDEC GPU-acceleration.
    If the video is corrupted or the process fails, it returns False.

    Args:
        video_path (Path): The path to the video file.
        output_root (Path): The root path to store the extracted frames.

    Returns:
        bool: True if frames were extracted successfully, False otherwise.
    """
    output_path = output_root / (video_path.stem+f"_{fps}fps")
    output_path.mkdir(parents=True, exist_ok=True)
    
    if wsl:
        ffmpeg_path = '/mnt/d/ffmpeg-6.0-full_build/bin/ffmpeg.exe'
    else:
        ffmpeg_path = '/usr/local/bin/ffmpeg'
        
    
    # FFmpeg command using NVDEC for GPU acceleration
    cmd = [
        ffmpeg_path,
        '-hwaccel', 'nvdec',  # Use CUDA for GPU acceleration
        '-hwaccel_device', '0',
        # '-hwaccel_output_format', 'cuda',  # Set the hardware acceleration output format to CUDA
        # '-c:v', 'h264_nvenc',  # Use the h264_nvenc decoder
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        str(output_path / 'frame_%04d.jpg')
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


def process_all_videos(video_folder, output_root,fps:int = 1 ,wsl=False):
    """
    Process all video files in a directory using multithreading.
    Prints out corrupted videos and skips them.

    Args:
        video_folder (str): The path to the folder containing video files.
        output_root (str): The root path to store the extracted frames.
    """
    video_folder = Path(video_folder)
    output_root = Path(output_root)
    video_files = []
    
    processed_videos = load_state()
    
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')
    video_files = [video for video in video_folder.glob('**/*') if video.suffix.lower() in video_extensions and str(video) not in processed_videos]
    
    corrupted_videos = []
    
    try:

        # Use ThreadPoolExecutor to process video files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(extract_frames, video_file, output_root, fps, wsl): video_file for video_file in
                       tqdm(video_files, desc="Processing Videos")}
            
            for future in concurrent.futures.as_completed(futures):
                if stop_requested:
                    break
                print(f"Processing {str(video_file)}")
                video_file = futures[future]
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
        print("Corrupted videos:")
        for video in corrupted_videos:
            print(video)
            
    # If all videos are processed, remove the state file
    if not stop_requested:
        try:
            STATE_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    fps = 8

    wsl = False
    input_videos_path = '/mnt/g/ava/AVADataset/ava_train'
    output_frames_path = '/mnt/g/ava/AVADataset/train_frames'

    if wsl:
        input_videos_path = wsl_to_windows_path(input_videos_path)
        output_frames_path = wsl_to_windows_path(output_frames_path)
    
    process_all_videos(input_videos_path, output_frames_path,fps,wsl)
