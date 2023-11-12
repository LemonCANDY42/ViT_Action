# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 14:29
# @FileName: wsl_tools.py

def wsl_to_windows_path(wsl_path):
    if wsl_path.startswith("/mnt/"):
        drive_letter = wsl_path[5]
        windows_path = drive_letter.upper() + ':' + wsl_path[6:].replace('/', '\\')
        return windows_path
    else:
        return "The path does not appear to be a mounted Windows drive in WSL."
    
def windows_to_wsl_path(windows_path):
    # Check if the path starts with a drive letter and a colon
    if len(windows_path) > 1 and windows_path[1] == ':':
        drive_letter = windows_path[0].lower()  # Convert drive letter to lowercase
        wsl_path = "/mnt/" + drive_letter + windows_path[2:].replace('\\', '/')
        return wsl_path
    else:
        return "The path does not appear to be a valid Windows path."

if __name__ == "__main__":
    import sys

    # Check if there are command line arguments provided
    if len(sys.argv) > 1:
        # Convert the first command line argument
        print(windows_to_wsl_path(sys.argv[1]))
    else:
        # Or else, run a test conversion
        test_windows_path = "C:\\Users\\Example\\Documents"
        print(windows_to_wsl_path(test_windows_path))
