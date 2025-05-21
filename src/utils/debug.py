import cv2
import sys
import os

def write_debug_info():
    with open('debug_output.txt', 'w') as f:
        f.write("=== Debug Information ===\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"OpenCV version: {cv2.__version__}\n")
        f.write("OpenCV build information:\n")
        f.write(cv2.getBuildInformation())
        f.write("\nCurrent working directory: " + os.getcwd() + "\n")
        f.write("\nEnvironment variables:\n")
        for key, value in os.environ.items():
            f.write(f"{key}={value}\n")

if __name__ == "__main__":
    print("Starting debug info collection...")
    write_debug_info()
    print("Debug info written to debug_output.txt")