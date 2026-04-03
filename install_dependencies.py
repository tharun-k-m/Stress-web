import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    # Force install the specific Linux binary for MediaPipe
    install("mediapipe==0.10.11") 
    install("opencv-python-headless")
