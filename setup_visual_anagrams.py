import subprocess
import sys
import os

def setup_visual_anagrams():
    # Install the package in development mode
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "visual_anagrams"])
    
    # Install additional dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
        "torch", 
        "torchvision", 
        "diffusers", 
        "transformers",
        "einops"
    ])

if __name__ == "__main__":
    setup_visual_anagrams() 