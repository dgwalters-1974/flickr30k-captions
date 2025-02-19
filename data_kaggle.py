import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Define paths
BASE_DIR = "/root/flickr30k-captions/"
DATA_DIR = os.path.join(BASE_DIR, "archive/")
IMAGES_DIR = os.path.join(DATA_DIR, "archive/flickr30k_images/")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def download_from_kaggle():
    """Download Flickr30k dataset from Kaggle"""
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    print("Downloading dataset...")
    api.dataset_download_files(
        'adityajn105/flickr30k',
        path=DATA_DIR,
        unzip=True
    )
    print("Download completed!")

if __name__ == "__main__":
    download_from_kaggle()