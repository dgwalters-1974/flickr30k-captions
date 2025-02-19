import os
import shutil
import json
import tarfile
from datasets import load_dataset

# Define paths
BASE_DIR = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration"
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_JSON = os.path.join(DATA_DIR, "dataset_flickr30k.json")

# Hugging Face dataset identifier
HF_DATASET_ID = "HuggingFaceM4/flickr30k"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def download_flickr30k():
    """Download the Flickr30k dataset from Hugging Face."""
    print(f"Downloading Flickr30k dataset from Hugging Face: {HF_DATASET_ID}...")
    dataset = load_dataset(HF_DATASET_ID)
    print("Download completed!")
    return dataset

def save_images(dataset):
    """Save images from the dataset to the local directory."""
    print("Saving images...")
    for example in dataset["train"]:
        image_path = os.path.join(IMAGES_DIR, example["image_id"] + ".jpg")
        with open(image_path, "wb") as f:
            f.write(example["image"])  # Assuming image is in binary format
    print("Images saved successfully!")

def save_annotations(dataset):
    """Save dataset annotations as a JSON file."""
    print("Saving annotations...")
    annotations = []
    for example in dataset["train"]:
        annotations.append({
            "image_id": example["image_id"],
            "caption": example["caption"]
        })
    with open(ANNOTATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)
    print("Annotations saved successfully!")

def main():
    dataset = download_flickr30k()
    save_images(dataset)
    save_annotations(dataset)
    print("Flickr30k dataset setup completed successfully!")

if __name__ == "__main__":
    main()
