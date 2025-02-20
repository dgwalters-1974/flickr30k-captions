import torch
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import os
from torch.utils.data import Dataset

class Flickr30kDataset(Dataset):
    def __init__(self, img_dir, captions_file, clip_model='openai/clip-vit-base-patch32', subset_size=None, random_subset=True):
        self.img_dir = img_dir
        
        # Read and process captions file
        self.captions_df = pd.read_csv(captions_file,
                                     sep=',',  # Use comma as separator
                                     names=['image_name', 'comment_number', 'comment'],  # Match file's column names
                                     skiprows=1)  # Skip the header row
        
        # Clean up image names by removing any whitespace
        self.captions_df['image_name'] = self.captions_df['image_name'].str.strip()
        
        # Store the last 100 rows separately
        self.unseen_captions_df = self.captions_df.iloc[-100:]
        
        # Ignore the last 100 rows in main dataset
        self.captions_df = self.captions_df.iloc[:-100]
        
        if subset_size:
            if random_subset:
                self.captions_df = self.captions_df.sample(n=subset_size, random_state=42).reset_index(drop=True)
            else:
                self.captions_df = self.captions_df.iloc[:subset_size].reset_index(drop=True)
        
        # Load CLIP Processor and Model
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model).eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)

    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        item = self.captions_df.iloc[idx]
        
        # Handle missing or NaN captions
        if 'comment' not in item or pd.isna(item['comment']):
            caption = "No caption available"  # Default caption for missing/nan values
        else:
            if isinstance(item['comment'], (list, tuple)):
                caption = item['comment'][0] if item['comment'] else "No caption available"
            else:
                caption = str(item['comment'])
        
        if caption == "nan" or pd.isna(caption):
            caption = "No caption available"
        
        img_path = os.path.join(self.img_dir, item['image_name'])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use get_image_features instead of vision_model
            image_features = self.clip_model.get_image_features(**image_inputs)

        # Debug prints
        
        
        caption_inputs = self.processor(
            text=caption,
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors='pt'
        )

        caption_ids = caption_inputs["input_ids"].squeeze(0)
        caption_mask = caption_inputs["attention_mask"].squeeze(0)

        return image_features.squeeze(0), caption_ids, caption_mask

    def save_unseen_images(self, save_dir):
        unseen_images_dir = os.path.join(save_dir, "UnseenImages", "images")
        unseen_captions_file = os.path.join(save_dir, "UnseenImages", "captions.txt")
        
        os.makedirs(unseen_images_dir, exist_ok=True)
        
        with open(unseen_captions_file, "w") as f:
            for _, row in self.unseen_captions_df.iterrows():
                img_path = os.path.join(self.img_dir, row['image_name'])
                save_path = os.path.join(unseen_images_dir, row['image_name'])
                
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    image.save(save_path)
                
                f.write(f"{row['image_name']},{row['comment_number']},{row['comment']}\n")
