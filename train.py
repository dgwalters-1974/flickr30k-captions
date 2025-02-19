import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import Flickr30kDataset
from model import TransformerDecoderCaption

# ======= CONFIGURATION ======= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/archive/flickr30k_images"
CAPTIONS_FILE = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/archive/captions.txt"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

# ======= LOAD DATASET ======= #
dataset = Flickr30kDataset(img_dir=IMG_DIR, captions_file=CAPTIONS_FILE, clip_model='openai/clip-vit-base-patch32', subset_size=1000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ======= MODEL SETUP ======= #
VOCAB_SIZE = dataset.processor.tokenizer.vocab_size  # Corrected way to access vocab size
model = TransformerDecoderCaption(vocab_size=VOCAB_SIZE, d_model=512, num_heads=8, num_layers=6, dropout=0.1).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======= TRAINING LOOP ======= #
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for idx, (image_embeddings, captions, masks) in enumerate(dataloader):
        image_embeddings, captions, masks = image_embeddings.to(DEVICE), captions.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()

        # Ensure captions are valid token indices
        tgt_seq = captions[:, :-1].to(torch.long).clamp(0, VOCAB_SIZE - 1)

        # Compute model outputs
        outputs = model(tgt_seq=tgt_seq, image_in=image_embeddings)

        # Compute loss
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "caption_decoder.pth")
print("Model saved!")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from data import Flickr30kDataset
# from model import TransformerDecoderCaption
# from data import Flickr30kDataset

# import torch.nn as nn

# # Add this projection layer to inference.py
# class CLIPEmbeddingProjector(nn.Module):
#     def __init__(self, input_dim=512, output_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.proj(x)


# # ======= CONFIGURATION ======= #
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMG_DIR = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/archive/flickr30k_images"
# CAPTIONS_FILE = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/archive/captions.txt"
# CLIP_MODEL = "openai/clip-vit-base-patch32"
# BATCH_SIZE = 64
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 1

# # ======= LOAD DATASET ======= #
# dataset = Flickr30kDataset(img_dir=IMG_DIR, captions_file=CAPTIONS_FILE, clip_model='openai/clip-vit-base-patch32', subset_size=1000)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# # ======= MODEL SETUP ======= #
# VOCAB_SIZE = dataset.processor.tokenizer.vocab_size  # Corrected way to access vocab size
# model = TransformerDecoderCaption(vocab_size=VOCAB_SIZE, d_model=512, num_heads=8, num_layers=6, dropout=0.1).to(DEVICE)

# criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # ======= TRAINING LOOP ======= #
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     total_loss = 0

#     projector = CLIPEmbeddingProjector(input_dim=512, output_dim=768).to(DEVICE)
    
#     for idx, (image_embeddings, captions, masks) in enumerate(dataloader):
#         image_embeddings, captions, masks = image_embeddings.to(DEVICE), captions.to(DEVICE), masks.to(DEVICE)

#         # Apply projection to image embeddings
#         image_embeddings = projector(image_embeddings)  # Converts [batch_size, 512] -> [batch_size, 768]

#         optimizer.zero_grad()

#         # Ensure captions are valid token indices
#         tgt_seq = captions[:, :-1].to(torch.long).clamp(0, VOCAB_SIZE - 1)

#         # Compute model outputs
#         outputs = model(tgt_seq=tgt_seq, encoder_output=image_embeddings)

#         # Compute loss
#         loss = criterion(outputs.reshape(-1, VOCAB_SIZE), captions[:, 1:].reshape(-1))
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "caption_decoder.pth")
# print("Model saved!")
