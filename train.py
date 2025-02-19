import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import Flickr30kDataset
from model import TransformerDecoderCaption
import wandb
from tqdm.auto import tqdm

# ======= CONFIGURATION ======= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "/root/flickr30k-captions/archive/flickr30k_images"
CAPTIONS_FILE = "/root/flickr30k-captions/archive/captions.txt"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

# Initialize wandb
wandb.init(
    project="flickr30k-caption-generation",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "clip_model": CLIP_MODEL,
    }
)

# ======= LOAD DATASET ======= #
dataset = Flickr30kDataset(img_dir=IMG_DIR, captions_file=CAPTIONS_FILE, clip_model='openai/clip-vit-base-patch32', subset_size=None)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ======= MODEL SETUP ======= #
VOCAB_SIZE = dataset.processor.tokenizer.vocab_size
model = TransformerDecoderCaption(vocab_size=VOCAB_SIZE, d_model=512, num_heads=8, num_layers=6, dropout=0.1).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======= TRAINING LOOP ======= #
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    # Add progress bar for each epoch
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for idx, (image_embeddings, captions, masks) in enumerate(progress_bar):
        image_embeddings, captions, masks = image_embeddings.to(DEVICE), captions.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()

        tgt_seq = captions[:, :-1].to(torch.long).clamp(0, VOCAB_SIZE - 1)
        outputs = model(tgt_seq=tgt_seq, image_in=image_embeddings)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), captions[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        current_loss = loss.item()
        
        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Log batch-level metrics
        wandb.log({
            "batch_loss": current_loss,
            "batch": idx + epoch * len(dataloader)
        })

    epoch_loss = total_loss/len(dataloader)
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
    
    # Log epoch-level metrics
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": epoch_loss,
    })

# Save the trained model
torch.save(model.state_dict(), "caption_decoder_1.pth")
print("Model saved!")

# Close wandb run
wandb.finish()
