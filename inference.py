import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
from model import TransformerDecoderCaption  # Import your trained model
import os


def load_models(clip_model_name="openai/clip-vit-base-patch32", decoder_path="caption_decoder.pth", device="cuda"):
    """
    Load the CLIP model for image embeddings and the trained Transformer decoder for caption generation.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load CLIP Processor & Model
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    
    # Load trained Transformer Decoder with d_model=512 to match saved weights
    decoder = TransformerDecoderCaption(
        vocab_size=clip_processor.tokenizer.vocab_size,
        d_model=512,  # Ensure consistent embedding dimension
        num_heads=8,
        num_layers=6
    ).to(device)
    
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    
    return clip_processor, clip_model, decoder, device


def get_clip_embedding(image_path, clip_processor, clip_model, device):
    """Processes an image and extracts its CLIP embedding."""
    print(f"Processing Image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)  # Output shape: [1, 512]
    
    print(f"CLIP embedding dimension: {image_embedding.shape}")  # Debug print
    return image_embedding  # Ensure no projection to 768


def generate_caption(image_embedding, decoder, clip_processor, max_length=50):
    """
    Generates a caption for an image using greedy decoding.
    """
    device = image_embedding.device
    sos_token = clip_processor.tokenizer.bos_token_id  # Start-of-sentence token
    eos_token = clip_processor.tokenizer.eos_token_id  # End-of-sentence token

    tgt_seq = torch.tensor([[sos_token]], device=device)  # Start with <SOS>
    
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(tgt_seq, image_in=image_embedding)  # Predict next token
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)  # Get most probable token

        # Append next token to sequence
        tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        
        # Stop if <EOS> token is generated
        if next_token.item() == eos_token:
            break

    # Decode token IDs into words
    caption = clip_processor.tokenizer.decode(tgt_seq.squeeze().tolist(), skip_special_tokens=True)
    return caption


def main(image_path, decoder_path="caption_decoder.pth", clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
    """
    Runs the full inference pipeline to generate a caption for an image.
    """
    print(f"\n📷 Processing Image: {os.path.basename(image_path)}")
    
    # Load models
    clip_processor, clip_model, decoder, device = load_models(clip_model_name, decoder_path, device)

    # Extract image embeddings
    image_embedding = get_clip_embedding(image_path, clip_processor, clip_model, device)
    print(f"Image embedding shape before generation: {image_embedding.shape}")  # Debug print

    # Generate caption
    caption = generate_caption(image_embedding, decoder, clip_processor)

    print(f"\n📝 Generated Caption: {caption}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained Transformer decoder.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--decoder_path", type=str, default="caption_decoder.pth", help="Path to trained decoder model.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model variant.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu).")

    args = parser.parse_args()
    main(args.image_path, args.decoder_path, args.clip_model, args.device)


# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import argparse
# from model import TransformerDecoderCaption  # Import your trained model
# import os
# from transformers import CLIPProcessor
# import torch.nn as nn

# # ===========================
# # 1️⃣ Load CLIP & Decoder Model
# # ===========================

# # Add this projection layer to inference.py
# class CLIPEmbeddingProjector(nn.Module):
#     def __init__(self, input_dim=512, output_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.proj(x)


# def load_models(clip_model_name="openai/clip-vit-base-patch32", decoder_path="caption_decoder.pth", device="cuda"):
#     """
#     Load the CLIP model for image embeddings and the trained Transformer decoder for caption generation.
#     """
#     device = torch.device(device if torch.cuda.is_available() else "cpu")
    
#     # Load CLIP Processor & Model
#     clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
#     clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    
#     # Load trained Transformer Decoder with d_model=512 to match saved weights
#     decoder = TransformerDecoderCaption(
#         vocab_size=clip_processor.tokenizer.vocab_size,
#         d_model=512,  # Changed to match saved model dimensions
#         num_heads=8,
#         num_layers=6
#     ).to(device)
    
#     decoder.load_state_dict(torch.load(decoder_path, map_location=device))
#     decoder.eval()
    
#     return clip_processor, clip_model, decoder, device

# # ===========================
# # 2️⃣ Extract CLIP Image Embeddings
# # ===========================
# def get_clip_embedding(image_path, clip_processor, clip_model, device):
#     """Processes an image and extracts its CLIP embedding, then projects it to 768 dimensions."""
#     print(f"Inference CLIP model: {clip_model.config.model_type}")
#     image = Image.open(image_path).convert("RGB")
#     inputs = clip_processor(images=image, return_tensors="pt").to(device)

#     with torch.no_grad():
#         image_embedding = clip_model.get_image_features(**inputs)  # Shape: [1, 512]
#         print(f"CLIP embedding dimension: {image_embedding.shape}")  

#     # Apply projection to match decoder input size
#     projector = CLIPEmbeddingProjector(input_dim=512, output_dim=768).to(device)
#     image_embedding = projector(image_embedding)  # Now shape [1, 768]

#     return image_embedding


# # ===========================
# # 3️⃣ Caption Generation (Greedy Decoding)
# # ===========================
# def generate_caption(image_embedding, decoder, clip_processor, max_length=50):
#     """
#     Generates a caption for an image using greedy decoding.
#     """
#     device = image_embedding.device
#     sos_token = clip_processor.tokenizer.bos_token_id  # Start-of-sentence token
#     eos_token = clip_processor.tokenizer.eos_token_id  # End-of-sentence token

#     tgt_seq = torch.tensor([[sos_token]], device=device)  # Start with <SOS>
    
#     for _ in range(max_length):
#         with torch.no_grad():
#             output = decoder(tgt_seq, image_embedding)  # Predict next token
#             next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)  # Get most probable token

#         # Append next token to sequence
#         tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        
#         # Stop if <EOS> token is generated
#         if next_token.item() == eos_token:
#             break

#     # Decode token IDs into words
#     caption = clip_processor.tokenizer.decode(tgt_seq.squeeze().tolist(), skip_special_tokens=True)
#     return caption

# # ===========================
# # 4️⃣ Main Function for Inference
# # ===========================
# def main(image_path, decoder_path="caption_decoder.pth", clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
#     """
#     Runs the full inference pipeline to generate a caption for an image.
#     """
#     print(f"\n📷 Processing Image: {os.path.basename(image_path)}")
    
#     # Load models
#     clip_processor, clip_model, decoder, device = load_models(clip_model_name, decoder_path, device)

#     # Extract image embeddings
#     image_embedding = get_clip_embedding(image_path, clip_processor, clip_model, device)
#     print(f"Image embedding shape before generation: {image_embedding.shape}")  # Debug print

#     # Generate caption
#     caption = generate_caption(image_embedding, decoder, clip_processor)

#     print(f"\n📝 Generated Caption: {caption}\n")

# # ===========================
# # 5️⃣ Run the Script from CLI
# # ===========================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained Transformer decoder.")
#     parser.add_argument("image_path", type=str, help="Path to the image file.")
#     parser.add_argument("--decoder_path", type=str, default="caption_decoder.pth", help="Path to trained decoder model.")
#     parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model variant.")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu).")

#     args = parser.parse_args()
#     main(args.image_path, args.decoder_path, args.clip_model, args.device)