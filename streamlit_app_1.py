import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from model import TransformerDecoderCaption

# Configuration - use same path as inference.py
DECODER = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/caption_decoder_20250220_165701.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

@st.cache_resource
def load_models(clip_model_name=CLIP_MODEL_NAME, decoder_path=DECODER, device="cuda"):
    """Exact same loading function as inference.py"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    
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
    """Exact same embedding function as inference.py"""
    print(f"Processing Image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    
    print(f"CLIP embedding dimension: {image_embedding.shape}")
    return image_embedding

def generate_caption(image_embedding, decoder, clip_processor, max_length=50):
    """Exact same caption generation as inference.py"""
    device = image_embedding.device
    sos_token = clip_processor.tokenizer.bos_token_id
    eos_token = clip_processor.tokenizer.eos_token_id

    tgt_seq = torch.tensor([[sos_token]], device=device)
    
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(tgt_seq, image_in=image_embedding)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)

        tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        if next_token.item() == eos_token:
            break

    return clip_processor.tokenizer.decode(tgt_seq.squeeze().tolist(), skip_special_tokens=True)

# Streamlit UI
st.title("Image Caption Generation")

# Load models
try:
    clip_processor, clip_model, decoder, device = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily to use same path-based logic as inference.py
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display image
    st.image(uploaded_file, use_container_width=True)
    
    # Generate caption using exact same process as inference.py
    with st.spinner("Generating caption..."):
        st.write(f"\n📷 Processing Image: {os.path.basename(temp_path)}")
        
        # Get embedding
        image_embedding = get_clip_embedding(temp_path, clip_processor, clip_model, device)
        st.write(f"Image embedding shape: {image_embedding.shape}")
        
        # Generate caption
        caption = generate_caption(image_embedding, decoder, clip_processor)
        
        # Display caption
        st.write("\n📝 Generated Caption:")
        st.markdown(f"**{caption}**")
    
    # Clean up temp file
    os.remove(temp_path)

# Add some instructions at the bottom
st.markdown("""
---
### Instructions:
1. Drag and drop image onto the top bit...
2. The model will  generate a caption
3. Be amazed
""")