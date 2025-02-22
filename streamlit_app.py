import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from model import TransformerDecoderCaption

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECODER_PATH = "/Users/dgwalters/ML Projects/MLX-4/CaptionGeneration/caption_decoder_20250221_151104.pth"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def load_models(clip_model_name=CLIP_MODEL_NAME, decoder_path=DECODER_PATH, device="cuda"):
    """Load models using exact same logic as inference.py"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    
    decoder = TransformerDecoderCaption(
        vocab_size=clip_processor.tokenizer.vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    
    return clip_processor, clip_model, decoder, device

def get_clip_embedding(image_path, clip_processor, clip_model, device):
    """Extract CLIP embedding with same debug prints as inference.py"""
    st.write(f"📷 Processing Image: {os.path.basename(image_path)}")
    st.write(f"Processing Image: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    
    st.write(f"CLIP embedding dimension: {image_embedding.shape}")
    return image_embedding

def generate_caption(image_embedding, decoder, clip_processor, max_length=50):
    """Generate caption using same logic as inference.py"""
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
    # Save uploaded file
    temp_dir = "unseen_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display image - fixed deprecation warning
    image = Image.open(temp_path)
    st.image(image, use_container_width=False)  # Display at original size
    
    # Generate caption with debug info
    with st.spinner("Generating caption..."):
        image_embedding = get_clip_embedding(temp_path, clip_processor, clip_model, device)
        st.write(f"Image embedding shape before generation: {image_embedding.shape}")
        
        caption = generate_caption(image_embedding, decoder, clip_processor)
        st.write("\n📝 Generated Caption:", caption)
    
    # Clean up
    os.remove(temp_path)

# Add some instructions at the bottom
st.markdown("""
---
### Instructions:
1. Click 'Browse files' or drag and drop an image
2. The model will automatically generate a caption
3. Try different images to see how the model performs!
""")
