import streamlit as st
import torch
import os
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model import TransformerDecoderCaption  # Import your trained model

# ======= CONFIGURATION ======= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "archive/flickr30k_images/"
DECODER_PATH = "caption_decoder.pth"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ======= LOAD MODELS ======= #
@st.cache_resource()
def load_models():
    """Load the CLIP model and Transformer decoder."""
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    
    decoder = TransformerDecoderCaption(
        vocab_size=clip_processor.tokenizer.vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6
    ).to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    decoder.eval()
    
    return clip_processor, clip_model, decoder

clip_processor, clip_model, decoder = load_models()

# ======= FUNCTION TO GENERATE CAPTION ======= #
def get_clip_embedding(image, clip_processor, clip_model):
    """Extract CLIP image embeddings."""
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)  # Output shape: [1, 512]
    return image_embedding

def generate_caption(image_embedding, decoder, clip_processor, max_length=50):
    """Generate a caption using greedy decoding."""
    sos_token = clip_processor.tokenizer.bos_token_id
    eos_token = clip_processor.tokenizer.eos_token_id
    tgt_seq = torch.tensor([[sos_token]], device=DEVICE)
    
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(tgt_seq, image_in=image_embedding)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
        
        tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        if next_token.item() == eos_token:
            break
    
    return clip_processor.tokenizer.decode(tgt_seq.squeeze().tolist(), skip_special_tokens=True)

# ======= STREAMLIT APP ======= #
st.title("Flickr30k Image Captioning App")

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if st.button("Get Random Image"):
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        random_image = random.choice(image_files)
        st.session_state.current_image = os.path.join(IMG_DIR, random_image)

if st.session_state.current_image:
    st.image(st.session_state.current_image, caption="Selected Image", use_container_width=True)
    
    image = Image.open(st.session_state.current_image).convert("RGB")
    image_embedding = get_clip_embedding(image, clip_processor, clip_model)
    
    if st.button("Generate Caption"):
        caption = generate_caption(image_embedding, decoder, clip_processor)
        st.write(f"### 📝 Caption: {caption}")
