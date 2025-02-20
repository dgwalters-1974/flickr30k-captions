
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import TransformerDecoderCaption
from transformers import CLIPProcessor


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count_by_layer(model):
    """Print parameter count for each layer/component of the model."""
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            print(f"{name}: {param_count:,} parameters")
            total_params += param_count
    print(f"\nTotal trainable parameters: {total_params:,}")

if __name__ == "__main__":
    # Initialize model with same parameters as in your training
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    
    model = TransformerDecoderCaption(
        vocab_size=clip_processor.tokenizer.vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6
    )
    
    # Print total parameters
    total_params = count_parameters(model)
    print(f"\nModel Summary:")
    print("=" * 50)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Print detailed breakdown
    print("\nParameter count by layer:")
    print("=" * 50)
    print_parameter_count_by_layer(model)