import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sipit_grad import sipit

def main():
    # Configuration
    MODEL_NAME = "gpt2"
    TEXT = "Hi, my name is Mikolaj"
    SEED = 42
    
    # Set device and seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model and Tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare Input
    inputs = tokenizer(TEXT, return_tensors="pt").to(device)
    orig_ids = inputs.input_ids[0].tolist()

    print("-" * 50)
    print(f"Original Text: {TEXT}")
    print(f"Original IDs:  {orig_ids}")
    print("-" * 50)

    # Get Target Hidden States
    with torch.no_grad():
        out = model(**inputs)
        # Taking the last layer's hidden states for the first sequence in batch
        target_hidden = out.hidden_states[-1][0]

    print(f"Target Hidden Shape: {target_hidden.shape}")
    print("-" * 50)

    # Run SIPIT Algorithm
    rec_ids = sipit(
        model=model, 
        target_hidden_states=target_hidden, 
        tokenizer=tokenizer,
        learning_rate=0.03,
        num_optimization_steps=1000,
        loss_threshold=1e-4,
        verbose=True
    )

    rec_text = tokenizer.decode(rec_ids)

    print("-" * 50)
    print(f"Reconstructed IDs:  {rec_ids}")
    print(f"Reconstructed Text: {rec_text}")
    print("-" * 50)
    

if __name__ == "__main__":
    main()
    