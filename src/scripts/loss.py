import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.sipit.sipit_grad import sipit
import os

def get_wikipedia_samples(num_samples=5, max_char_length=60):
    print("Ładowanie strumienia danych z wikimedia/wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    samples = []
    iterator = iter(ds)
    
    while len(samples) < num_samples:
        try:
            row = next(iterator)
            text = row['text']
            
            lines = [line for line in text.split('\n') if len(line) > 20]
            
            if not lines:
                continue
                
            candidate = lines[0]
            
            if len(candidate) > max_char_length:
                candidate = candidate[:max_char_length].rsplit(' ', 1)[0]
            
            samples.append(candidate)
            
        except StopIteration:
            break
            
    return samples

def main():
    MODEL_NAME = "gpt2"
    STEPS = 1000
    LR = 0.03
    NUM_SAMPLES = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device).eval()

    dataset = get_wikipedia_samples(num_samples=NUM_SAMPLES, max_char_length=150)
    
    output_dir = "results/loss_plots_wiki"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Pobrano {len(dataset)} próbek z Wikipedii.")

    for idx, TEXT in enumerate(dataset):
        print(f"\n--- Przetwarzanie próbki {idx+1}/{len(dataset)} ---")
        
        inputs = tokenizer(TEXT, return_tensors="pt").to(device)
        
        if inputs.input_ids.shape[1] > 100:
            print(f"Skipping text {idx+1} (too long: {inputs.input_ids.shape[1]} tokens)")
            continue

        with torch.no_grad():
            out = model(**inputs)
            target_hidden = out.hidden_states[-1][0]

        print(f"Target text: '{TEXT}'")

        rec_ids, loss_history, rank_history = sipit(
            model=model,
            target_hidden_states=target_hidden,
            tokenizer=tokenizer,
            learning_rate=LR,
            num_optimization_steps=STEPS,
            loss_threshold=1e-4, 
            verbose=True,
            return_loss_history=True,
            max_candidates=100000
        )

        rec_text = tokenizer.decode(rec_ids)
        print(f"Reconstructed: {rec_text}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        steps_count = len(loss_history)
        if steps_count > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, steps_count))

            for i, (losses, token_id) in enumerate(zip(loss_history, rec_ids)):
                token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
                rank_val = rank_history[i]
                label = f"T{i+1}: '{token_str}' (R:{rank_val})"
                
                ax1.plot(losses, label=label, color=colors[i], linewidth=1.5, alpha=0.8)

        ax1.set_yscale('log')
        ax1.set_title(f"Optimization Landscape: '{TEXT[:40]}...'", fontsize=14)
        ax1.set_ylabel("MSE Loss (Log Scale)", fontsize=12)
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        if len(rec_ids) <= 20:
            ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        x_indices = np.arange(len(rank_history)) + 1
        bars = ax2.bar(x_indices, rank_history, color=colors[:len(rank_history)], alpha=0.7)
        
        for rect in bars:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

        ax2.set_title("Candidate Rank (Position in Euclidian Search)", fontsize=12)
        ax2.set_xlabel("Token Index", fontsize=12)
        ax2.set_ylabel("Rank (Lower is Better)", fontsize=12)
        ax2.set_xticks(x_indices)
        
        token_labels = [tokenizer.decode([tid]).replace('\n', '\\n') for tid in rec_ids]
        ax2.set_xticklabels(token_labels, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f"analysis_wiki_{idx+1}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to: {output_file}")
        plt.close()

if __name__ == "__main__":
    main()