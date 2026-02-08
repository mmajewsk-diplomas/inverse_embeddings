import argparse
import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sipit_w_rank import sipit_hpc
from datetime import datetime
from difflib import SequenceMatcher
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="SIPIT Short Text Analysis")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results_short", help="Output directory")
    parser.add_argument("--min_len", type=int, default=25, help="Min char length (to avoid garbage)")
    parser.add_argument("--max_len", type=int, default=70, help="Max char length")
    return parser.parse_args()

def save_plot(trajectory, original_text, reconstructed_text, save_path):
    steps_data = [t['loss_history'] for t in trajectory]
    final_ranks = [t['rank'] for t in trajectory]
    tokens = [t['token_str'] for t in trajectory]
    rank_evo_vals = [t['rank_evolution'] for t in trajectory]
    rank_evo_steps = [t['rank_evolution_steps'] for t in trajectory]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1.5, 1, 1.5]})
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_data)))
    
    # LOSS
    for i, losses in enumerate(steps_data):
        ax1.plot(losses, color=colors[i], alpha=0.6, linewidth=1.5)
    ax1.set_yscale('log')
    sim_ratio = SequenceMatcher(None, original_text, reconstructed_text).ratio()
    ax1.set_title(f"Loss (Sim: {sim_ratio:.2f}) | '{original_text}'", fontsize=11, fontweight='bold')
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)

    # FINAL RANK
    x = np.arange(len(final_ranks))
    bars = ax2.bar(x, final_ranks, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('\n', '\\n') for t in tokens], rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel("Final Rank")
    ax2.set_yscale('symlog') 
    ax2.grid(True, axis='y', alpha=0.3)
    for rect in bars:
        h = rect.get_height()
        if h > 0: ax2.text(rect.get_x() + rect.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=8)

    for i, (r_vals, r_steps) in enumerate(zip(rank_evo_vals, rank_evo_steps)):
        token_label = tokens[i].replace('\n', '\\n')
        ax3.plot(r_steps, r_vals, color=colors[i], alpha=0.7, linewidth=2, label=f"T{i}:{token_label}")
        if r_vals: ax3.scatter(r_steps[-1], r_vals[-1], color=colors[i], s=15)

    ax3.set_yscale('log')
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Rank")
    ax3.set_title("Rank Convergence", fontsize=10)
    ax3.grid(True, which="both", alpha=0.2)
    ax3.set_ylim(0.5, 60000)
    
    if len(tokens) < 15:
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.out_dir, f"{args.model}_short_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"--- Short Text Analysis: {exp_dir} ---")
    print(f"Target length: {args.min_len}-{args.max_len} chars")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    iterator = iter(ds)
    
    results_summary = []
    pbar = tqdm(total=args.samples, desc="Searching & Processing")
    
    processed = 0
    while processed < args.samples:
        row = next(iterator)
        full_text = row['text']
        
        sentences = full_text.split('.')
        target_text = None
        
        for s in sentences:
            s_clean = s.strip()
            if args.min_len <= len(s_clean) <= args.max_len:
                if ' ' in s_clean:
                    target_text = s_clean + "."
                    break
        
        if target_text is None:
            continue

        inputs = tokenizer(target_text, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] > 20: 
            continue 

        with torch.no_grad():
            out = model(**inputs)
            target_hidden = out.hidden_states[-1][0]

        result = sipit_hpc(
            model=model,
            target_hidden_states=target_hidden,
            tokenizer=tokenizer,
            lr=args.lr,
            steps=args.steps,
            rank_check_interval=10,
            verbose=True
        )

        sim_ratio = SequenceMatcher(None, target_text, result['recovered_text']).ratio()
        ranks = [t['rank'] for t in result['trajectory']]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        sample_id = f"sample_{processed:03d}"
        result['original_text'] = target_text
        result['similarity'] = sim_ratio
        
        with open(os.path.join(exp_dir, f"{sample_id}.json"), "w") as f:
            json.dump(result, f)
        
        save_plot(result['trajectory'], target_text, result['recovered_text'], 
                    os.path.join(exp_dir, f"{sample_id}.png"))

        results_summary.append({
            "id": sample_id,
            "text": target_text,
            "similarity": sim_ratio,
            "avg_rank": avg_rank
        })

        processed += 1
        pbar.update(1)
        pbar.set_postfix({"Last Sim": f"{sim_ratio:.2f}"})
            

    pbar.close()
    
    df = pd.DataFrame(results_summary)
    csv_path = os.path.join(exp_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Done. Check {exp_dir}")

if __name__ == "__main__":
    main()