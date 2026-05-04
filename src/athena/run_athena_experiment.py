import argparse
import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sipit_core import sipit_hpc
from datetime import datetime
from difflib import SequenceMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="SIPIT Athena Experiment")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results_athena", help="Output directory")
    parser.add_argument("--max_len", type=int, default=150, help="Max char length of text")
    return parser.parse_args()

def compute_text_metrics(original, reconstructed):
    """
    Computes text similarity metrics without external dependencies (like Levenshtein).
    """
    matcher = SequenceMatcher(None, original, reconstructed)
    char_similarity = matcher.ratio()

    ref_words = original.split()
    hyp_words = reconstructed.split()
    
    set_ref = set(ref_words)
    set_hyp = set(hyp_words)
    intersection = len(set_ref.intersection(set_hyp))
    union = len(set_ref.union(set_hyp))
    word_jaccard = intersection / union if union > 0 else 0.0
    
    return {
        "char_similarity": char_similarity,
        "word_jaccard": word_jaccard
    }

def save_plot(trajectory, original_text, reconstructed_text, save_path):
    steps_data = [t['loss_history'] for t in trajectory]
    ranks = [t['rank'] for t in trajectory]
    tokens = [t['token_str'] for t in trajectory]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_data)))
    for i, losses in enumerate(steps_data):
        ax1.plot(losses, color=colors[i], alpha=0.6, linewidth=1.5)
    
    ax1.set_yscale('log')
    sim_ratio = SequenceMatcher(None, original_text, reconstructed_text).ratio()
    ax1.set_title(f"Loss Landscape (Sim: {sim_ratio:.2f})\nOrig: {original_text[:50]}...", fontsize=10)
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)

    x = np.arange(len(ranks))
    bars = ax2.bar(x, ranks, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('\n', '\\n') for t in tokens], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Rank (Log Scale)")
    ax2.set_yscale('symlog') 
    ax2.grid(True, axis='y', alpha=0.3)
    
    for rect in bars:
        h = rect.get_height()
        if h > 0:
            ax2.text(rect.get_x() + rect.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.out_dir, f"{args.model}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"--- Starting Experiment: {exp_dir} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()

    print("Loading local dataset...")
    data_path = os.path.join(os.environ["SCRATCH"], "inverse_embeddings", "wiki_local.json")
    
    with open(data_path, "r") as f:
        local_samples = json.load(f)

    iterator = iter([{'text': txt} for txt in local_samples])
    processed = 0
    
    while processed < args.samples:
        try:
            row = next(iterator)
            text = row['text'].split('\n')[0]
            if len(text) < 30: 
                candidates = [s for s in row['text'].split('.') if len(s) > 50]
                if candidates: text = candidates[0] + "."
                else: continue
            
            text = text[:args.max_len]
            inputs = tokenizer(text, return_tensors="pt").to(device)
            if inputs.input_ids.shape[1] > 60: continue

            print(f"[{processed+1}/{args.samples}] Processing: {text[:30]}...")

            with torch.no_grad():
                out = model(**inputs)
                target_hidden = out.hidden_states[-1][0]

            result = sipit_hpc(
                model=model,
                target_hidden_states=target_hidden,
                tokenizer=tokenizer,
                lr=args.lr,
                steps=args.steps,
                verbose=False
            )

            metrics = compute_text_metrics(text, result['recovered_text'])
            
            ranks = [t['rank'] for t in result['trajectory']]
            avg_rank = sum(ranks) / len(ranks) if ranks else 0
            
            summary_entry = {
                "id": f"sample_{processed:03d}",
                "original_text": text,
                "reconstructed_text": result['recovered_text'],
                "char_similarity": metrics['char_similarity'], 
                "word_jaccard": metrics['word_jaccard'],
                "avg_rank": avg_rank,
                "token_count": len(ranks)
            }
            results_summary.append(summary_entry)

            result.update(metrics) 
            result['original_text'] = text
            
            with open(os.path.join(exp_dir, f"sample_{processed:03d}.json"), "w") as f:
                json.dump(result, f)

            save_plot(result['trajectory'], text, result['recovered_text'], 
                     os.path.join(exp_dir, f"sample_{processed:03d}.png"))

            processed += 1
            
        except StopIteration:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    df = pd.DataFrame(results_summary)
    csv_path = os.path.join(exp_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nExperiment finished.")
    print(f"Average Similarity: {df['char_similarity'].mean():.4f}")
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()