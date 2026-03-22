import argparse
import torch
import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sipit_w_rank import sipit
from datetime import datetime
from difflib import SequenceMatcher
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="SIPIT Batch Processing")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to process")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="results_batch")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "l1", "cosine", "dot"], help="Metrics")
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.out_dir, f"{args.model}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(ds)
    
    results_summary = []
    pbar = tqdm(total=args.samples, desc="Processing")
    
    processed = 0
    while processed < args.samples:
        try:
            row = next(iterator)
            full_text = row['text']
            
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            
            if len(full_tokens) < 20: 
                continue
                
            target_token_ids = full_tokens[:15]
            text = tokenizer.decode(target_token_ids)
            
            inputs_ids = torch.tensor([target_token_ids]).to(device)
            
            with torch.no_grad():
                out = model(inputs_ids)
                target_hidden = out.hidden_states[-1][0] 

            result = sipit(
                model=model,
                target_hidden_states=target_hidden,
                target_ids=inputs_ids[0],
                tokenizer=tokenizer,
                steps=args.steps,
                verbose=False,
                metric=args.metric
            )

            sim_ratio = SequenceMatcher(None, text, result['recovered_text']).ratio()
            
            sample_id = f"sample_{processed:04d}"
            result['original_text'] = text
            result['similarity'] = sim_ratio
            
            with open(os.path.join(exp_dir, f"{sample_id}.json"), "w") as f:
                json.dump(result, f)

            results_summary.append({
                "id": sample_id,
                "original": text,
                "recovered": result['recovered_text'],
                "similarity": sim_ratio
            })

            processed += 1
            pbar.update(1)
            pbar.set_postfix({"Sim": f"{sim_ratio:.2f}"})

        except StopIteration:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    pbar.close()
    
    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(exp_dir, "summary.csv"), index=False)
    print(f"Batch processing done. Results in {exp_dir}")

if __name__ == "__main__":
    main()