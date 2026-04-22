import argparse
import sys
import os
import json
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SIPIT'))

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import transformers
transformers.logging.set_verbosity_error()

import torch
import pandas as pd
from datasets import load_dataset
from difflib import SequenceMatcher
from tqdm import tqdm
from datetime import datetime

from src.utils.model import setup
from src.algorithm.SIPIT import SIPIT as _SIPIT


class SIPITWithRecovery(_SIPIT):
    """Thin wrapper that also exposes recovered token ids."""

    def inversion_attack(self, *, input_ids, model, tokenizer, layer_idx,
                         step_size, seed=8, **kwargs):
        from src.utils.utils import set_seed
        set_seed(seed)

        new_input_ids = (
            torch.cat([
                torch.tensor([self.special_start_token_id],
                             dtype=torch.long, device=input_ids.device),
                input_ids
            ], dim=0)
            if self.special_start_token_id is not None
            else input_ids
        )

        target_hidden_states = self.target_extraction_fn(
            new_input_ids, model, layer_idx)
        start_from = new_input_ids.size(0) - input_ids.size(0)

        result = self.find_prompt(
            model=model, tokenizer=tokenizer, layer_idx=layer_idx,
            target_hidden_states=target_hidden_states[start_from:],
            step_size=step_size, **kwargs)

        inversion_time, discovered_ids, timesteps, times = result

        if inversion_time is None:
            return False, None, None, None, None

        match = all(x == y for x, y in
                    zip(input_ids.tolist(), discovered_ids[start_from:]))
        return match, inversion_time, timesteps, times, discovered_ids[start_from:]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIPIT Reference Batch Time Analysis")
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=15)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--precision", type=int, default=32,
                        choices=[4, 8, 16, 32])
    parser.add_argument("--out_dir", type=str, default="results_batch")
    parser.add_argument("--seed", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split('/')[-1]
    exp_dir = os.path.join(
        args.out_dir, f"sipit_ref_{model_short}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    model, tokenizer, device, layer_idx = setup(
        model_id=args.model,
        precision=args.precision,
        layer_idx=args.layer_idx,
        print_stats=True,
    )

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    algorithm = SIPITWithRecovery(log_dir=log_dir, log_name=None, use_scheduler=False)

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
            if len(full_tokens) < args.tokens + 5:
                continue

            target_token_ids = full_tokens[:args.tokens]
            text = tokenizer.decode(target_token_ids)
            input_ids = torch.tensor(
                target_token_ids, dtype=torch.long, device=device)

            out = algorithm.inversion_attack(
                input_ids=input_ids,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                step_size=args.step_size,
                seed=args.seed,
            )

            match, inversion_time, timesteps, times, recovered_ids = out

            if inversion_time is None:
                print(f"\nSample {processed}: inversion diverged, skipping")
                continue

            recovered_text = tokenizer.decode(recovered_ids)
            sim_ratio = SequenceMatcher(None, text, recovered_text).ratio()

            per_token_times = times if times else []
            per_token_steps = timesteps if timesteps else []
            avg_token_time_s = (sum(per_token_times) / len(per_token_times)
                                if per_token_times else 0.0)
            total_time_s = inversion_time

            sample_id = f"sample_{processed:04d}"

            sample_result = {
                "sample_id": sample_id,
                "original_text": text,
                "original_ids": target_token_ids,
                "recovered_text": recovered_text,
                "recovered_ids": list(recovered_ids),
                "match": bool(match),
                "similarity": sim_ratio,
                "total_time_s": total_time_s,
                "per_token_times": per_token_times,
                "per_token_steps": per_token_steps,
                "avg_token_time_s": avg_token_time_s,
                "num_tokens": len(target_token_ids),
            }

            with open(os.path.join(exp_dir, f"{sample_id}.json"), "w") as f:
                json.dump(sample_result, f, indent=2)

            results_summary.append({
                "id": sample_id,
                "original": text,
                "recovered": recovered_text,
                "match": bool(match),
                "similarity": sim_ratio,
                "num_tokens": len(target_token_ids),
                "total_time_s": total_time_s,
                "avg_token_time_s": avg_token_time_s,
                "per_token_times": "_".join(
                    f"{t:.4f}" for t in per_token_times),
                "per_token_steps": "_".join(
                    str(s) for s in per_token_steps),
            })

            processed += 1
            pbar.update(1)
            pbar.set_postfix({
                "match": match,
                "sim": f"{sim_ratio:.2f}",
                "time": f"{total_time_s:.1f}s"
            })

            gc.collect()
            torch.cuda.empty_cache()

        except StopIteration:
            break
        except Exception as e:
            print(f"\nError on sample: {e}")
            import traceback
            traceback.print_exc()
            continue

    pbar.close()

    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(exp_dir, "summary.csv"), index=False)

    if len(df) > 0:
        print(f"\n{'='*60}")
        print(f"Results: {exp_dir}")
        print(f"Samples processed: {len(df)}")
        print(f"Exact matches:     {df['match'].sum()}/{len(df)} "
              f"({100 * df['match'].mean():.1f}%)")
        print(f"Total time (all):  {df['total_time_s'].sum():.1f}s")
        print(f"Avg time/prompt:   {df['total_time_s'].mean():.2f} "
              f"± {df['total_time_s'].std():.2f}s")
        print(f"Avg time/token:    {df['avg_token_time_s'].mean():.4f} "
              f"± {df['avg_token_time_s'].std():.4f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
