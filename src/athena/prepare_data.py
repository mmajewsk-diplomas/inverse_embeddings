# prepare_data.py
import json
from datasets import load_dataset
import os

print("Downloading data...")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
iterator = iter(ds)

samples = []
for _ in range(50):
    try:
        row = next(iterator)
        samples.append(row['text'])
    except StopIteration:
        break

output_path = os.path.expandvars("$SCRATCH/inverse_embeddings/wiki_local.json")
with open(output_path, "w") as f:
    json.dump(samples, f)

print(f"Saved {len(samples)} samples to: {output_path}")