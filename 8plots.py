import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sipit_w_rank import sipit
from tqdm import tqdm


PROMPTS = [
    "The weather in London is usually rainy.",
    "Artificial intelligence is changing the world.",
    "To be or not to be, that is the question.",
    "Machine learning models require data.",

    "Xylophone quartets rarely occur in vacuum fluctuations.",
    "def __init__(self, *args, **kwargs): pass",
    "Gemma-2b parameter initialization #@! error.",
    "Quantum entanglement violates local realism 404."
]

MODEL_NAME = "gpt2"
STEPS = 500
OUTPUT_FILE = "cumulative_loss_analysis.png"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device).eval()

    fig, axes = plt.subplots(8, 1, figsize=(12, 24))
    plt.subplots_adjust(hspace=0.8)

    print(f"Processing {len(PROMPTS)} prompts...")

    for i, text in enumerate(tqdm(PROMPTS)):
        ax = axes[i]
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids[0]
        
        with torch.no_grad():
            out = model(**inputs)
            target_hidden = out.hidden_states[-1][0]

        result = sipit(
            model=model,
            target_hidden_states=target_hidden,
            target_ids=input_ids,
            tokenizer=tokenizer,
            steps=STEPS, 
            verbose=False
        )

        # trajectory = result['trajectory']
        # recovered_text = "".join([t['token_str'] for t in trajectory])
        
        # final_losses = [t['loss_history'][-1] for t in trajectory]
        # tokens_str = [t['token_str'] for t in trajectory]
        
        trajectory = result['trajectory']
        recovered_text = "".join([t['token_str'] for t in trajectory])
        
        final_losses = [t['discrete_loss'] for t in trajectory] 
        tokens_str = [t['token_str'] for t in trajectory]
        
        cumulative_losses = np.cumsum(final_losses)
        
        diffs = np.concatenate(([cumulative_losses[0]], np.diff(cumulative_losses)))
        colors = plt.cm.RdYlGn_r(diffs / (np.max(diffs) + 1e-9)) 

        bars = ax.bar(range(len(cumulative_losses)), cumulative_losses, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.plot(range(len(cumulative_losses)), cumulative_losses, color='darkblue', linestyle='--', alpha=0.5)

        ax.set_xticks(range(len(tokens_str)))
        ax.set_xticklabels(tokens_str, rotation=45, ha='right', fontsize=9, fontfamily='monospace')
        ax.set_ylabel("Cumulative Loss")
        
        is_perfect = (text == recovered_text)
        title_bg = "#e6fffa" if is_perfect else "#fff5f5" 
        
        info_text = (
            f"Prompt {i+1} | Length: {len(tokens_str)} tokens\n"
            f"GT:  {text}\n"
            f"REC: {recovered_text}"
        )
        
        ax.set_title(info_text, loc='left', fontsize=10, family='monospace', 
                     bbox=dict(facecolor=title_bg, alpha=1.0, pad=5, edgecolor='gray'))
        
        if len(tokens_str) < 15:
            for idx, rect in enumerate(bars):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        f'{final_losses[idx]:.1e}',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)

if __name__ == "__main__":
    main()