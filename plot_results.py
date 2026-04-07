import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.legend import Legend

def parse_args():
    parser = argparse.ArgumentParser(description="Plot SIPIT Results")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing JSON results")
    return parser.parse_args()

def generate_plot(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    trajectory = data['trajectory']
    original_text = data.get('original_text', '')
    recovered_text = data.get('recovered_text', '')
    sim = data.get('similarity', 0.0)
    
    steps_data = [t['loss_history'] for t in trajectory]
    tokens = [t['token_str'] for t in trajectory]
    true_tokens = [t.get('true_token_str', '?') for t in trajectory]
    
    rank_evo_winner = [t['rank_evolution_winner'] for t in trajectory]
    rank_evo_true = [t['rank_evolution_true'] for t in trajectory]
    rank_steps = [t['rank_evolution_steps'] for t in trajectory]
    banned_counts = [t.get('banned_tokens_count', 0) for t in trajectory]
    
    fig, (ax1, ax3, ax_text) = plt.subplots(
    3, 1,
    figsize=(14, 14),
    gridspec_kw={'height_ratios': [1, 1.5, 0.6]}
)

    colors = plt.cm.jet(np.linspace(0, 1, len(tokens)))

    # ---------------- LOSS ----------------
    for i, losses in enumerate(steps_data):
        ax1.plot(losses, color=colors[i], alpha=0.5, linewidth=1)

    ax1.set_yscale('log')
    ax1.set_title(f"Loss | Similarity: {sim:.2f}", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---------------- RANK ----------------
    solid_handles = []
    solid_labels = []

    dashed_handles = []
    dashed_labels = []

    for i, (r_steps, r_win, r_true) in enumerate(
        zip(rank_steps, rank_evo_winner, rank_evo_true)
    ):
        lbl = tokens[i].replace('\n', '\\n')
        true_lbl = true_tokens[i].replace('\n', '\\n')

        line_solid, = ax3.plot(
            r_steps,
            r_win,
            color=colors[i],
            linewidth=2,
            alpha=0.9
        )

        solid_handles.append(line_solid)
        solid_labels.append(f"T{i}: {lbl} | banned={banned_counts[i]}")

        line_dashed, = ax3.plot(
            r_steps,
            r_true,
            color=colors[i],
            linestyle='--',
            linewidth=1.8,
            alpha=0.8
        )

        dashed_handles.append(line_dashed)
        dashed_labels.append(f"T{i}: {true_lbl}")

    ax3.set_yscale('log')
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Rank (Log Scale)")
    ax3.set_title("Rank Convergence\nSolid: Reconstructed | Dashed: Ground Truth", fontsize=11)
    ax3.grid(True, which="both", alpha=0.2)
    ax3.set_ylim(0.5, 60000)

    if len(tokens) < 40:

        all_handles = solid_handles + dashed_handles
        all_labels = solid_labels + dashed_labels

        ax3.legend(
            all_handles,
            all_labels,
            ncol=2,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize='small',
            columnspacing=1.5,
            handlelength=2.5
        )

    ax_text.axis("off")

    wrapped_gt = original_text
    wrapped_rec = recovered_text

    ax_text.text(
        0.01, 0.95,
        "Ground Truth:",
        fontsize=11,
        fontweight='bold',
        verticalalignment='top'
    )

    ax_text.text(
        0.01, 0.80,
        wrapped_gt,
        fontsize=9,
        verticalalignment='top',
        wrap=True
    )

    ax_text.text(
        0.01, 0.45,
        "Reconstructed:",
        fontsize=11,
        fontweight='bold',
        verticalalignment='top'
    )

    ax_text.text(
        0.01, 0.30,
        wrapped_rec,
        fontsize=9,
        verticalalignment='top',
        wrap=True
    )

    plt.tight_layout()
    save_path = json_path.replace('.json', '.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    if not os.path.exists(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    json_files = glob.glob(os.path.join(args.dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {args.dir}")
    
    for json_file in json_files:
        try:
            generate_plot(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    main()