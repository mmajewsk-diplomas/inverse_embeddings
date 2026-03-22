import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest

def compute_mean_curve(curves):
    if not curves:
        return []
    # zip_longest grupuje wartości dla każdego tokena; krótsze krzywe dostają 'None'
    # Następnie wyliczamy średnią z wartości, które nie są 'None'
    return [np.mean([v for v in token_vals if v is not None]) for token_vals in zip_longest(*curves)]

def generate_cumulative_loss_plot(results_dir, output_file="cumulative_loss_plot.png"):
    search_pattern = os.path.join(results_dir, "sample_*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"Error: No JSON files found in directory '{results_dir}'.")
        return

    plt.figure(figsize=(9, 6), dpi=200)

    added_correct_label = False
    added_incorrect_label = False

    correct_curves = []
    incorrect_curves = []

    print(f"Found {len(json_files)} files in '{results_dir}'. Processing data...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trajectory = data.get('trajectory', [])
        if not trajectory:
            continue

        original_text = data.get('original_text', "")
        recovered_text = data.get('recovered_text', "")

        losses = [t.get('discrete_loss', 0.0) for t in trajectory]
        cumulative_losses = np.cumsum(losses)
        x_values = np.arange(1, len(cumulative_losses) + 1)

        is_perfect = (original_text == recovered_text)

        if is_perfect:
            correct_curves.append(cumulative_losses)
            color = 'blue'
            alpha = 0.15 
            label = "Correct reconstruction" if not added_correct_label else None
            added_correct_label = True
        else:
            incorrect_curves.append(cumulative_losses)
            color = 'red'
            alpha = 0.15
            label = "Incorrect reconstruction" if not added_incorrect_label else None
            added_incorrect_label = True

        plt.plot(x_values, cumulative_losses, color=color, alpha=alpha, linewidth=1.0, label=label)

    if correct_curves:
        mean_correct = compute_mean_curve(correct_curves)
        x_mean_correct = np.arange(1, len(mean_correct) + 1)
        plt.plot(x_mean_correct, mean_correct, color='darkgreen', linewidth=2.5, 
                 label="Mean (correct)")

    if incorrect_curves:
        mean_incorrect = compute_mean_curve(incorrect_curves)
        x_mean_incorrect = np.arange(1, len(mean_incorrect) + 1)
        plt.plot(x_mean_incorrect, mean_incorrect, color='limegreen', linewidth=2.5, linestyle='--', 
                 label="Mean (incorrect)")

    plt.xlabel("Token number", fontsize=12)
    plt.ylabel("Cumulative Loss (Log Scale)", fontsize=12)
    plt.title("Cumulative reconstruction loss per token with means", fontsize=14)

    plt.yscale('log')

    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        for handle in handles:
            if handle.get_alpha() is not None and handle.get_alpha() < 1.0:
                handle.set_alpha(1.0)
                handle.set_linewidth(2.0)
        # Przesunięcie legendy poza wykres (na prawo)
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Dopasowanie układu, aby legenda zmieściła się na obszarze roboczym
    plt.tight_layout()
    # Zapis z bbox_inches='tight' gwarantuje, że wystająca legenda nie zostanie ucięta
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Success! Plot saved as '{output_file}'.")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate cumulative loss plot from JSON files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the folder with JSON results (e.g., results_batch/gpt2...)")
    parser.add_argument("--out_file", type=str, default="cumulative_loss_plot.png", help="Output file name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_cumulative_loss_plot(args.data_dir, args.out_file)