import os
import json
import glob
import argparse
import matplotlib.pyplot as plt

def generate_first_error_per_token_histograms(results_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"Nie znaleziono plików .json w folderze: {results_dir}")
        return

    first_errors_per_position = {i: [] for i in range(14)}
    perfect_reconstructions = 0
    total_prompts = len(json_files)

    print(f"Analiza {total_prompts} promptów...")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "trajectory" not in data:
            continue

        error_found_in_prompt = False

        for i, step_data in enumerate(data["trajectory"]):
            if step_data["token_str"] != step_data["true_token_str"]:
                if i < 14:
                    if "rank_evolution_true" in step_data and step_data["rank_evolution_true"]:
                        final_true_rank = step_data["rank_evolution_true"][-1]
                        first_errors_per_position[i].append(final_true_rank)
                
                error_found_in_prompt = True
                break

        if not error_found_in_prompt:
            perfect_reconstructions += 1

    print(f"Idealnych rekonstrukcji: {perfect_reconstructions}/{total_prompts}")

    for i in range(14):
        ranks = first_errors_per_position[i]
        token_number = i + 1
        
        if not ranks:
            print(f"Token {token_number}: Brak pierwszych błędów na tej pozycji, pomijam wykres.")
            continue

        plt.figure(figsize=(12, 7))

        plt.hist(ranks, bins=40, color="#d62728", edgecolor="black", alpha=0.8)

        plt.title(f"Rank of the correct token at the first error (Error at token {token_number})", fontsize=14, pad=15)
        plt.xlabel("Rank in the vocabulary", fontsize=12)
        plt.ylabel("Number of occurrences (number of prompts)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        
        output_image = os.path.join(output_dir, f"histogram_pierwszy_blad_token_{token_number:02d}.png")
        plt.savefig(output_image, dpi=300)
        plt.close()

        print(f"Token {token_number}: Zapisano wykres ({len(ranks)} promptów, gdzie błąd zaczął się tutaj) -> {output_image}")

def main():

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Folder containing JSON result files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output folder for the generated images"
    )

    args = parser.parse_args()

    generate_first_error_per_token_histograms(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()