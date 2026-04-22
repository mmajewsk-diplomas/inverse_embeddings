import os
import json
import glob
import argparse
import matplotlib.pyplot as plt


def generate_first_error_histogram(results_dir: str, output_image: str):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"Nie znaleziono plików .json w folderze: {results_dir}")
        return

    first_error_ranks = []
    perfect_reconstructions = 0
    total_prompts = len(json_files)

    print(f"Analiza {total_prompts} promptów...")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "trajectory" not in data:
            continue

        error_found_in_prompt = False

        for step_data in data["trajectory"]:
            if step_data["token_str"] != step_data["true_token_str"]:
                if "rank_evolution_true" in step_data and step_data["rank_evolution_true"]:
                    final_true_rank = step_data["rank_evolution_true"][-1]
                    first_error_ranks.append(final_true_rank)

                error_found_in_prompt = True
                break

        if not error_found_in_prompt:
            perfect_reconstructions += 1

    plt.figure(figsize=(12, 7))

    plt.hist(first_error_ranks, bins=40, color="#d62728", edgecolor="black", alpha=0.8)

    plt.title("Rank of the correct token at the first error in the sequence", fontsize=14, pad=15)
    plt.xlabel("Rank in the vocabulary", fontsize=12)
    plt.ylabel("Number of occurrences (number of prompts)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()

    print(f"Wykres zapisano jako: {output_image}")
    print(f"Perfect reconstructions: {perfect_reconstructions}/{total_prompts}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Folder containing JSON result files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="histogram_pierwszego_bledu.png",
        help="Output image file"
    )

    args = parser.parse_args()

    print("results_dir:", args.results_dir)
    print("output:", args.output)

    generate_first_error_histogram(args.results_dir, args.output)


if __name__ == "__main__":
    main()