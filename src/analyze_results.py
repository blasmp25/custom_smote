import pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def load_results(filename="results.pkl"):
    print(f"Loading {filename}...")
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def results_to_dataframe(results):
    rows = []

    for dataset, samplers in results.items():

        # ---------- ORIGINAL ----------
        if "Original" in samplers:
            original = samplers["Original"]
            if "models" in original:
                for model_name, model_info in original["models"].items():
                    rows.append({
                        "dataset": dataset,
                        "sampler": "Original",
                        "model": model_name,
                        "ratio": None,
                        "accuracy": model_info["metrics"]["Accuracy"],
                        "precision": model_info["metrics"]["Precision"],
                        "recall": model_info["metrics"]["Recall"],
                        "f1": model_info["metrics"]["F1"]
                    })

        # ---------- RESAMPLED ----------
        for sampler_name, sampler_info in samplers.items():

            if sampler_name == "Original":
                continue

            if "ratios" not in sampler_info:
                continue

            for ratio_name, models in sampler_info["ratios"].items():
                for model_name, metrics in models.items():
                    rows.append({
                        "dataset": dataset,
                        "sampler": sampler_name,
                        "model": model_name,
                        "ratio": ratio_name.replace("ratio_", ""),
                        "accuracy": metrics["Accuracy"],
                        "precision": metrics["Precision"],
                        "recall": metrics["Recall"],
                        "f1": metrics["F1"]
                    })

    return pd.DataFrame(rows)


def compute_imbalance_metrics(y):
    counts = Counter(y)
    n_classes = len(counts)

    n_max = max(counts.values())
    ir_per_class = [n_max / n for n in counts.values()]

    return {
        "n_classes": n_classes,
        "imbalance_rate": np.mean(ir_per_class)
    }


def compute_mean_ranks(df, dataset_info, mode="all"):
    """
    mode: all | binary | multiclass
    """
    ranks = defaultdict(list)

    for dataset in df["dataset"].unique():

        n_classes = dataset_info[dataset]["n_classes"]

        if mode == "binary" and n_classes != 2:
            continue
        if mode == "multiclass" and n_classes == 2:
            continue

        subset = df[df["dataset"] == dataset]

        # Best F1 per sampler
        best_f1 = (
            subset.groupby("sampler")["f1"]
            .max()
            .reset_index()
            .sort_values(by="f1", ascending=False)
        )

        
        best_f1["rank"] = best_f1["f1"].rank(
            method="min", ascending=False
        )

        for _, row in best_f1.iterrows():
            ranks[row["sampler"]].append(row["rank"])

    # Average rank
    mean_ranks = {
        sampler: np.mean(ranks_list)
        for sampler, ranks_list in ranks.items()
        if len(ranks_list) > 0
    }

    return mean_ranks


def plot_mean_ranks(mean_ranks, title, filename):
    
    samplers_sorted = sorted(mean_ranks, key=lambda s: mean_ranks[s])
    values_sorted = [mean_ranks[s] for s in samplers_sorted]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(samplers_sorted, values_sorted)

    plt.ylabel("Mean Rank")
    plt.title(title)
    plt.xticks(rotation=30)

   
    for bar, value in zip(bars, values_sorted):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def analyze_results(filename="results.pkl", output_txt="results_report.txt"):
    results = load_results(filename)
    df = results_to_dataframe(results)

    # Save csv
    df.to_csv("results_summary.csv", index=False)
    print("Saved full results to results_summary.csv\n")

    with open(output_txt, "w", encoding="utf-8") as f:

        def log(msg=""):
            print(msg)
            f.write(msg + "\n")

        dataset_info = {}

        log("RESULTS SUMMARY REPORT")
        log("=" * 90)
        log("")

        # Ranking per dataset
        for dataset in df["dataset"].unique():
            
            y_train = results[dataset]["Original"]["y_train"]
            imbalance_info = compute_imbalance_metrics(y_train)
            dataset_info[dataset] = imbalance_info

            n_classes = imbalance_info["n_classes"]
            imbalance_rate = imbalance_info["imbalance_rate"]

            log(
                f"\n================ Dataset: {dataset} | "
                f"Classes: {n_classes} | Imbalance rate: {imbalance_rate:.2f} "
                f"================\n"
            )

            subset = df[df["dataset"] == dataset]

            # Get the row with the highest F1-score
            best_per_sampler = subset.loc[
                subset.groupby("sampler")["f1"].idxmax()
            ]

            # Sort by F1-score
            best_per_sampler = best_per_sampler.sort_values(by="f1", ascending=False)

            log(
                best_per_sampler[
                    ["sampler", "model", "ratio", "f1", "accuracy", "precision", "recall"]
                ].to_string(index=False)
            )

        # Plots
        mean_all = compute_mean_ranks(df, dataset_info, mode="all")
        mean_bin = compute_mean_ranks(df, dataset_info, mode="binary")
        mean_multi = compute_mean_ranks(df, dataset_info, mode="multiclass")

        plot_mean_ranks(
            mean_all,
            "Mean Rank per Sampler (All Datasets)",
            "mean_rank_all.png"
        )

        plot_mean_ranks(
            mean_bin,
            "Mean Rank per Sampler (Binary Datasets)",
            "mean_rank_binary.png"
        )

        plot_mean_ranks(
            mean_multi,
            "Mean Rank per Sampler (Multiclass Datasets)",
            "mean_rank_multiclass.png"
        )

        log("\nSaved ranking plots:")
        log("- mean_rank_all.png")
        log("- mean_rank_binary.png")
        log("- mean_rank_multiclass.png")
        log("\nEnd of report.")



if __name__ == "__main__":
    analyze_results()
