import pickle
import pandas as pd

def load_results(filename="results.pkl"):
    print(f"Loading {filename}...")
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def results_to_dataframe(results):
    print("Converting to DataFrame...")
    rows = []
    for dataset, samplers in results.items():
        for sampler_name, sampler_info in samplers.items():
            # Solo procesamos los samplers que tienen modelos
            if "models" in sampler_info:
                for model_name, model_info in sampler_info["models"].items():
                    rows.append({
                        "dataset": dataset,
                        "sampler": sampler_name,
                        "model": model_name,
                        "accuracy": model_info["metrics"]["Accuracy"],
                        "f1": model_info["metrics"]["F1"],
                        "precision": model_info["metrics"]["Precision"],
                        "recall": model_info["metrics"]["Recall"]
                    })
    df = pd.DataFrame(rows)
    return df

def analyze_results(filename="results.pkl"):
    results = load_results(filename)
    df = results_to_dataframe(results)

    # Guardar todos los resultados en CSV completo
    df.to_csv("results_summary.csv", index=False)
    print("Saving DataFrame to results_summary.csv...\n")

    # Mostrar solo el mejor modelo por sampler para cada dataset
    print("Best model per sampler for each dataset (by F1):\n")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        # Para cada sampler, coger el modelo con mayor F1
        best_per_sampler = subset.loc[subset.groupby('sampler')['f1'].idxmax()]
        # Ordenar de mayor a menor por F1
        best_subset = best_per_sampler.sort_values(by='f1', ascending=False)
        print(f"Dataset: {dataset}")
        print(best_subset[['sampler', 'model', 'accuracy', 'f1', 'precision', 'recall']].to_string(index=False))
        print()

if __name__ == "__main__":
    analyze_results()
