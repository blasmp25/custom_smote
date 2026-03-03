import os
import pickle
import argparse
from experiments import optimize_models_parameters, evaluate_models, tune_sampler_for_dataset
from data_setup import prepare_datasets
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from cor_smote import CorSMOTE
import warnings
from sklearn.exceptions import DataConversionWarning
from tqdm import tqdm
import pandas as pd 
import numpy as np
# Ignorar solo los warnings de DataConversionWarning (nombre de columnas)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")


MODELS_MAP = {
    'rf' : 'Random Forest',
    'lightgbm' : 'LightGBM',
    'xgboost' : 'XGBoost'
}

# Parámetros a optimizar para cada modelo
param_grids = {
    
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"]
    },
    
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    
    "LightGBM": {
        "n_estimators": [200, 500],
        "num_leaves": [31, 64],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20]
    }
}

SAMPLING_RATIOS = [0.6, 0.8, 1.0]


def run_training_pipeline(
    prepared_datasets: dict,
    models,
    param_grids: dict,
    samplers,
    seed,
    table, 
    results_file: str = "results.pkl",
    overwrite: bool = False
) -> dict:
    """
    Pipeline for TFM: trains models on original and resampled datasets,
    optimizes sampler and model parameters, and saves results incrementally.

    :param prepared_datasets: dict returned by prepare_datasets()
    :param models: dict of models to train
    :param param_grids: dict of hyperparameter grids for models
    :param results_file: Pickle file to save accumulated results
    :param overwrite: if True, recalculates datasets even if already in results
    :return: dict with all results
    """
    
    # models = {k: v for k, v in MODELS.items() if k in models}
    
    # Load previous results if they exist
    if os.path.exists(results_file):
        # print("[INFO] Found results file.")
        with open(results_file, "rb") as f:
            results = pickle.load(f)
            
    else:
        results = {}

    
    # Iterate over datasets
    for ds_name, data in prepared_datasets.items():
        
        np.random.seed(seed)

        current_models = {
            'Random Forest': RandomForestClassifier(random_state=seed, n_jobs=-1),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=-1),
            'LightGBM': LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=-1)
        }
        # Filter to only the models requested via CLI
        active_models = {k: v for k, v in current_models.items() if k in models}

        dataset = ds_name.split(".")[0]
        results[ds_name] = {}
        
        # print(f"\n=== Training dataset: {ds_name} ===")
        
        # ---------- Original dataset ----------
        # print("Training on original dataset...")
        
        results[ds_name]["Original"] = {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"]
        }
        
        X_train_orig = data["X_train"].copy()
        y_train_orig = data["y_train"].copy()
        X_test_orig = data["X_test"].copy()
        y_test_orig = data["y_test"].copy()
        values, n_classes = np.unique(y_train_orig, return_counts=True)
        # print("Optimizing models parameters...")
        best_model_params = optimize_models_parameters(X_train_orig, y_train_orig, active_models, param_grids, seed=seed)
        
        # print("Evaluating models...")
        metrics = evaluate_models(X_train_orig, y_train_orig, X_test_orig, y_test_orig, active_models, best_model_params, seed=seed)
        
        results[ds_name]["Original"]["models"] = {
            m: {"best_params": best_model_params[m], "metrics": metrics[m]}
            for m in best_model_params
        }

        for model in metrics.keys():
            metrics[model]["dataset"] = dataset
            metrics[model]["ratio"] = ""
            metrics[model]["sampler"] = "Original"
            metrics[model]["seed"] = seed
            metrics[model]["n_classes"] = len(n_classes)
            metrics[model]["nrows"] = X_train_orig.shape[0] + X_test_orig.shape[0]
            metrics[model]["nfeat"] = X_train_orig.shape[1]
            row = pd.Series(metrics[model])


            table = pd.concat([table, row.to_frame().T], ignore_index=True)
            table.to_csv("TablaResultados.csv")
        
        # ------------ Resampled datasets ----------
        for sampler_type in samplers:
            # print(f"Applying {sampler_type} ...")
            
            # Tune sampler parameters
            # print("Optimizing sampler parameters...")
            tune_type = sampler_type.lower()
            sampler_params = tune_sampler_for_dataset(
                X_train_orig.to_numpy().copy(),
                y_train_orig.to_numpy().copy(),
                smote_type=tune_type,
                seed=seed
            )
            
            if sampler_params is None:
                continue
            
            if sampler_type == "smote":
                sampler = SMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    random_state=seed
                )
            elif sampler_type == "corsmote":
                sampler = CorSMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    top_k_features=sampler_params['best_params']['sampler__top_k_features'],
                    random_state=seed
                )
            elif sampler_type == "adasyn":
                sampler = ADASYN(
                    n_neighbors=sampler_params['best_params']['sampler__n_neighbors'],
                    random_state=seed
                )
            elif sampler_type == "borderline":
                sampler = BorderlineSMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    kind=sampler_params['best_params']['sampler__kind'],
                    random_state=seed
                )
            else:
                raise ValueError(f"Unknown sampler type: {sampler_type}")
            
            X_train_orig = X_train_orig.astype(float).copy()
            y_train_orig = y_train_orig.astype(int).copy()
            X_res, y_res = sampler.fit_resample(X_train_orig, y_train_orig)
            
            results[ds_name][sampler_type] = {
                "best_sampler_params" : sampler_params['best_params'],
                "X_resampled" : X_res,
                "y_resampled" : y_res
            }
                
            # Train models on resampled dataset
            # print("Optimizing models parameters...")
            best_model_params = optimize_models_parameters(X_res, y_res, active_models, param_grids, seed=seed)
            
            results[ds_name][sampler_type]["best_model_params"] = best_model_params
            
            # print("Evaluating models...")
            
            results[ds_name][sampler_type]["ratios"] = {}
            for ratio in SAMPLING_RATIOS:
            
                # Determinar sampling_strategy según si es binario o multiclase
                class_counts = y_train_orig.value_counts()
                majority_class = class_counts.idxmax()
                minority_classes = [c for c in class_counts.index if c != majority_class]

                
                sampling_strategy = {
                    c: max(class_counts[c], int(class_counts[majority_class] * ratio))
                    for c in minority_classes
                    }


                
                
                if sampler_type == "smote":
                    sampler_ratio = SMOTE(
                        k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                        random_state=seed,
                        sampling_strategy=sampling_strategy,
                        
                    )
                elif sampler_type == "corsmote":
                    sampler_ratio = CorSMOTE(
                        k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                        top_k_features=sampler_params['best_params']['sampler__top_k_features'],
                        sampling_strategy=sampling_strategy,
                        random_state=seed
                    )
                elif sampler_type == "adasyn":
                    sampler_ratio = ADASYN(
                        n_neighbors=sampler_params['best_params']['sampler__n_neighbors'],
                        sampling_strategy=sampling_strategy,
                        random_state=seed
                    )
                elif sampler_type == "borderline":
                    sampler_ratio = BorderlineSMOTE(
                        k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                        kind=sampler_params['best_params']['sampler__kind'],
                        sampling_strategy=sampling_strategy,
                        random_state=seed
                    )
                    
                try:
                    X_res_r, y_res_r = sampler_ratio.fit_resample(X_train_orig, y_train_orig)
                except ValueError as e:
                    print(f"[WARN] {sampler_type} with ratio {ratio} failed: {e}")
                    continue  # pasa al siguiente ratio si falla
                
                metrics = evaluate_models(X_res_r, y_res_r, X_test_orig, y_test_orig, active_models, best_model_params, seed)
                
                results[ds_name][sampler_type]["ratios"][f"ratio_{ratio}"] = metrics
                
                for model in metrics.keys():
                    metrics[model]["dataset"] = dataset
                    metrics[model]["ratio"] = ratio
                    metrics[model]["sampler"] = sampler_type
                    metrics[model]["seed"] = seed
                    metrics[model]["n_classes"] = len(n_classes)
                    metrics[model]["nrows"] = X_train_orig.shape[0] + X_test_orig.shape[0]
                    metrics[model]["nfeat"] = X_train_orig.shape[1]
                    row = pd.Series(metrics[model])
                    table = pd.concat([table, row.to_frame().T], ignore_index=True)
                    table.to_csv("TablaResultados.csv")
                
    # Save results
    with open(results_file, "wb") as f:
        pickle.dump(results,f)
        
    print(f"\nPipeline completed. Results saved to {results_file}")
    
    return results, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    
    parser.add_argument(
        "--datasets",   
        nargs="+",
        default=None,
        help= "List of dataset filenames to process. Default: all CSVs in the datasets/ directory."
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing results for datasets"
    )
    
    
    parser.add_argument(
        "--results_file",
        type=str,
        default="results.pkl",
        help="Pickle file to save/load results."
    )
    
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=None,
        choices=["none", "smote", "borderline", "adasyn", "corsmote"],
        help=(
            "Samplers to use. Default: all_samplers"
            "Options: none, smote, corsmote, adasyn, borderline"
        )
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["rf", "lightgbm", "xgboost"],
        help=(
            "Models to use. Default: all_models"
            "Options: rf, lightgbm, xgboost"
        )
    )
    
    args = parser.parse_args()
    
    if args.samplers is None:
        selected_samplers = ["smote", "corsmote", "adasyn", "borderline"]
    
    elif "none" in args.samplers:
        selected_samplers = []
        
    else:
        selected_samplers = args.samplers
        
    
    if args.models is None:
        selected_models = ["rf", "xgboost", "lightgbm"]
    
    else:
        selected_models = args.models
        
    selected_models = [MODELS_MAP[m] for m in selected_models]

    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'dataset', 'ratio', 'sampler', 'seed', 'n_classes', "nrows", "nfeat"])  
    for iter in tqdm(range(30)):
        
        prepared_datasets = prepare_datasets(dataset_names=args.datasets, seed=iter)
        sorted_names = sorted(prepared_datasets.keys())
    
        for ds_name in sorted_names:
            # Create a dictionary with just one dataset to process
            single_ds = {ds_name: prepared_datasets[ds_name]}
            
            # Run the pipeline for just this one dataset
            res, results_df = run_training_pipeline(
                prepared_datasets=single_ds,
                models=selected_models,
                param_grids=param_grids,
                results_file=args.results_file,
                overwrite=args.overwrite,
                samplers=selected_samplers,
                seed=iter,
                table = results_df 
            )
    print(res)