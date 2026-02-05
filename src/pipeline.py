import os
import pickle
import argparse
from .experiments import optimize_models_parameters, evaluate_models, tune_sampler_for_dataset
from .data_setup import prepare_datasets
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from .custom_smote import CustomSMOTE
import warnings
from sklearn.exceptions import DataConversionWarning

# Ignorar solo los warnings de DataConversionWarning (nombre de columnas)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")

MODELS = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbosity=-1)
}

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
    
    models = {k: v for k, v in MODELS.items() if k in models}
    
    # Load previous results if they exist
    if os.path.exists(results_file):
        print("[INFO] Found results file.")
        with open(results_file, "rb") as f:
            results = pickle.load(f)
            
    else:
        results = {}
        
    # Iterate over datasets
    for ds_name, data in prepared_datasets.items():
        
        results[ds_name] = {}
        
        print(f"\n=== Training dataset: {ds_name} ===")
        
        # ---------- Original dataset ----------
        print("Training on original dataset...")
        
        results[ds_name]["Original"] = {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"]
        }
        
        X_train_orig = data["X_train"]
        y_train_orig = data["y_train"]
        X_test_orig = data["X_test"]
        y_test_orig = data["y_test"]
        
        print("Optimizing models parameters...")
        best_model_params = optimize_models_parameters(X_train_orig, y_train_orig, models, param_grids)
        
        print("Evaluating models...")
        metrics = evaluate_models(X_train_orig, y_train_orig, X_test_orig, y_test_orig, models, best_model_params)
        
        results[ds_name]["Original"]["models"] = {
            m: {"best_params": best_model_params[m], "metrics": metrics[m]}
            for m in best_model_params
        }
        
        # ------------ Resampled datasets ----------
        for sampler_type in samplers:
            print(f"Applying {sampler_type} ...")
            
            # Tune sampler parameters
            print("Optimizing sampler parameters...")
            tune_type = sampler_type.lower()
            sampler_params = tune_sampler_for_dataset(
                X_train_orig.to_numpy().copy(),
                y_train_orig.to_numpy().copy(),
                smote_type=tune_type
            )
            
            if sampler_params is None:
                continue
            
            if sampler_type == "smote":
                sampler = SMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    random_state=42
                )
            elif sampler_type == "customsmote":
                sampler = CustomSMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    top_k_features=sampler_params['best_params']['sampler__top_k_features']
                )
            elif sampler_type == "adasyn":
                sampler = ADASYN(
                    n_neighbors=sampler_params['best_params']['sampler__n_neighbors']
                )
            elif sampler_type == "borderline":
                sampler = BorderlineSMOTE(
                    k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                    kind=sampler_params['best_params']['sampler__kind']
                )
            else:
                raise ValueError(f"Unknown sampler type: {sampler_type}")
            
            X_train_orig = X_train_orig.astype(float)
            y_train_orig = y_train_orig.astype(int)
            X_res, y_res = sampler.fit_resample(X_train_orig, y_train_orig)
            
            results[ds_name][sampler_type] = {
                "best_sampler_params" : sampler_params['best_params'],
                "X_resampled" : X_res,
                "y_resampled" : y_res
            }
                
            # Train models on resampled dataset
            print("Optimizing models parameters...")
            best_model_params = optimize_models_parameters(X_res, y_res, models, param_grids)
            
            results[ds_name][sampler_type]["best_model_params"] = best_model_params
            
            print("Evaluating models...")
            
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
                        random_state=42,
                        sampling_strategy=sampling_strategy
                    )
                elif sampler_type == "customsmote":
                    sampler_ratio = CustomSMOTE(
                        k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                        top_k_features=sampler_params['best_params']['sampler__top_k_features'],
                        sampling_strategy=sampling_strategy
                    )
                elif sampler_type == "adasyn":
                    sampler_ratio = ADASYN(
                        n_neighbors=sampler_params['best_params']['sampler__n_neighbors'],
                        sampling_strategy=sampling_strategy
                    )
                elif sampler_type == "borderline":
                    sampler_ratio = BorderlineSMOTE(
                        k_neighbors=sampler_params['best_params']['sampler__k_neighbors'],
                        kind=sampler_params['best_params']['sampler__kind'],
                        sampling_strategy=sampling_strategy
                    )
                    
                try:
                    X_res_r, y_res_r = sampler_ratio.fit_resample(X_train_orig, y_train_orig)
                except ValueError as e:
                    print(f"[WARN] {sampler_type} with ratio {ratio} failed: {e}")
                    continue  # pasa al siguiente ratio si falla
                
                metrics = evaluate_models(X_res_r, y_res_r, X_test_orig, y_test_orig, models, best_model_params)
                
                results[ds_name][sampler_type]["ratios"][f"ratio_{ratio}"] = metrics
            
    # Save results
    with open(results_file, "wb") as f:
        pickle.dump(results,f)
        
    print(f"\nPipeline completed. Results saved to {results_file}")
    
    return results


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
        choices=["none", "smote", "borderline", "adasyn", "customsmote"],
        help=(
            "Samplers to use. Default: all_samplers"
            "Options: none, smote, customsmote, adasyn, borderline"
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
        selected_samplers = ["smote", "customsmote", "adasyn", "borderline"]
    
    elif "none" in args.samplers:
        selected_samplers = []
        
    else:
        selected_samplers = args.samplers
        
    
    if args.models is None:
        selected_models = ["rf", "lightgbm", "xgboost"]
    
    else:
        selected_models = args.models
        
    selected_models = [MODELS_MAP[m] for m in selected_models]

    
    prepared_datasets = prepare_datasets(dataset_names=args.datasets)
    res = run_training_pipeline(prepared_datasets=prepared_datasets,
                          models=selected_models,
                          param_grids=param_grids,
                          results_file=args.results_file,
                          overwrite=args.overwrite,
                          samplers=selected_samplers)
    
    print(res)