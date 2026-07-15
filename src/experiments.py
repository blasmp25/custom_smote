import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from cor_smote import CorSMOTE

cpus_asignadas = int(os.getenv('SLURM_CPUS_PER_TASK', 1))


def tune_sampler_for_dataset(X_train, y_train, smote_type, seed):
    """
    Ajusta hiperparámetros de un sampler (SMOTE, CorSMOTE, ADASYN, Borderline-SMOTE)
    usando RandomForest como modelo base.
    Retorna un dict con 'best_params' y 'best_score'.
    """

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    
    rf = RandomForestClassifier(random_state=seed)

    if smote_type == "corsmote":
        pipe = Pipeline([('sampler', CorSMOTE(random_state=seed)), ('clf', rf)])
        param_grid = {
            'sampler__k_neighbors': [3, 5, 7, 9],
            'sampler__top_k_features': [3, 5, 7, 10]
        }
    elif smote_type == "smote":
        pipe = Pipeline([('sampler', SMOTE(random_state=seed)), ('clf', rf)])
        param_grid = {'sampler__k_neighbors': [3, 5, 7, 9]}
    elif smote_type == "adasyn":
        pipe = Pipeline([('sampler', ADASYN(random_state=seed)), ('clf', rf)])
        param_grid = {'sampler__n_neighbors': [3, 5, 7, 9]}
    elif smote_type == "borderline":
        pipe = Pipeline([('sampler', BorderlineSMOTE(random_state=seed)), ('clf', rf)])
        param_grid = {'sampler__k_neighbors': [3, 5, 7, 9], 'sampler__kind': ['borderline-1', 'borderline-2']}
    else:
        raise ValueError(
            "smote_type debe ser uno de: 'corsmote', 'smote', 'adasyn', "
            "'borderline', 'safe_level_smote', 'smote_ipf', 'mdo'"
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score, average='macro')

    grid = GridSearchCV(pipe, param_grid, scoring=scorer, cv=cv, verbose=0, error_score=np.nan, n_jobs=cpus_asignadas)
    
    try:
        grid.fit(X_train, y_train)

    except Exception as e:
        print(f"{smote_type} failed on this dataset.")
        #print(e)
        return None
    
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }


def optimize_models_parameters(X, y, models, param_grids, seed):
    """
    Optimiza hiperparámetros de varios modelos usando GridSearchCV para un dataset.
    
    :param X: pd.DataFrame o np.array, features de entrenamiento
    :param y: pd.Series o np.array, etiquetas de entrenamiento
    :param models: dict, modelos a entrenar
    :param param_grids: dict, grids de hiperparámetros para cada modelo
    :return: dict, {model_name: {"best_params": ..., "best_score": ...}}
    """
    
    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    for model_name, model in models.items():
        grid_dict = param_grids.get(model_name, {})
        num_combinations = np.prod([len(v) for v in grid_dict.values()]) if grid_dict else 0
        
        
        if num_combinations <=1:
          print(f"    ⏩ Skipping optimization for {model_name} (Fixed parameters)...")
          if grid_dict:
            best_params = {k: v[0] for k, v in grid_dict.items()}
          else:
            best_params = {}
                
            results[model_name] = best_params
          continue # Saltamos directamente al siguiente modelo del bucle
        
        
        
        print(f"   ⚙️ Optimizing {model_name} ...")
        
        # Manejo especial si el modelo requiere array en lugar de DataFrame
        X_fit = X.values if hasattr(X, "values") else X
        y_fit = y.values if hasattr(y, "values") else y
        
        # Ajustes especiales
        if model_name == "SVM" and len(X_fit) > 2000:
            X_fit, _, y_fit, _ = train_test_split(
                X_fit, y_fit, train_size=2000, stratify=y_fit, random_state=seed
            )
        #if model_name == "TabTransformer":
        #    model = TabTransformerClassifier(num_features=X_fit.shape[1])
        
        
        
        
        # GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="f1_macro",
            n_jobs=cpus_asignadas,
            verbose=0,
            error_score=np.nan
        )
        
        X_fit = np.array(X_fit, dtype=np.float32, copy=True)
        y_fit = np.array(y_fit, dtype=np.int64, copy=True)
        
        grid.fit(X_fit, y_fit)
        
        results[model_name] = grid.best_params_
            
        
        
    return results




def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Entrena un modelo con X_train / y_train y evalúa en X_test / y_test.
    Devuelve un diccionario de métricas de test.
    
    :param name: str, nombre del modelo
    :param model: instancia de sklearn con los mejores hiperparámetros
    :param X_train, y_train: datos de entrenamiento
    :param X_test, y_test: datos de test
    :return: dict con métricas de test
    """
    
    # Convertir a arrays numpy por seguridad
    X_tr = np.array(X_train, dtype=np.float32, copy=True)
    y_tr = np.array(y_train, dtype=np.int64, copy=True)
    X_te = np.array(X_test, dtype=np.float32, copy=True)
    y_te = np.array(y_test, dtype=np.int64, copy=True)

    # Entrenamiento final con todo el train
    model.fit(X_tr, y_tr)

    # Predicción en test
    y_pred = model.predict(X_te)
    
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_te, y_pred, average="macro", zero_division=0)
    }


def evaluate_models(X_train, y_train, X_test, y_test, models, best_model_params, seed):
    """
    Evalúa un conjunto de modelos entrenándolos en train y evaluando en test.

    :param X_train, y_train: datos de entrenamiento
    :param X_test, y_test: datos de test
    :param models: dict {nombre: instancia_modelo}
    :param best_model_params: dict {nombre: best_params}
    :return: dict {modelo: métricas de test}
    """
    
    results = {}
    
    for model_name, model_class in models.items():
        if model_name not in best_model_params:
            print(f"⚠️ No hay mejores parámetros para {model_name}, saltando...")
            continue

        # Reconstruir modelo con los mejores parámetros
        params = best_model_params[model_name]
        if "random_state" in model_class.get_params():
            params["random_state"] = seed
        model = model_class.__class__(**params)

        # Evaluar en test usando la función de arriba
        metrics = evaluate_model(
            name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        results[model_name] = metrics
        
    return results