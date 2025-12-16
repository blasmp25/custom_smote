import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#from IPython.display import display

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import FunctionTransformer

from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

from .custom_smote import CustomSMOTE
#from .utils import TabTransformerClassifier

# Función para optimizar parámetros de una técnica de sobremuestreo
def tune_sampler_for_dataset(X_train, y_train, dataset_name, smote_type, results_csv):
    """
    Ajusta los hiperparámetros de un sampler (SMOTE, CustomSMOTE, ADASYN, Borderline-SMOTE).
    Si los resultados ya están en el CSV, los recupera en lugar de recalcularlos.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Conjunto de características de entrenamiento
    y_train : pd.Series
        Etiquetas correspondientes
    dataset_name : str
        Nombre del dataset (clave para guardar/buscar en CSV)
    smote_type : str
        Tipo de sampler: "smote", "custom", "adasyn" o "borderline"
    results_csv : str
        Ruta al CSV donde se guardan los resultados
    
    Retorna:
    --------
    dict con 'dataset', 'best_params', 'best_score'
    """

    results_path = os.path.join("Resultados", results_csv)
    
    # Si existe el CSV, cargarlo
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=["dataset", "smote_type", "best_params", "best_score"])
    
    # ¿Ya tenemos resultados para este dataset?
    if (results_df["dataset"] == dataset_name).any():
        print(f"⚡ Resultados ya calculados para {dataset_name} con {smote_type}, recuperando del CSV...")
        row = results_df[(results_df["dataset"] == dataset_name)].iloc[0]
        return {
            "dataset": dataset_name,
            "best_params": eval(row["best_params"]),  # guardado como string
            "best_score": row["best_score"]
        }
    
    # Modelo base
    rf = RandomForestClassifier(random_state=42)

    # Definir pipeline y grids de parámetros según el tipo
    if smote_type == "custom":
        pipe = Pipeline([
            ('sampler', CustomSMOTE(random_state=42)),
            ('clf', rf)
        ])
        param_grid = {
            'sampler__k_neighbors': [3, 5, 7, 9],
            'sampler__top_k_features': [3, 5, 7, 10]
        }

    elif smote_type == "smote":
        pipe = Pipeline([
            ('sampler', SMOTE(random_state=42)),
            ('clf', rf)
        ])
        param_grid = {
            'sampler__k_neighbors': [3, 5, 7, 9]
        }

    elif smote_type == "adasyn":
        pipe = Pipeline([
            ('sampler', ADASYN(random_state=42)),
            ('clf', rf)
        ])
        param_grid = {
            'sampler__n_neighbors': [3, 5, 7, 9]
            
        }

    elif smote_type == "borderline":
        pipe = Pipeline([
            ('sampler', BorderlineSMOTE(random_state=42)),
            ('clf', rf)
        ])
        param_grid = {
            'sampler__k_neighbors': [3, 5, 7, 9],
            'sampler__kind': ['borderline-1', 'borderline-2']
        }

    else:
        raise ValueError("smote_type debe ser uno de: 'custom', 'smote', 'adasyn', 'borderline'")

    # CV y scorer
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
    scorer = make_scorer(f1_score, average='weighted')

    # GridSearch
    print(f"🔍 Buscando mejores hiperparámetros para {dataset_name} con {smote_type}...")
    grid = GridSearchCV(pipe, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=0, error_score=np.nan)
    grid.fit(X_train, y_train)

    # Guardar resultados
    new_row = {
        "dataset": dataset_name,
        "best_params": str(grid.best_params_),  # guardamos como string
        "best_score": grid.best_score_
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    results_df.to_csv(results_path, index=False)

    print(f"✅ Guardados resultados para {dataset_name} con {smote_type}")
    return {
        "dataset": dataset_name,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_
    }


def tune_sampler_for_dataset(X_train, y_train, smote_type):
    """
    Ajusta hiperparámetros de un sampler (SMOTE, CustomSMOTE, ADASYN, Borderline-SMOTE)
    usando RandomForest como modelo base.
    Retorna un dict con 'best_params' y 'best_score'.
    """

    X_train = X_train.astype(float)
    y_train = y_train.astype(int)
    
    rf = RandomForestClassifier(random_state=42)

    if smote_type == "customsmote":
        pipe = Pipeline([('sampler', CustomSMOTE(random_state=42)), ('clf', rf)])
        param_grid = {
            'sampler__k_neighbors': [3, 5, 7, 9],
            'sampler__top_k_features': [3, 5, 7, 10]
        }
    elif smote_type == "smote":
        pipe = Pipeline([('sampler', SMOTE(random_state=42)), ('clf', rf)])
        param_grid = {'sampler__k_neighbors': [3, 5, 7, 9]}
    elif smote_type == "adasyn":
        pipe = Pipeline([('sampler', ADASYN(random_state=42)), ('clf', rf)])
        param_grid = {'sampler__n_neighbors': [3, 5, 7, 9]}
    elif smote_type == "borderline":
        pipe = Pipeline([('sampler', BorderlineSMOTE(random_state=42)), ('clf', rf)])
        param_grid = {'sampler__k_neighbors': [3, 5, 7, 9], 'sampler__kind': ['borderline-1', 'borderline-2']}
    else:
        raise ValueError("smote_type debe ser uno de: 'customsmote', 'smote', 'adasyn', 'borderline'")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='weighted')

    grid = GridSearchCV(pipe, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=0, error_score=np.nan)
    
    try:
        grid.fit(X_train, y_train)

    except Exception as e:
        print(f"{smote_type} failed on this dataset.")
        return None
    
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }

def optimize_models_parameters(X_dict,y_dict,models,param_grids,results_csv):

    """
    Función para optimizar los hiperparámetros de varios modelos usando GridSearchCV.
    
    Parámetros:
    -----------
    X_dict : dict
        Diccionario con los datasets de entrada (features), donde cada clave es el nombre del dataset 
        y el valor es un DataFrame/array con las variables independientes.
    y_dict : dict
        Diccionario con las variables objetivo correspondientes a cada dataset.
    models : dict
        Diccionario con los modelos a entrenar, donde la clave es el nombre del modelo 
        y el valor es la instancia del estimador.
    param_grids : dict
        Diccionario con las rejillas de hiperparámetros para cada modelo, 
        compatible con GridSearchCV.
    results_csv : str
        Ruta al archivo CSV donde se almacenarán los resultados. Si ya existe, 
        se cargarán los resultados previos y se evitará recalcular los mismos modelos.

    Descripción:
    ------------
    - Para cada dataset y modelo, verifica si ya existe un resultado guardado en el CSV.
    - Si no existe, entrena el modelo usando validación cruzada estratificada con 3 particiones.
    - Optimiza los hiperparámetros con GridSearchCV usando como métrica principal el F1 Macro.
    - Guarda los mejores parámetros y el mejor score en memoria y en el CSV.
    - Evita recalcular combinaciones ya evaluadas.

    Retorna:
    --------
    None. Los resultados se guardan en el CSV y se muestran en pantalla.
    """
    
    results_path = os.path.join("Resultados", results_csv)
    
    # Si ya existe el archivo, lo cargamos, si no, lo creamos vacío
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
    else:
        df_results = pd.DataFrame(columns=["Dataset", "Modelo", "Mejores Parametros", "CV F1 Macro"])

    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Resultados en memoria
    results_models = {}

    for ds_name in X_dict.keys():
        print(f"\n🔹 Dataset: {ds_name}")
        results_models[ds_name] = {}

        X = X_dict[ds_name]
        y = y_dict[ds_name]

        for model_name, model in models.items():
            # Verificar si ya se calculó este dataset+modelo
            exists = ((df_results["Dataset"] == ds_name) & (df_results["Modelo"] == model_name)).any()

            if exists:
                print(f"   ⏩ {model_name} ya optimizado, saltando...")
                continue

            print(f"   ⚙️ Optimizando {model_name}...")

            if model_name == "TabNet":
                X = X.values if hasattr(X, "values") else X
                y = y.values if hasattr(y, "values") else y
                
            
            # Si es SVM reducir tamaño a 2000 manteniendo el desbalance
            if model_name == "SVM":
                if len(X) > 200:
                    X, _, y, _ = train_test_split(
                        X, y, train_size=200, stratify=y, random_state=42
                    )
                    
            #if model_name == "TabTransformer":
             #   model = TabTransformerClassifier(num_features=X.shape[1])
            
            # Definir GridSearchCV
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=cv,
                scoring="f1_macro",   # métrica principal
                n_jobs=-1,
                verbose=0,
                error_score=np.nan
            )

            # Entrenamiento
            
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.int64, copy=True)


            grid.fit(X, y)

            # Guardar en memoria
            results_models[ds_name][model_name] = {
                "best_params": grid.best_params_,
                "best_score": grid.best_score_
            }

            # Guardar en CSV
            df_results = pd.concat([
                df_results,
                pd.DataFrame([{
                    "Dataset": ds_name,
                    "Modelo": model_name,
                    "Mejores Parametros": grid.best_params_,
                    "CV F1 Macro": grid.best_score_
                }])
            ], ignore_index=True)

            df_results.to_csv(results_path, index=False)

    print("\n✅ Optimización completada")
    #display(df_results)


from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

def optimize_models_parameters(X, y, models, param_grids):
    """
    Optimiza hiperparámetros de varios modelos usando GridSearchCV para un dataset.
    
    :param X: pd.DataFrame o np.array, features de entrenamiento
    :param y: pd.Series o np.array, etiquetas de entrenamiento
    :param models: dict, modelos a entrenar
    :param param_grids: dict, grids de hiperparámetros para cada modelo
    :return: dict, {model_name: {"best_params": ..., "best_score": ...}}
    """
    
    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"   ⚙️ Optimizing {model_name} ...")
        
        # Manejo especial si el modelo requiere array en lugar de DataFrame
        X_fit = X.values if hasattr(X, "values") else X
        y_fit = y.values if hasattr(y, "values") else y
        
        # Ajustes especiales
        if model_name == "SVM" and len(X_fit) > 2000:
            X_fit, _, y_fit, _ = train_test_split(
                X_fit, y_fit, train_size=2000, stratify=y_fit, random_state=42
            )
        #if model_name == "TabTransformer":
        #    model = TabTransformerClassifier(num_features=X_fit.shape[1])
        
        # GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
            error_score=np.nan
        )
        
        X_fit = np.array(X_fit, dtype=np.float32, copy=True)
        y_fit = np.array(y_fit, dtype=np.int64, copy=True)
        
        grid.fit(X_fit, y_fit)
        
        results[model_name] = grid.best_params_
            
        
        
    return results


# Función que evalua un modelo sobre un conjunto de datos y devuelve las métricas de evaluación en un diccionario
def evaluate_model(name, model, X, y, kf):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0, average='weighted'),
        "recall": make_scorer(recall_score, zero_division=0, average='weighted'),
        "f1": make_scorer(f1_score, zero_division=0, average='weighted'),
        #"roc_auc": make_scorer(roc_auc_score, multi_class='ovr', average='weighted')
    }
    scores = cross_validate(model, X, y, cv=kf, scoring=scoring)

    results = {
        "Model": name,
        "Accuracy": scores['test_accuracy'].mean(),
        "Precision": scores['test_precision'].mean(),
        "Recall": scores['test_recall'].mean(),
        "F1": scores['test_f1'].mean()
        #"ROC AUC": scores['test_roc_auc'].mean()
    }

    return results


def evaluate_models(X_dict, y_dict, best_params, results_csv, model_list=None):
    """
    Evalúa los modelos especificados en múltiples datasets.
    
    Parámetros
    ----------
    X_dict, y_dict : dict
        Diccionarios con los datasets (X e y por cada dataset).
    best_params : dict
        Diccionario con los mejores parámetros para cada modelo y dataset.
    results_csv : str
        Nombre del archivo CSV donde se guardarán los resultados.
    model_list : list o None
        Lista de modelos a evaluar. 
        Si None, se evalúan todos los modelos disponibles.
    """
    
    results_path = os.path.join("Resultados", results_csv)

    # Cargar CSV previo si existe
    if os.path.exists(results_path):
        eval_df = pd.read_csv(results_path)
    else:
        eval_df = pd.DataFrame(columns=["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"])

    # Validación cruzada estratificada
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for ds_name in X_dict.keys():
        X, y = X_dict[ds_name], y_dict[ds_name]

        all_models = {}

        if "Decision Tree" in best_params[ds_name]:
            all_models["Decision Tree"] = DecisionTreeClassifier(
                **best_params[ds_name]["Decision Tree"]["best_params"], random_state=42
            )

        if "Random Forest" in best_params[ds_name]:
            all_models["Random Forest"] = RandomForestClassifier(
                **best_params[ds_name]["Random Forest"]["best_params"], random_state=42
            )

        if "Bagging" in best_params[ds_name]:
            all_models["Bagging"] = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                **best_params[ds_name]["Bagging"]["best_params"],
                random_state=42
            )

        if "XGBoost" in best_params[ds_name]:
            all_models["XGBoost"] = XGBClassifier(
                **best_params[ds_name]["XGBoost"]["best_params"], random_state=42
            )

        if "MLP" in best_params[ds_name]:
            all_models["MLP"] = MLPClassifier(
                max_iter=500, random_state=42,
                **best_params[ds_name]["MLP"]["best_params"]
            )

        if "TabNet" in best_params[ds_name]:
            all_models["TabNet"] = TabNetClassifier(verbose=0, seed=42)

        if "SVM" in best_params[ds_name]:
            all_models["SVM"] = SVC(
                probability=True, random_state=42,
                **best_params[ds_name]["SVM"]["best_params"]
            )

        if "LightGBM" in best_params[ds_name]:
            all_models["LightGBM"] = LGBMClassifier(
                random_state=42, **best_params[ds_name]["LightGBM"]["best_params"]
            )
            
        #if "TabTransformer" in best_params[ds_name]:
        #    all_models["TabTransformer"] = TabTransformerClassifier(
        #        num_features=X.shape[1], **best_params[ds_name]["TabTransformer"]["best_params"]
        #    )

        # Si el usuario especifica una lista, filtramos
        models = {name: model for name, model in all_models.items() if (model_list is None or name in model_list)}

        for model_name, model in models.items():
            if ((eval_df["Dataset"] == ds_name) & (eval_df["Model"] == model_name)).any():
                print(f"⚠️ {model_name} en {ds_name} ya evaluado, saltando...")
                continue

            print(f"✅ Evaluando {model_name} en {ds_name}...")

            try:
                X = np.array(X, dtype=np.float32, copy=True)
                y = np.array(y, dtype=np.int64, copy=True)
                result = evaluate_model(model_name, model, X, y, kf)
                result["Dataset"] = ds_name

                # Añadir al DataFrame y guardar
                eval_df = pd.concat([eval_df, pd.DataFrame([result])], ignore_index=True)
                eval_df.to_csv(results_path, index=False)
            except Exception as e:
                print(f"❌ Error evaluando {model_name} en {ds_name}: {e}")

    print("🎉 Evaluación completada. Resultados guardados en", results_path)


def evaluate_models(X, y, models, best_model_params, kf_splits=10):
    """
    Evalúa un conjunto de modelos en un dataset dado usando los mejores parámetros.
    
    :param X: pd.DataFrame o np.array, features
    :param y: pd.Series o np.array, labels
    :param models: dict de modelos, {nombre: instancia_clase_sklearn}
    :param best_model_params: dict con mejores parámetros por modelo {nombre: dict de params}
    :param kf_splits: int, número de splits para StratifiedKFold
    :return: dict {modelo: métricas}
    """
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=kf_splits, shuffle=True, random_state=42)
    
    results = {}
    
    for model_name, model_class in models.items():
        if model_name not in best_model_params:
            print(f"⚠️ No hay mejores parámetros para {model_name}, saltando...")
            continue
        
        # Reconstruir modelo con los mejores parámetros
        params = best_model_params[model_name]
        model = model_class.__class__(**params)
        
        X_arr = np.array(X, dtype=np.float32, copy=True)
        y_arr = np.array(y, dtype=np.int64, copy=True)
        
        # Asumimos que tienes una función `evaluate_model` que devuelve métricas
        metrics = evaluate_model(model_name, model, X_arr, y_arr, kf)
        results[model_name] = metrics
    
    return results
