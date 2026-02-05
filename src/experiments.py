import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from .custom_smote import CustomSMOTE


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
    scorer = make_scorer(f1_score, average='macro')

    grid = GridSearchCV(pipe, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=0, error_score=np.nan)
    
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


def evaluate_models(X_train, y_train, X_test, y_test, models, best_model_params):
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