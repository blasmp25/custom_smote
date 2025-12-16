import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
#from tab_transformer_pytorch import TabTransformer
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import time
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from src.custom_smote import CustomSMOTE


# Función que lee los resultados de un csv y los carga en un diccionario
def csv_to_dict(csv):
    # Leer CSV
    df = pd.read_csv(csv)

    # Convertir strings a diccionarios reales
    df["Mejores Parametros"] = df["Mejores Parametros"].apply(ast.literal_eval)

    # Construir diccionario jerárquico
    results_dict = {}

    for _, row in df.iterrows():
        dataset = row["Dataset"]
        model = row["Modelo"]
        params = row["Mejores Parametros"]
        score = row["CV F1 Macro"]

        if dataset not in results_dict:
            results_dict[dataset] = {}
        
        results_dict[dataset][model] = {
            "best_params": params,
            "best_score": score
        }
        
    return results_dict

def measure_default_times(X_dict, y_dict, sampling_technique, params, n_runs=10):
    """
    Calcula el tiempo de ejecución de SMOTE, Borderline-SMOTE, ADASYN y Custom SMOTE
    usando los parámetros por defecto.

    Args:
        X_dict (dict): diccionario con datasets de entrada, {nombre: X}.
        y_dict (dict): diccionario con etiquetas, {nombre: y}.
        custom_smote_class (class, optional): clase de tu Custom SMOTE. Si None, se omite.
        n_runs (int): número de veces que se va a ejecutar cada técnica.

    Returns:
        dict: tiempos de ejecución, estructura {metodo: {dataset: tiempo}}
    """
    
    start = time.time()
    for _ in range(n_runs):
        for ds_name in X_dict:
            X = X_dict[ds_name]
            y = y_dict[ds_name]
            
            if sampling_technique == "SMOTE":
                neighbors = params[ds_name]['best_params']['sampler__k_neighbors']
                sampler = SMOTE(k_neighbors=neighbors, random_state=42)
                
            if sampling_technique == "ADASYN":
                neighbors = params[ds_name]['best_params']['sampler__n_neighbors']
                sampler = ADASYN(n_neighbors=neighbors, random_state=42)
                
            if sampling_technique == "BORDERLINE":
                neighbors = params[ds_name]['best_params']['sampler__k_neighbors']
                kind = params[ds_name]['best_params']['sampler__kind']
                sampler = BorderlineSMOTE(k_neighbors=neighbors,kind=kind, random_state=42)
                
            if sampling_technique == "CUSTOM":
                neighbors = params[ds_name]['best_params']['sampler__k_neighbors']
                top_k = params[ds_name]['best_params']['sampler__top_k_features']
                sampler = CustomSMOTE(k_neighbors=neighbors, top_k_features=top_k, random_state=42)

            _,_ = sampler.fit_resample(X,y)

    end = time.time()
    
    average_time = (end - start) / n_runs
    
    return average_time
 
"""    
##############################   Modelo TabTransformer
class TabTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_features, dim=32, dim_out=2, depth=6, epochs=10, lr=1e-3, batch_size=32, device=None):
        # 🔹 Solo guardamos parámetros, no creamos modelo
        self.num_features = num_features
        self.dim = dim
        self.dim_out = dim_out
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def _build_model(self):
        self.model = TabTransformer(
            categories=(),
            num_continuous=self.num_features,
            dim=self.dim,
            dim_out=self.dim_out,
            depth=self.depth,
            heads=4
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        # Convertir a numpy si es DataFrame
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
            
        self.classes_ = np.unique(y)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        self._build_model()
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb_cont, yb in loader:
                xb_cat = torch.zeros((xb_cont.shape[0], 0), dtype=torch.long).to(self.device)
                outputs = self.model(xb_cat, xb_cont)
                loss = self.loss_fn(outputs, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self


    def predict(self, X):
        self.model.eval()
        if hasattr(X, "values"):
            X = X.values
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            xb_cat = torch.zeros((X_tensor.shape[0], 0), dtype=torch.long).to(self.device)
            outputs = self.model(xb_cat, X_tensor)

            if outputs.shape[1] == 1:  # caso binario con una salida
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:  # clasificación multiclase
                preds = outputs.argmax(dim=1)

        return preds.cpu().numpy()


    def predict_proba(self, X):
        self.model.eval()
        if hasattr(X, "values"):
            X = X.values
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            xb_cat = torch.zeros((X_tensor.shape[0], 0), dtype=torch.long).to(self.device)
            outputs = self.model(xb_cat, X_tensor)

            if outputs.shape[1] == 1:  # binario
                probs = torch.sigmoid(outputs)
                probs = torch.cat([1 - probs, probs], dim=1)  # [p0, p1]
            else:  # multiclase
                probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()


    def get_params(self, deep=True):
        return {
            "num_features": self.num_features,
            "dim": self.dim,
            "dim_out": self.dim_out,
            "depth": self.depth,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
"""