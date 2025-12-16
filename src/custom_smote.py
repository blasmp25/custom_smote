import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif

"""
En este archivo .py se implementa la variante propuesta de SMOTE que toma en cuenta la correlación de variables mediante 
una clase. La clase recibe como parámetros de entrada el número de vecinos que se tomarán para generar las muestras así
como el número de características que se tendrán en cuenta. También recibe una semilla para poder controlar la reproducibilidad.
"""

class CustomSMOTE(SMOTE):
    """
    Variante personalizada de SMOTE para datasets multiclase, que selecciona columnas relevantes
    mediante información mutua y genera muestras sintéticas usando min, max y mediana.

    Parámetros:
    -----------
    k_neighbors : int, por defecto=5
        Número de vecinos a considerar.
    top_k_features : int, por defecto=5
        Número de características más relevantes a modificar.
    random_state : int, por defecto=42
        Semilla para reproducibilidad.
    sampling_strategy : 'auto' o dict
        'auto': genera muestras para igualar la clase mayoritaria.
        dict: diccionario {clase: n_muestras} para controlar el número de sintéticas por clase.
    """

    _parameter_constraints = {}  # Para compatibilidad con GridSearchCV

    def __init__(self, k_neighbors=5, top_k_features=5, random_state=42, sampling_strategy='auto'):
        super().__init__(k_neighbors=k_neighbors)
        self.top_k_features = top_k_features
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_classes = [c for c in class_counts.index if c != majority_class]

        synthetic_samples = []
        synthetic_labels = []

        
        discrete_mask = []
        for col in X.columns:
            if X[col].dtype.name in ["object", "category", "bool"]:
                discrete_mask.append(True)
            elif np.issubdtype(X[col].dtype, np.integer):
                discrete_mask.append(X[col].nunique() <= 20)
            else:
                discrete_mask.append(False)
        
        
        
        
        
        for minority_class in minority_classes:
            y_ovr = (y==minority_class).astype(int)
            mi = mutual_info_classif(X, y_ovr, discrete_features=discrete_mask)
            top_indices = np.argsort(mi)[-self.top_k_features:]
            
            
            X_minority = X[y == minority_class].reset_index(drop=True)
            X_minority_values = X_minority.values

            
            
            # Determinar cuántas muestras generar
            if self.sampling_strategy == 'auto':
                samples_needed = class_counts[majority_class] - class_counts[minority_class]
            elif isinstance(self.sampling_strategy, dict):
                samples_needed = self.sampling_strategy.get(minority_class, 0) - class_counts[minority_class]
            else:
                raise ValueError("sampling_strategy debe ser 'auto' o un diccionario {clase: n_muestras}")

            if samples_needed <= 0:
                continue

            

            # Vecinos
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
            nn.fit(X_minority)
            neighbors = nn.kneighbors(X_minority, return_distance=False)

            neighbor_pointer = 1
            generated = 0

            while generated < samples_needed:
                for idx, neighbor_idx in enumerate(neighbors):
                    if generated >= samples_needed:
                        break

                    current_neighbor = neighbor_idx[neighbor_pointer % len(neighbor_idx)]
                    if current_neighbor == idx:
                        current_neighbor = neighbor_idx[(neighbor_pointer + 1) % len(neighbor_idx)]

                    vals = X_minority_values[[idx, current_neighbor], :][:, top_indices]
                    mins = vals.min(axis=0)
                    maxs = vals.max(axis=0)
                    meds = np.median(vals, axis=0)

                    base = X_minority_values[idx].copy()

                    for replacement in [mins, maxs, meds]:
                        if generated >= samples_needed:
                            break
                        new_sample = base.copy()
                        new_sample[top_indices] = replacement
                        synthetic_samples.append(new_sample)
                        synthetic_labels.append(minority_class)
                        generated += 1

                neighbor_pointer += 1

        X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
        y_synthetic = pd.Series(synthetic_labels, name=y.name if y.name else "target")

        X_final = pd.concat([X, X_synthetic], ignore_index=True)
        y_final = pd.concat([y, y_synthetic], ignore_index=True)

        return X_final, y_final