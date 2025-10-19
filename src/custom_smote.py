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
    _parameter_constraints = {}  # Esto permite usar GridSearchCV en el futuro
    
    def __init__(self, k_neighbors=5, top_k_features=5, random_state=42):
        super().__init__(k_neighbors=k_neighbors)
        self.top_k_features = top_k_features
        self.random_state = random_state

    def fit_resample(self, X, y):
        
        """
        Genera muestras sintéticas para balancear un conjunto de datos desbalanceado utilizando una
        variante personalizada de SMOTE que incorpora información mutua para seleccionar atributos relevantes.

        Para cada muestra de la clase minoritaria, se generan nuevas muestras tomando los valores
        mínimo, máximo y mediana de las columnas más correlacionadas (según información mutua)
        dentro de sus vecinos más cercanos.

        Parámetros:
        ----------
        X : pandas.DataFrame
            Conjunto de características (features).
        y : pandas.Series
            Vector de etiquetas correspondiente a X.
        k_neighbors : int, opcional (por defecto=5)
            Número de vecinos a considerar al buscar similitudes entre muestras minoritarias.
        top_k_features : int, opcional (por defecto=5)
            Número de características más relevantes (según Mutual Information) a modificar para generar muestras sintéticas.

        Retorna:
        -------
        X_final : pandas.DataFrame
            Conjunto de características balanceado con muestras sintéticas añadidas.
        y_final : pandas.Series
            Etiquetas correspondientes al conjunto X_final.
        """  
        
        # Convertir a DataFrame/Series si vienen como numpy
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Identificar clases
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        samples_needed = class_counts[majority_class] - class_counts[minority_class]

        X_minority = X[y == minority_class].reset_index(drop=True)
        X_minority_values = X_minority.values  # Convertir a numpy para cálculo más rápido
        
        # Mutual Information (solo en min.)
        mi = mutual_info_classif(X_minority, y[y == minority_class], discrete_features=False)
        top_indices = np.argsort(mi)[-self.top_k_features:]

        # Vecinos
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X_minority)
        neighbors = nn.kneighbors(X_minority, return_distance=False)

        synthetic_samples = []
        neighbor_pointer = 1  # empieza con el más cercano

        while len(synthetic_samples) < samples_needed:
            for idx, neighbor_idx in enumerate(neighbors):
                if len(synthetic_samples) >= samples_needed:
                    break

                # Seleccionamos un vecino en rotación
                current_neighbor = neighbor_idx[neighbor_pointer % (self.k_neighbors + 1)]

                # Evitar que sea la propia muestra
                if current_neighbor == idx:
                    current_neighbor = neighbor_idx[(neighbor_pointer + 1) % (self.k_neighbors + 1)]

                # Tomar solo las columnas relevantes de los dos vecinos (original + vecino)
                vals = X_minority_values[[idx, current_neighbor], :][:, top_indices]

                # Calcular min, max y mediana de estas columnas relevantes
                mins = vals.min(axis=0)
                maxs = vals.max(axis=0)
                meds = np.median(vals, axis=0)

                # Base de la nueva muestra = copia de la muestra original
                base = X_minority_values[idx].copy()

                # Generar hasta 3 muestras: usando min, max y mediana
                for replacement in [mins, maxs, meds]:
                    if len(synthetic_samples) >= samples_needed:
                        break  # Parar si ya tenemos suficientes
                    new_sample = base.copy()
                    new_sample[top_indices] = replacement  # Reemplazar solo las columnas más importantes
                    synthetic_samples.append(new_sample)   # Añadir a la lista

            neighbor_pointer += 1  # Cambiar de vecino para la siguiente iteración

        # Dataset con las muestras sintéticas
        X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
        y_synthetic = pd.Series([minority_class] * len(X_synthetic), name=y.name if y.name else "target")

        # Unir las muestras sintéticas con los datos originales
        X_final = pd.concat([X, X_synthetic], ignore_index=True)
        y_final = pd.concat([y, y_synthetic], ignore_index=True)

        return X_final, y_final
        