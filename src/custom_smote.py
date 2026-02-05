import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif

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
        # Ensure data is in Pandas for easier column-wise manipulation
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_classes = [c for c in class_counts.index if c != majority_class]

        synthetic_samples = []
        synthetic_labels = []

        # --- STEP 1: Feature Type Detection ---
        # mutual_info_classif needs to know which features are discrete (categorical)
        # to apply the correct entropy-based calculation.
        discrete_mask = []
        for col in X.columns:
            if X[col].dtype.name in ["object", "category", "bool"]:
                discrete_mask.append(True)
            elif np.issubdtype(X[col].dtype, np.integer):
                # Heuristic: integers with few unique values are treated as discrete
                discrete_mask.append(X[col].nunique() <= 20)
            else:
                discrete_mask.append(False)
        
        
        # --- STEP 2: Iterate through each Minority Class ---
        for minority_class in minority_classes:
            # Calculate feature importance specifically for this class vs others
            y_ovr = (y==minority_class).astype(int)
            mi = mutual_info_classif(X, y_ovr, discrete_features=discrete_mask)
            
            # Identify indices of the 'top_k' most informative features
            top_indices = np.argsort(mi)[-self.top_k_features:]
            
            X_minority = X[y == minority_class].reset_index(drop=True)
            X_minority_values = X_minority.values

            
            # --- STEP 3: Determine Oversampling Volume ---
            if self.sampling_strategy == 'auto':
                samples_needed = class_counts[majority_class] - class_counts[minority_class]
            elif isinstance(self.sampling_strategy, dict):
                if minority_class not in self.sampling_strategy:
                    raise ValueError(
                        f"Class {minority_class} missing from sampling_strategy."
                    )
                target_n = self.sampling_strategy[minority_class]
                samples_needed = target_n - class_counts[minority_class]
            else:
                raise ValueError("sampling_strategy must be 'auto' or a dict.")

            
            if samples_needed <= 0:
                continue

            # --- STEP 4: Neighborhood Search ---
            # Find the k-nearest neighbors within the same minority class
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
            nn.fit(X_minority)
            neighbors = nn.kneighbors(X_minority, return_distance=False)

            neighbor_pointer = 1 # Used to cycle through different neighbors if needed
            generated = 0

            # --- STEP 5: Synthetic Sample Generation ---
            while generated < samples_needed:
                for idx, neighbor_idx in enumerate(neighbors):
                    if generated >= samples_needed:
                        break
                    
                    # Select a neighbor, ensuring we don't pick the point itself
                    current_neighbor = neighbor_idx[neighbor_pointer % len(neighbor_idx)]
                    if current_neighbor == idx:
                        current_neighbor = neighbor_idx[(neighbor_pointer + 1) % len(neighbor_idx)]

                    # Focus only on the 'important' features for the calculation
                    vals = X_minority_values[[idx, current_neighbor], :][:, top_indices]
                    
                    # Calculate statistical markers between the point and its neighbor
                    mins = vals.min(axis=0)
                    maxs = vals.max(axis=0)
                    meds = np.median(vals, axis=0)

                    base = X_minority_values[idx].copy()

                    # Generate up to 3 different synthetic versions per neighbor pair
                    for replacement in [mins, maxs, meds]:
                        if generated >= samples_needed:
                            break
                        new_sample = base.copy()
                        new_sample[top_indices] = replacement # Override only the most relevant features with statistical values
                        
                        synthetic_samples.append(new_sample)
                        synthetic_labels.append(minority_class)
                        generated += 1
                
                # Move to the next neighbor in the list to avoid repetitive points
                neighbor_pointer += 1

        # Combine original data with the new synthetic samples
        X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
        y_synthetic = pd.Series(synthetic_labels, name=y.name if y.name else "target")

        X_final = pd.concat([X, X_synthetic], ignore_index=True)
        y_final = pd.concat([y, y_synthetic], ignore_index=True)

        return X_final, y_final