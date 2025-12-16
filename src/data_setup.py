import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

DATASET_DIR = "datasets/"

def load_datasets(names: Optional[List[str]]=None) -> Dict[str, pd.DataFrame]:
    """
    Loads one or multiple datasets from a fixed directory.

    :param names: List of dataset filenames to load.
                  If None, loads all CSV files in the directory.
    :return: Dictionary {filename: dataframe}
    """
    
    print("[INFO] Loading datasets...")
    
    # If no names were provided, load all CSV files in the directory
    if names is None:
        names = [
            f for f in os.listdir(DATASET_DIR)
            if os.path.isfile(os.path.join(DATASET_DIR, f)) and f.endswith(".csv")
        ]
        
    datasets = {}
    
    for name in names:
        path = os.path.join(DATASET_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{name}' does not exist in {DATASET_DIR}")
        
        try:
            df = pd.read_csv(path)
            datasets[name] = df
            
        except Exception as e:
            raise RuntimeError(f"Failed to read '{name}': {e}")
        
    return datasets


def split_X_y(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Splits each dataset into X (all columns except last) and y (last column).

    :param datasets: Dictionary {filename: dataframe}
    :return: Dictionary {filename: (X, y)}
    """
    
    result = {}
    
    for name, df in datasets.items():
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        result[name] = (X,y)
        
    return result
    
    
def create_train_test_splits(XY_dict: Dict[str, Tuple],
                             test_size: float = 0.3,
                             seed: int = 42) -> Dict[str,Dict[str,object]]:
    """
    Creates train/test splits for each dataset.

    :param XY_dict: Dictionary {filename: (X, y)} from split_X_y()
    :param test_size: Fraction for test set size (default 0.3)
    :param seed: Random seed for reproducibility
    :return: Dictionary {filename: {"X_train", "X_test", "y_train", "y_test"}}
    """
    
    print("[INFO] Creating training and testing data...")
    
    result = {}
    
    for name, (X,y) in XY_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        
        result[name] = {
            "X_train" : X_train,
            "X_test" : X_test,
            "y_train" : y_train,
            "y_test" : y_test
        }
        
    return result


def apply_feature_selection(
    split_dict: Dict[str, Dict[str, object]],
    k: int = 40
) -> Dict[str, Dict[str, object]]:
    """
    Applies feature selection to the datasets in split_dict.
    Returns the same dictionary structure, replacing X_train/X_test
    with the selected features. y_train/y_test remain unchanged.

    :param split_dict: Dict returned by create_train_test_splits()
    :param k: Number of features to keep if feature selection is applied
    :return: Updated split_dict with transformed X_train/X_test
    """
    
    print("[INFO] Applying feature selection if necessary...")
    
    for name, data in split_dict.items():
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        
        # If the dataset has more columns than k → apply feature selection
        if X_train.shape[1] > k:
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_train, y_train)
            
            selected_cols = X_train.columns[selector.get_support()]
            
            # update X_train and X_test
            data["X_train"] = pd.DataFrame(
                selector.transform(X_train),
                columns=selected_cols,
                index=X_train.index
            )
            
            data["X_test"] = pd.DataFrame(
                selector.transform(X_test),
                columns=selected_cols,
                index=X_test.index
            )
            
    return split_dict


def prepare_datasets(
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.3,
    seed: int = 42,
    k_features: int = 40
) -> Dict[str, Dict[str, object]]:
    """
    Master function to load datasets, split X/y, create train/test splits,
    and apply feature selection.
    
    :param dataset_names: List of dataset filenames to load (None = all in DATASET_DIR)
    :param test_size: Fraction for the test set
    :param seed: Random seed for reproducibility
    :param k_features: Number of features to keep when applying SelectKBest
    :return: Dictionary {dataset_name: {"X_train", "X_test", "y_train", "y_test"}}
    """
    
    datasets = load_datasets(dataset_names)
    XY = split_X_y(datasets)
    splits = create_train_test_splits(XY, test_size=test_size, seed=seed)
    splits = apply_feature_selection(splits, k=k_features)
    
    return splits
