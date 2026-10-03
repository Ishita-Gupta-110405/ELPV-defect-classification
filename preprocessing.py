import numpy as np
from src.elpv_dataset.utils import load_dataset
from sklearn.model_selection import train_test_split

def get_splits():
    # 1. Dataset Acquisition
    print("Loading ELPV dataset...")
    images, proba, types = load_dataset()
    
    # 2. Preprocessing & Encoding
    # Map probabilities (0.0, 0.33, 0.67, 1.0) to discrete classes (0, 1, 2, 3)
    classes = np.unique(proba)
    class_mapping = {val: idx for idx, val in enumerate(classes)}
    y = np.array([class_mapping[p] for p in proba])
    
    # 3. Stratified Train/Val/Test Split (70% / 15% / 15%)
    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1764, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test