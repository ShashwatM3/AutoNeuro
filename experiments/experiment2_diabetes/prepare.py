"""
Data loading and preprocessing for Diabetes Progression Regression.

Loads the sklearn Diabetes dataset (442 patients, 10 features),
splits into train/val, and saves as numpy arrays.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = ".cache/diabetes"

# --- PREPROCESSING CONFIG (agent may tune) ---
VAL_SPLIT = 0.2
POLYNOMIAL_FEATURES = False   # set True to add degree-2 interaction terms
POLY_DEGREE = 2


def run_prepare() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    data = load_diabetes()
    X, y = data.data, data.target  # (442, 10), continuous target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    if POLYNOMIAL_FEATURES:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_val   = poly.transform(X_val)
        print(f"[prepare] Polynomial features: {X_train.shape[1]} features after degree-{POLY_DEGREE} expansion")

    np.save(f"{DATA_DIR}/X_train.npy", X_train)
    np.save(f"{DATA_DIR}/X_val.npy",   X_val)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_val.npy",   y_val)

    print(f"[prepare] Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"[prepare] Target range: {y.min():.1f} – {y.max():.1f}")


if __name__ == "__main__":
    run_prepare()
