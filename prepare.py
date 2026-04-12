import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import os

DATA_DIR = ".cache/diabetes"

def run_prepare() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target  # (442, 10), (442,)

    # Enable polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    np.save(f"{DATA_DIR}/X_train.npy", X_train)
    np.save(f"{DATA_DIR}/X_val.npy", X_val)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_val.npy", y_val)

    print(f"[prepare] Train: {X_train.shape}, Val: {X_val.shape}")

if __name__ == "__main__":
    run_prepare()
