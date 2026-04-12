"""
Data loading and preprocessing for Iris Classification.

Loads the Iris dataset from sklearn, splits into train/val sets,
and saves as numpy arrays for train.py to consume.
"""

import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = ".cache/iris"


def run_prepare() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target  # (150, 4), (150,)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    np.save(f"{DATA_DIR}/X_train.npy", X_train)
    np.save(f"{DATA_DIR}/X_val.npy", X_val)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_val.npy", y_val)

    print(f"[prepare] Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"[prepare] Classes: {iris.target_names.tolist()}")


if __name__ == "__main__":
    run_prepare()
