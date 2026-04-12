"""
Model architecture and training loop for Iris Classification.

Trains a simple MLP classifier on the Iris dataset.
Must print METRIC=<float> and VRAM_MB=<int> to stdout at end.
METRIC = validation accuracy (higher is better, range 0.0-1.0).
"""

import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

DATA_DIR = ".cache/iris"

# --- HYPERPARAMETERS (agent may tune these) ---
SVC_KERNEL = 'rbf'  # Linear kernel didn't reach the performance target
SVC_C = 1.0  # Regularization parameter
SVC_GAMMA = 'scale'  # Kernel coefficient


def run_train() -> None:
    # Ensure data exists
    if not os.path.exists(f"{DATA_DIR}/X_train.npy"):
        import prepare

        prepare.run_prepare()

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val = np.load(f"{DATA_DIR}/y_val.npy")

    model = SVC(kernel=SVC_KERNEL, C=SVC_C, gamma=SVC_GAMMA, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"[train] Val accuracy: {accuracy:.4f}")
    print(f"METRIC={accuracy:.6f}", flush=True)
    print("VRAM_MB=0", flush=True)


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
