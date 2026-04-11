"""
Model architecture and training loop for Iris Classification.

Trains a simple MLP classifier on the Iris dataset.
Must print METRIC=<float> and VRAM_MB=<int> to stdout at end.
METRIC = validation accuracy (higher is better, range 0.0-1.0).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

DATA_DIR = ".cache/iris"

# --- HYPERPARAMETERS (agent may tune these) ---
HIDDEN_LAYER_SIZES = (64, 32)
ACTIVATION = "relu"          # relu | tanh | logistic
SOLVER = "adam"
LEARNING_RATE_INIT = 0.001
MAX_ITER = 500
ALPHA = 0.0001               # L2 regularisation


def run_train() -> None:
    # Ensure data exists
    if not os.path.exists(f"{DATA_DIR}/X_train.npy"):
        import prepare
        prepare.run_prepare()

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        alpha=ALPHA,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"[train] Val accuracy: {accuracy:.4f}")
    print(f"METRIC={accuracy:.6f}", flush=True)
    print(f"VRAM_MB=0", flush=True)


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
