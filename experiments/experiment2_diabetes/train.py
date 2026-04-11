"""
Model and training loop for Diabetes Progression Regression.

Predicts continuous disease progression score from 10 clinical features.
Must print METRIC=<float> and VRAM_MB=<int> to stdout at end.
METRIC = R² score (higher is better, range roughly 0.0–1.0).
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import os

DATA_DIR = ".cache/diabetes"

# --- HYPERPARAMETERS (agent may tune) ---
HIDDEN_LAYER_SIZES = (128, 64, 32)
ACTIVATION = "relu"           # relu | tanh | logistic
SOLVER = "adam"
LEARNING_RATE_INIT = 0.001
MAX_ITER = 1000
ALPHA = 0.001                  # L2 regularisation
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 20


def run_train() -> None:
    if not os.path.exists(f"{DATA_DIR}/X_train.npy"):
        import prepare
        prepare.run_prepare()

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        alpha=ALPHA,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    print(f"[train] Val R²: {r2:.4f}")
    print(f"METRIC={r2:.6f}", flush=True)
    print(f"VRAM_MB=0", flush=True)


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
