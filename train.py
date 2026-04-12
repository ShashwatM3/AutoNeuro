import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import os

DATA_DIR = ".cache/iris"  # This path should be updated to reflect the correct dataset

# --- HYPERPARAMETERS (agent may tune these) ---
N_ESTIMATORS = 100  # Number of boosting stages
LEARNING_RATE = 0.1  # Step size shrinkage
MAX_DEPTH = 3  # Maximum depth of individual regression estimators

def run_train() -> None:
    # Ensure data exists
    if not os.path.exists(f"{DATA_DIR}/X_train.npy"):
        import prepare
        prepare.run_prepare()

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    # Create and train the Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    print(f"[train] Val R² score: {r2:.6f}")
    print(f"METRIC={r2:.6f}", flush=True)
    print(f"VRAM_MB=0", flush=True)


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
