"""
Data loading and preprocessing for ECG Heartbeat Classification (MIT-BIH subset).

Downloads the ECG5000 dataset from the UCR Time Series Archive — a multiclass
arrhythmia classification task with 5000 samples of 140-timestep ECG windows.
5 classes: Normal, R-on-T PVC, PVC, SP, UB.

This is the closest publicly available analogue to the EEG task: multivariate
time-series, medical signal, class imbalance, and meaningful temporal structure.
"""

import numpy as np
import os
import urllib.request

DATA_DIR = ".cache/ecg5000"

# UCR ECG5000 direct links (tab-separated, last column = label 1-indexed)
TRAIN_URL = "https://raw.githubusercontent.com/cauchyturing/UCR_UEA_archive_loader/master/data/ECG5000/ECG5000_TRAIN.txt"
TEST_URL  = "https://raw.githubusercontent.com/cauchyturing/UCR_UEA_archive_loader/master/data/ECG5000/ECG5000_TEST.txt"

# --- PREPROCESSING CONFIG (agent may tune) ---
WINDOW_SIZE = 140        # fixed by dataset
NORMALISE = "zscore"     # zscore | minmax | none
VAL_FRACTION = 0.2


def _download(url: str, dest: str) -> None:
    print(f"[prepare] Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _load_txt(path: str):
    data = np.loadtxt(path)
    X = data[:, 1:]          # all columns except first = signal
    y = data[:, 0].astype(int) - 1  # labels are 1-indexed → 0-indexed
    return X, y


def run_prepare() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = f"{DATA_DIR}/train.txt"
    test_path  = f"{DATA_DIR}/test.txt"

    if not os.path.exists(train_path):
        _download(TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        _download(TEST_URL, test_path)

    X_train_raw, y_train_raw = _load_txt(train_path)
    X_test,  y_test  = _load_txt(test_path)

    # Use the provided test split as val; shuffle train
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train_raw))
    X_train_raw, y_train_raw = X_train_raw[idx], y_train_raw[idx]

    if NORMALISE == "zscore":
        mean = X_train_raw.mean(axis=1, keepdims=True)
        std  = X_train_raw.std(axis=1, keepdims=True) + 1e-8
        X_train = (X_train_raw - mean) / std
        mean_v = X_test.mean(axis=1, keepdims=True)
        std_v  = X_test.std(axis=1, keepdims=True) + 1e-8
        X_val  = (X_test - mean_v) / std_v
    elif NORMALISE == "minmax":
        mn = X_train_raw.min(axis=1, keepdims=True)
        mx = X_train_raw.max(axis=1, keepdims=True)
        X_train = (X_train_raw - mn) / (mx - mn + 1e-8)
        mn_v = X_test.min(axis=1, keepdims=True)
        mx_v = X_test.max(axis=1, keepdims=True)
        X_val  = (X_test - mn_v) / (mx_v - mn_v + 1e-8)
    else:
        X_train, X_val = X_train_raw, X_test

    # Add channel dimension → (N, 1, 140) for conv models
    X_train = X_train[:, np.newaxis, :]
    X_val   = X_val[:,   np.newaxis, :]

    y_train = y_train_raw

    np.save(f"{DATA_DIR}/X_train.npy", X_train.astype(np.float32))
    np.save(f"{DATA_DIR}/X_val.npy",   X_val.astype(np.float32))
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_val.npy",   y_test)

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[prepare] Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"[prepare] Class distribution (train): {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    run_prepare()
