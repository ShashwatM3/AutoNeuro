"""
Model and training loop for ECG Heartbeat Classification.

1D CNN classifier on ECG5000 (140-timestep windows, 5 arrhythmia classes).
Must print METRIC=<float> and VRAM_MB=<int> to stdout at end.
METRIC = macro-averaged F1 score (higher is better, handles class imbalance).
"""

import numpy as np
import os

DATA_DIR = ".cache/ecg5000"
NUM_CLASSES = 5
SEQ_LEN = 140

# --- HYPERPARAMETERS (agent may tune) ---
BATCH_SIZE      = 64
EPOCHS          = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4

# CNN architecture
CONV_CHANNELS   = [32, 64, 128]   # filters per conv block
KERNEL_SIZE     = 5
DROPOUT         = 0.3
FC_HIDDEN       = 128


def build_model(seq_len: int, num_classes: int):
    try:
        import torch
        import torch.nn as nn

        class ECGNet(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                in_ch = 1
                cur_len = seq_len
                for out_ch in CONV_CHANNELS:
                    layers += [
                        nn.Conv1d(in_ch, out_ch, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Dropout(DROPOUT),
                    ]
                    in_ch = out_ch
                    cur_len = cur_len // 2
                self.conv = nn.Sequential(*layers)
                flat = in_ch * cur_len
                self.fc = nn.Sequential(
                    nn.Linear(flat, FC_HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(FC_HIDDEN, num_classes),
                )

            def forward(self, x):
                return self.fc(self.conv(x).flatten(1))

        return ECGNet()

    except ImportError:
        return None


def run_train_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    X_train = torch.tensor(np.load(f"{DATA_DIR}/X_train.npy"))
    y_train = torch.tensor(np.load(f"{DATA_DIR}/y_train.npy"), dtype=torch.long)
    X_val   = torch.tensor(np.load(f"{DATA_DIR}/X_val.npy"))
    y_val   = torch.tensor(np.load(f"{DATA_DIR}/y_val.npy"),   dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(SEQ_LEN, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"[train] Epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_dl):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()

    y_true = y_val.numpy()
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, preds, average="macro")

    vram = 0
    if torch.cuda.is_available():
        vram = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    print(f"[train] Val macro-F1: {f1:.4f}")
    print(f"METRIC={f1:.6f}", flush=True)
    print(f"VRAM_MB={vram}", flush=True)


def run_train_sklearn_fallback():
    """Fallback if PyTorch is not installed."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score

    X_train = np.load(f"{DATA_DIR}/X_train.npy").squeeze(1)  # (N, 140)
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy").squeeze(1)
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")

    print(f"[train] (sklearn fallback) Val macro-F1: {f1:.4f}")
    print(f"METRIC={f1:.6f}", flush=True)
    print(f"VRAM_MB=0", flush=True)


def run_train() -> None:
    if not os.path.exists(f"{DATA_DIR}/X_train.npy"):
        import prepare
        prepare.run_prepare()

    try:
        import torch
        run_train_torch()
    except ImportError:
        print("[train] PyTorch not found, using sklearn fallback")
        run_train_sklearn_fallback()


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
