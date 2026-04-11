# AutoNeuro — Experiment 3: ECG Arrhythmia Classification

## Goal
Maximise **macro-averaged F1 score** classifying 5 types of heartbeat arrhythmia
from 140-timestep ECG windows. This is a 1D time-series classification task with
significant class imbalance — the closest publicly available analogue to the
EEG brain signal task this system is ultimately built for.

## Metric
- **Name**: macro-F1 (macro-averaged F1 score across 5 classes)
- **Direction**: HIGHER IS BETTER
- **Range**: 0.0 – 1.0
- **Reason for macro-F1 over accuracy**: heavy class imbalance (~60% Normal),
  so accuracy is misleading. Macro-F1 penalises ignoring minority classes.
- **Target**: ≥ 0.85

## What the Coding Agent may change
Only `train.py` and `prepare.py`. Specifically:
- Architecture in `train.py`: `CONV_CHANNELS`, `KERNEL_SIZE`, `DROPOUT`,
  `FC_HIDDEN` — or replace with LSTM/Transformer/ResNet1D
- Training hyperparameters: `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `WEIGHT_DECAY`
- Class imbalance handling: add class weights to CrossEntropyLoss
- Preprocessing in `prepare.py`: `NORMALISE` strategy, data augmentation
  (e.g. jitter, time-shift), feature extraction (e.g. FFT features)

## What the Coding Agent must NOT change
`evaluate.py`, `wrapper.sh`, `orchestrator.py`, anything in `agents/` or `dashboard/`.

## Simplicity criterion
Prefer targeted changes. Do not add >100 lines in a single iteration.
One architectural or training change per iteration.

## Dataset
- Source: ECG5000 from UCR Time Series Archive
- 5000 samples (500 train + 4500 test split from original archive)
- 140 timesteps, 1 channel, 5 classes:
  - 0: Normal
  - 1: R-on-T Premature Ventricular Contraction
  - 2: Premature Ventricular Contraction
  - 3: Supra-ventricular Premature / Ectopic Beat
  - 4: Unclassified Beat
- Class imbalance: ~60% class 0

## Baseline
1D CNN with 3 conv blocks ([32, 64, 128] channels), kernel=5, MaxPool, Dropout=0.3,
FC(128), CosineAnnealingLR, 30 epochs.
Expected baseline macro-F1: ~0.72–0.80.

## Notes for the Research Agent
Key literature:
- Rajpurkar et al. 2017 (Stanford ECG): deep 1D ResNet achieves cardiologist-level
  performance — suggest ResNet-style skip connections if plateau is hit.
- Class imbalance: weighted CrossEntropyLoss or focal loss (Lin et al. 2017)
  consistently helps on imbalanced medical time-series.
- Temporal models: BiLSTM or temporal convolutional networks (TCN) often
  outperform plain CNNs on ECG by capturing long-range dependencies.

## Connection to the EEG experiment
This experiment is a dry run for the neuroscience pipeline:
- Time-series → time-series (same data structure as EEG channels)
- Class imbalance → mirrors ERP event sparsity in EEG
- 1D conv/transformer architecture → same building blocks as the EEG diffusion model
- Macro-F1 → similar multi-class evaluation logic to ERP component detection
