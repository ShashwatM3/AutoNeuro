# AutoNeuro — Experiment 1: Iris Classification

## Goal
Maximise **validation accuracy** on the Iris dataset (3-class flower classification).
This is the simplest possible experiment — its purpose is to validate that the agentic
loop, git tracking, dashboard, and HITL escalation all work correctly end-to-end.

## Metric
- **Name**: `accuracy`
- **Direction**: HIGHER IS BETTER
- **Range**: 0.0 – 1.0
- **Target**: ≥ 0.97

## What the Coding Agent may change
Only `train.py` and `prepare.py`. Specifically:
- Hyperparameters in `train.py`: `HIDDEN_LAYER_SIZES`, `ACTIVATION`, `SOLVER`,
  `LEARNING_RATE_INIT`, `MAX_ITER`, `ALPHA`
- Model class (e.g. switch from MLPClassifier to RandomForest, SVC, etc.)
- Preprocessing in `prepare.py` (e.g. add PCA, change scaler, adjust split ratio)

## What the Coding Agent must NOT change
`evaluate.py`, `wrapper.sh`, `orchestrator.py`, anything in `agents/` or `dashboard/`.

## Simplicity criterion
Prefer the simplest change that improves the metric. Do not introduce unnecessary
complexity. One logical change per iteration.

## Dataset
- Source: `sklearn.datasets.load_iris`
- 150 samples, 4 features, 3 classes (setosa / versicolor / virginica)
- 80/20 train/val split, stratified, StandardScaler normalisation

## Baseline
MLPClassifier with hidden layers (64, 32), ReLU, Adam, lr=0.001, 500 epochs.
Expected baseline accuracy: ~0.93–0.97.

## Notes for the Research Agent
No external papers needed for this experiment. It is intentionally trivial.
If the agent plateaus below 0.97 for more than 3 iterations, suggest switching
to a kernel SVM (SVC with RBF kernel) as it is known to achieve near-perfect
accuracy on Iris.
