# AutoNeuro — Experiment 2: Diabetes Progression Regression

## Goal
Maximise **R² score** predicting disease progression in diabetes patients.
This is a regression task — harder than Iris because the target is continuous,
the relationship is noisy, and the agent must reason about regularisation and
feature engineering trade-offs.

## Metric
- **Name**: `R²` (coefficient of determination)
- **Direction**: HIGHER IS BETTER
- **Range**: –∞ to 1.0 (1.0 = perfect, 0.0 = predicts mean, negative = worse than mean)
- **Target**: ≥ 0.55

## What the Coding Agent may change
Only `train.py` and `prepare.py`. Specifically:
- Hyperparameters in `train.py`: `HIDDEN_LAYER_SIZES`, `ACTIVATION`, `SOLVER`,
  `LEARNING_RATE_INIT`, `MAX_ITER`, `ALPHA`, `EARLY_STOPPING`, `N_ITER_NO_CHANGE`
- Model class (e.g. switch to GradientBoostingRegressor, Ridge, ElasticNet, etc.)
- Feature engineering in `prepare.py`: set `POLYNOMIAL_FEATURES=True`,
  change `VAL_SPLIT`, add feature selection

## What the Coding Agent must NOT change
`evaluate.py`, `wrapper.sh`, `orchestrator.py`, anything in `agents/` or `dashboard/`.

## Simplicity criterion
Prefer the simplest change that improves R². Avoid overfitting — watch the
gap between train and val scores. One logical change per iteration.

## Dataset
- Source: `sklearn.datasets.load_diabetes`
- 442 patients, 10 normalised clinical features
- Continuous target: disease progression measure one year after baseline
- 80/20 random train/val split, StandardScaler normalisation

## Baseline
MLPRegressor with hidden layers (128, 64, 32), ReLU, Adam, lr=0.001, early stopping.
Expected baseline R²: ~0.45–0.52.

## Notes for the Research Agent
Gradient Boosted Trees (GradientBoostingRegressor or XGBRegressor) typically
outperform MLPs on small tabular datasets like this due to their inductive bias
toward piecewise-constant functions. Ridge regression with polynomial features
(degree 2) is also a strong baseline. Suggest these if MLP plateaus.
