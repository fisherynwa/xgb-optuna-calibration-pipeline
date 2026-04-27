# xgb-optuna-calibration-pipeline
![Tests](https://github.com/fisherynwa/xgb-optuna-calibration-pipeline/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python package providing an XGBoost classifier with built-in Optuna hyperparameter optimization, wrapped in a scikit-learn-compatible interface. Experiment tracking is integrated via MLflow. Some detailed pipeline walkthrough can be found in `notebooks/xgb_optuna_synthetic_data.ipynb`. 

For implementation suggestions or bug reports, feel free to open an issue or contact me at vkvutov@gmail.com. Note that this package is under active development.

## Features
- XGBoost classifier with automatic hyperparameter tuning via Optuna
- Scikit-learn compatible API (`fit`, `predict`, `predict_proba`, `score`, `eval`)
- Penalized objective that rewards both high AUC and stability across folds
- Auto-detects environment -- utilizes all `n-1` cores locally, a single core in Docker
- Generalization estimates via nested cross-validation to prevent (from some) optimization bias
- Decision threshold selection via Youden's J statistic, the Euclidean method, or a user-defined threshold (additional metrics will be implemented)
- Probability calibration assessment via the Brier score
- Experiment tracking with MLflow (parameters, metrics, artifacts, and Optuna plots)

## Installation


```bash
uv sync
uv run jupyter lab
```

**With Docker:**
```bash
docker build -t xgb-optuna .
docker run -p 8888:8888 -p 5002:5002 xgb-optuna
```
Then open `http://127.0.0.1:8888` in your browser.

## Usage
```python
from src.xgb_opt_clf import XGBOptClf
from src.helper_functions import nested_cv_score

clf = XGBOptClf(n_trials=20, nfold=5, std_penalty=0.5)
clf.fit(X_dev, y_dev)                                        # optimize + train
clf.predict(X_test, threshold=0.5)                           # binary predictions
clf.predict_proba(X_test)                                    # probabilities (n_samples, 2)
clf.score(X_test, y_test)                                    # ROC-AUC
clf.eval(X_dev, y_dev, X_test, y_test, method="J_statistic") # full evaluation report
clf.trials_dataframe()                                       # Optuna trials as DataFrame
clf.plot_optuna_insights()                                   # optimization visualizations
nested_cv_score(clf, X, y, n_outer=5)                        # unbiased generalization estimate
```

## Parameters
| Parameter | Default | Description |
|---|---|---|
| `n_trials` | 20 | Number of Optuna trials |
| `nfold` | 5 | Number of inner CV folds |
| `random_state` | 42 | Seed for reproducibility |
| `n_jobs` | `None` | Parallelism — auto-detects environment (1 in Docker, -1 locally). Override with any int |
| `std_penalty` | 0.5 | Penalty weight on CV std |
| `eval_metric` | `"auc"` | XGBoost eval metric (must be a gain metric) |
| `use_multivariate` | `False` | Whether TPE models hyperparameter interactions |
| `is_stratified` | `True` | Whether inner CV folds are stratified |
| `scale_pos_weight` | `None` | Positive class weight — `None`, `"auto"`, or a float |

## Experiment Tracking
Runs are tracked with MLflow. To view the UI:
```bash
cd notebooks
mlflow ui --port 5002
```
Then open `http://127.0.0.1:5002` in your browser. Each run logs:
- Hyperparameters and number of estimators
- Validation AUC, CV AUC (mean/std), dev/test AUC, dev/test MCC, and applied threshold
- Nested CV AUC, Matthews Correlation Coefficients (MCCs) (mean/std), and Brier Scores per fold
- Thresholds, Matthews Correlation Coefficients (MCCs), and Brier scores across folds (CSV)
- ROC curve and applied threshold
- Optuna visualizations (optimization history, parameter importances, slice, parallel coordinate)
- Trained XGBoost model (and its interface)

## Tests
The package includes 30+ unit tests covering interface, predictions, evaluation, fitting, and reproducibility.

```bash
uv run pytest tests/ -v
```

## Project Structure
```
xgb-optuna/
├── .github/
│   └── workflows/
│       └── tests.yml        
├── src/
│   ├── xgb_opt_clf.py        # XGBoost + Optuna classifier
│   └── helper_functions.py   # nested_cv_score and shared utilities
│   └── xgb_opt_clf_warm.py   # warm-start for XGBoost + Optuna 
├── notebooks/
│   ├── xgb_optuna_synthetic_data.ipynb
│   └── xgb_optuna_pima_diabetes.ipynb
│   └── xgb_optuna_warm_start.ipynb
├── tests/
│   └── test_xgb_opt_clf.py  
│   └── test_xgb_opt_clf_warm.py
├── conftest.py
├── Dockerfile
├── pyproject.toml
├── main.py
└── README.md
```

# Notebooks
| Notebook | Description |
|---|---|
| `xgb_optuna_synthetic_data.ipynb` | End-to-end demo on synthetic data |
| `xgb_optuna_pima_diabetes.ipynb`  | Binary classification on the Pima Indians Diabetes dataset|
| `xgb_optuna_warm_start.ipynb`     | Toy example of the warm-start strategy using `XGBOptClfWarm` |