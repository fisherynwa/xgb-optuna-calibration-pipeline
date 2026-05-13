# xgb-optuna-calibration-pipeline
![Tests](https://github.com/fisherynwa/xgb-optuna-calibration-pipeline/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange.svg)
![Optuna](https://img.shields.io/badge/Optuna-4.x-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

The xgb-opt framework automates model tuning and validation
and provides a `Scikit-learn`-like interface that handles nested evaluation and decision threshold optimization for **binary classification tasks**.

## 📚 Documentation & Examples
A detailed pipeline walkthrough is available in `notebooks/xgb_optuna_synthetic_data.ipynb`.

A real-world data analysis (using the Pima dataset) can be found in `notebooks/pima_analysis.ipynb`.

For implementation suggestions or bug reports, feel free to open an issue or contact me at vkvutov@gmail.com. 

## 🚀 Features
- XGBoost classifier with automated hyperparameter tuning via Optuna (the TPE sampler)
- Scikit-learn compatible API (`fit`, `predict`, `predict_proba`, `score`, `eval`)
- Penalized objective that rewards both high AUC and stability across folds
- Auto-detects environment -- utilizes all `n-1` cores locally, a single core in Docker
- Generalization estimates via nested cross-validation to prevent (from some) optimization bias
- Decision threshold selection by means of Youden's J statistic, the Euclidean method, or a user-defined threshold
- Probability calibration assessment using the Brier score
- Experiment tracking with MLflow (parameters, metrics, artifacts, and Optuna plots)
- (NEW) Freeze low-importance hyperparameters via `frozen_params` to reduce the hyperparameter space and improve optimization stability

## 📦 Installation

```bash
uv sync
uv run jupyter lab
```

**With Docker:**
```bash
docker-compose up --build
```
Then open :`http://127.0.0.1:8888` in your browser.

## 🛠 Usage
```python
from src.xgb_opt_clf import XGBOptClf
from src.helper_functions import nested_cv_score

# Initialize
clf = XGBOptClf(n_trials=20, nfold=5, std_penalty=0.5)

# Core interface
clf.fit(X_dev, y_dev)                                         # optimize + train
clf.predict(X_test, threshold=0.5)                            # binary predictions
clf.predict_proba(X_test)                                     # probabilities (n_samples, 2)
clf.score(X_test, y_test)                                     # ROC-AUC

# Evaluation
clf.eval(X_dev, y_dev, X_test, y_test, method="J_statistic")  # full evaluation report

# Inspection
clf.trials_dataframe()                                        # Optuna trials as DataFrame
clf.plot_optuna_insights()                                    # optimization visualizations
clf.param_importances()                                       # fANOVA hyperparameter importances

# Unbiased generalization estimate
nested_cv_score(clf, X, y, n_outer=5)                         # nested CV score
```

## ⚙️ Parameters
| Parameter | Default | Description |
|---|---|---|
| `n_trials` | 20 | Number of Optuna trials |
| `nfold` | 5 | Number of inner CV folds |
| `random_state` | 42 | Seed for reproducibility |
| `n_jobs` | `None` | Parallelism — auto-detects environment (1 in Docker, -1 locally). Override with any int |
| `std_penalty` | 0.5 | Penalty weight on CV std |
| `eval_metric` | `"auc"` | XGBoost eval metric (must be a gain metric) |
| `use_multivariate` | `False` | Whether TPE models hyperparameter interactions (worth using, i.e. `True`)|
| `is_stratified` | `True` | Whether inner CV folds are stratified |
| `scale_pos_weight` | `None` | Positive class weight — `None`, `"auto"`, or a float |
| `frozen_params` | `{}` | Hyperparameters fixed at specified values and removed from the Optuna search space. Useful when parameter importance analysis identifies low-impact parameters 
that might not need tuning |

## 🛠 Experiment Tracking
Runs are tracked with MLflow. To view the UI:
```bash
cd notebooks
mlflow ui --port 5002
```
Then open `http://127.0.0.1:5002` in your browser. Each run logs:
- Hyperparameters and number of estimators
- Validation AUC, CV AUC (mean/std), dev/test AUC, dev/test MCC, and applied threshold
- All nested CV metrics 
- ROC curve and applied threshold
- Optuna visualizations (optimization history, parameter importances, slice, parallel coordinate)
- Trained XGBoost model (and its interface)

## Tests
The package includes 30+ unit tests covering interface, predictions, evaluation, fitting, frozen_params, and reproducibility.

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