import logging
import numpy as np
import optuna
import optuna.visualization as vis
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.validation import check_is_fitted 
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.xgb_opt_clf import XGBOptClf

class XGBOptClfWarm(XGBOptClf):
    """XGBOptClf with warm-start support via Optuna trial enqueuing.

        Extends XGBOptClf by injecting a plausible hyperparameter region as the first Optuna trial;
        navigating the TPE toward a promising area of the search space from the start.

        This class is inspired by the common practice of "warm-starting" hyperparameter optimization with results from a previous experiment,
        and leverages Optuna's `enqueue_trial()` method to achieve this.
        
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html
        
        Parameters
        ----------
        initial_params : dict or None
            Hyperparameter configuration to enqueue the first trial.
            Must contain the same keys as the Optuna search space.
            Defaults to None, which behaves identically to XGBOptClf.

        Warning
        -------
        If `initial_params` are derived from the same dataset employed for fitting
        (i.e. extracted from a previous nested_cv_score() run on the same X, y),
        this introduces data leakage — the model **will** overfit to the dev set
        and the dev-test gap will increase. To utilize warm-starting safely:

            i. Retain a held-out test set before running nested_cv_score()
            ii. Extract best_params from the nested CV on dev solely for the purpose of warm-starting
            iii. Warm-start on dev, evaluate on the held-out test

        Warm-starting is most beneficial when transferring hyperparameters
        from a previous experiment on a similar dataset.

        You can find a toy example of how to utilize this class in the `xgb_optuna_warm_up.ipynb` notebook.

        """

    def __init__(self, *args, initial_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_params = initial_params

    def fit(self, X_dev, y_dev):
        # Identical to parent fit() but enqueues initial_params before optimising
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.scale_pos_weight == "auto":
            spw = self._compute_scale_pos_weight(y_dev)
            self.scale_pos_weight_ = spw
        else:
            self.scale_pos_weight_ = self.scale_pos_weight

        dtrain = xgb.DMatrix(X_dev, label=y_dev)
        n_startup_trials = round(max(10, self.n_trials // 4))

        sampler = optuna.samplers.TPESampler(
            multivariate=self.use_multivariate,
            n_startup_trials=n_startup_trials,
            seed=self.random_state
        )
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        self.study_ = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        if self.initial_params is not None:
            self.study_.enqueue_trial(self.initial_params)

        self.study_.optimize(
            lambda trial: self._objective(trial, dtrain),
            n_trials=self.n_trials,
        )

        self.best_params_ = self.study_.best_params
        self.best_num_boost_round_ = self.study_.best_trial.user_attrs["n_estimators"]
        self.cv_results_ = {
            f"test_{self.eval_metric}_mean": self.study_.best_trial.user_attrs[f"test_{self.eval_metric}_mean"],
            f"test_{self.eval_metric}_std":  self.study_.best_trial.user_attrs[f"test_{self.eval_metric}_std"],
            "n_estimators": self.best_num_boost_round_,
        }
        self.final_params_ = {**self._base_params(), **self.best_params_}
        self.best_model_ = xgb.train(
            self.final_params_,
            dtrain,
            num_boost_round=self.best_num_boost_round_
        )
        return self