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

logger = logging.getLogger(__name__)
 
class XGBOptClf(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with Optuna hyperparameter optimization.
 
    Implements sklearn-like interface:
        - fit(X_dev, y_dev)
        - predict(X)
        - predict_proba(X)
    Also supports:
        - score(X, y, sample_weight) — returns ROC-AUC with optional sample weights
        - eval(X_dev, y_dev, X_test, y_test, threshold=None) — full evaluation report
        - Optuna visualizations via plot_optuna_insights()
        - unbiased nested CV evaluation via nested_cv_score()
 
    Inherits from BaseEstimator and ClassifierMixin, which provides:
        - get_params() / set_params() for free (used by clone(), Pipeline, etc.)
 
    The Optuna study minimizes the penalized objective:
        mean_metric - std_penalty * std_metric
 
    This rewards both high AUC and stability (low std) across folds. The eval_metric
    should therefore be a gain metric (the higher is for the better), e.g. "auc".
    The default is "auc".
 
    The TPE sampler is used with this adaptive configuration:
        - n_startup_trials = max(10, n_trials // 4)
 
    Parameters
    ----------
    n_trials : int
        Number of Optuna trials. Default 20.
    nfold : int
        Number of inner CV folds used in each Optuna trial. Default 5.
    random_state : int
        Seed for reproducibility. Default 42.
    n_jobs : int or None
        Parallelism for XGBoost. None = auto-detect (1 in Docker, -1 locally).
        Parallelism is not used for Optuna.
    std_penalty : float
        Weight applied to CV std in the objective:
            mean_metric - std_penalty * std_metric
        Higher values reward stability over raw performance. Default 0.5.
    eval_metric : str
        XGBoost eval metric used during CV. Must be a gain metric
        (higher is better). Default "auc".
    use_multivariate : bool
        Whether to use the TPE multivariate option, which models interactions
        between hyperparameters.
    is_stratified : bool
        Whether to use stratified folds in the inner CV. Controls inner CV only.
        For outer CV stratification, see the `stratified` parameter of
        nested_cv_score(). Default True.
    scale_pos_weight : float, "auto", or None
        Weight for positive class. "auto" computes neg/pos ratio from
        training data. Default None.
    """
    def __init__(
        self,
        n_trials=20,
        nfold=5,
        random_state=42,
        n_jobs=None,
        std_penalty=0.5,
        eval_metric="auc",
        use_multivariate=False,
        is_stratified=True,
        scale_pos_weight=None
    ):
        self.n_trials = n_trials
        self.nfold = nfold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.std_penalty = std_penalty
        self.eval_metric = eval_metric
        self.use_multivariate = use_multivariate
        self.is_stratified = is_stratified
        self.scale_pos_weight = scale_pos_weight
 
    # ------------------------
    # Internal helpers
    # ------------------------
    
    def _check_is_fitted(self):
        # Uses sklearn's check_is_fitted: looks for attributes ending in '_'
        check_is_fitted(self)
    
    def _resolve_n_jobs(self):
        if self.n_jobs is None:
            return 1 if os.path.exists("/.dockerenv") else -1
        return self.n_jobs

    def _base_params(self):
        """Return fixed params shared by CV and final training."""
        params = {         
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": self.eval_metric,
            "booster": "gbtree",
            "tree_method": "hist",
            "n_jobs": self._resolve_n_jobs(),
            "seed": self.random_state,
        }
        if hasattr(self, "scale_pos_weight_"):
            spw = self.scale_pos_weight_
        else:
            spw = self.scale_pos_weight
        if spw is not None:
            params["scale_pos_weight"] = spw
        return params
    
    def _compute_scale_pos_weight(self, y):
        """Compute scale_pos_weight as ratio of negatives to positives."""
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        if n_pos == 0:
            raise ValueError("No positive samples found in y.")
        return n_neg / n_pos
    # ------------------------
    # Optuna Objective
    # ------------------------
    def _objective(self, trial, dtrain: xgb.DMatrix):
        """
        Objective function for Optuna hyperparameter search by means of CV
        This implementation is inspired by this repo:
        
        https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv_integration.py
        
        """
        params = {
            **self._base_params(),
            # Hyperparameter search space
            "lambda": trial.suggest_float("lambda", 1e-2, 12.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 15.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 12),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        }

        # Callback class that integrates the Optuna pruning system into the XGBoost-based training loop
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"test-{self.eval_metric}")
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            early_stopping_rounds=50,
            nfold=self.nfold,
            seed=self.random_state,
            callbacks=[pruning_callback],
            stratified=self.is_stratified,
        )

        # Set n_estimators as a trial attribute; len(cv_results) — this identifies the best iteration in terms of its performance (unlike catboost)
        trial.set_user_attr("n_estimators", len(cv_results))
        
        mean_auc = cv_results[f"test-{self.eval_metric}-mean"].iloc[-1] # type: ignore
        std_auc = cv_results[f"test-{self.eval_metric}-std"].iloc[-1] # type: ignore

        trial.set_user_attr(f"test_{self.eval_metric}_mean", mean_auc)
        trial.set_user_attr(f"test_{self.eval_metric}_std", std_auc)

        # Return the penalised AUC to reward both performance and stability
        return mean_auc - (std_auc * self.std_penalty)

    # ------------------------
    # Fit + Hyperparameter Optimization
    # ------------------------
    def fit(self, X_dev, y_dev):
        """
        Optimize hyperparameters using CV and train final model.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info(f"Starting Optuna optimization for {self.n_trials} trials")
        
        if self.scale_pos_weight == "auto":
            spw = self._compute_scale_pos_weight(y_dev)
            logger.info(f"scale_pos_weight set automatically to {spw:.2f}")
            self.scale_pos_weight_ = spw  # store computed value
        else:
            self.scale_pos_weight_ = self.scale_pos_weight

        dtrain = xgb.DMatrix(X_dev, label=y_dev)
        
        # By default, the TPE starts with 10 random trials to explore the space, then learns from them to make informed suggestions.
        # If n_trials is high, we can afford more random exploration before the TPE kicks in, so I set n_startup_trials to n_trials // 4.
        n_startup_trials = round(max(10, self.n_trials // 4))
        
        sampler = optuna.samplers.TPESampler(multivariate=self.use_multivariate,
                                             n_startup_trials=n_startup_trials, # 10 is the default value, but I set it adaptively based on n_trials
                                             seed=self.random_state)
        # Stop trials performing worse than the median after 5 boosting rounds
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) 

        self.study_ = optuna.create_study(direction="maximize",
                                          sampler=sampler,
                                          pruner=pruner)

        self.study_.optimize(lambda trial: self._objective(trial, dtrain),
                            n_trials=self.n_trials,
                            )

        self.best_params_ = self.study_.best_params

        self.best_num_boost_round_ = self.study_.best_trial.user_attrs["n_estimators"]

        # Store cv_results_ from its best trial's user attrs - here users can check out some interim results
        self.cv_results_ = {
            f"test_{self.eval_metric}_mean": self.study_.best_trial.user_attrs[f"test_{self.eval_metric}_mean"],
            f"test_{self.eval_metric}_std":  self.study_.best_trial.user_attrs[f"test_{self.eval_metric}_std"],
            "n_estimators":  self.best_num_boost_round_,
            }

        # Train the final model
        self.final_params_ = {
            **self._base_params(),
            **self.best_params_,
            }
    
        self.best_model_ = xgb.train(self.final_params_, 
                                    dtrain, 
                                    num_boost_round=self.best_num_boost_round_)
        
        return self
    
    # ------------------------
    # Prediction
    # ------------------------

    def predict_proba(self, X):
        """Return predicted probabilities with shape (n_samples, 2).

        Uses the native XGBoost booster API (xgb.Booster), where .predict()
        returns probabilities directly when objective='binary:logistic'.
        This differs from the sklearn API (XGBClassifier), where .predict()
        returns class labels instead.
        
        Compatible with sklearn pipelines and cross_val_score.
        https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict
        """
        self._check_is_fitted()
        prob_pos = self.best_model_.predict(xgb.DMatrix(X))
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X, threshold=0.5):
        """Return binary predictions; i.e., Y in {0, 1}.
        Parameters
        ----------
        X : array-like
        threshold : float, optional
            Decision threshold applied to P(Y=1|X = x). Default 0.5.
        """
        self._check_is_fitted()
        probs = self.predict_proba(X)[:, 1] # the second coordinate corresponds to Pr(Y_i = 1 | X = x)
        return (probs >= threshold).astype(int)

    # ------------------------
    # Scoring
    # ------------------------
    def score(self, X, y, sample_weights = None): # type: ignore
        """Return ROC-AUC score."""
        self._check_is_fitted()
        # Sampe weights; 1\sum_i wi \Sum 1(\hat{y_i} = y_i) \hat{w_i}
        return roc_auc_score(y, self.predict_proba(X)[:, 1], sample_weight=sample_weights) # Sample weights; ()
    # ------------------------
    # Evaluation
    # ------------------------
    def eval(self, X_dev, y_dev, X_test, y_test, method="J_statistic", threshold=None):
        """Evaluate on dev and test sets.

        Parameters
        ----------
        X_dev, y_dev : array-like
            Dev data and labels.
        X_test, y_test : array-like
            Test data and labels.
        method : {"J_statistic", "Euclidean"} or None: for more details, https://link.springer.com/article/10.1186/s12874-024-02198-2#Sec23
            Method to compute the decision threshold from dev data.
            Ignored if threshold is provided. Default "J_statistic".
        threshold : float or None
            Fixed decision threshold. If provided, overrides method.
            Default None.

        Returns
        -------
        dict with keys: dev_auc, test_auc, dev_mcc, test_mcc, matthews_corr_dev,
                        applied_threshold, dev_report, test_report.
        """
        self._check_is_fitted()

        # Get raw probabilities
        dev_probs  = self.predict_proba(X_dev)[:, 1]
        test_probs = self.predict_proba(X_test)[:, 1]

        # Threshold determination
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"threshold must be a float:  {threshold!r}")
            final_threshold = threshold
        elif method == "J_statistic":
            fpr, tpr, thresholds = roc_curve(y_dev, dev_probs)
            final_threshold = thresholds[np.argmax(tpr - fpr)]
        elif method == "Euclidean":
            fpr, tpr, thresholds = roc_curve(y_dev, dev_probs)
            final_threshold = thresholds[np.argmin(fpr**2 + (1 - tpr)**2)]
        else:
            raise ValueError(f"Unrecognized method: {method!r}")

        # Apply threshold
        dev_preds  = (dev_probs  >= final_threshold).astype(int)
        test_preds = (test_probs >= final_threshold).astype(int)

        return {
            "dev_auc":           roc_auc_score(y_dev,  dev_probs),
            "test_auc":          roc_auc_score(y_test, test_probs),
            "dev_mcc":           matthews_corrcoef(y_dev,  dev_preds),
            "test_mcc":          matthews_corrcoef(y_test, test_preds),
            "applied_threshold": final_threshold,
            "dev_report":        classification_report(y_dev,  dev_preds, output_dict=True),
            "test_report":       classification_report(y_test, test_preds, output_dict=True),
        }
    # -------------------------------------------------------
    # Access Optuna Trials - Users can go over each trial
    # -------------------------------------------------------
    def trials_dataframe(self):
        """Return Optuna trials as a DataFrame."""
        self._check_is_fitted()
        return self.study_.trials_dataframe()
    # -------------------------------------------------------
    # Visualization
    # -------------------------------------------------------
    def plot_optuna_insights(self):
        """Return Optuna visualisation figures."""
        self._check_is_fitted()
        self._check_is_fitted()
        fig1 = vis.plot_optimization_history(self.study_)
        fig2 = vis.plot_param_importances(self.study_)
        fig3 = vis.plot_slice(self.study_)
        fig4 = vis.plot_parallel_coordinate(self.study_)
        return fig1, fig2, fig3, fig4
        
    def plot_calibration(self, X_test, y_test, n_bins=10, strategy="uniform"):
        """Plot calibration curve (reliability diagram).

        Parameters: More details can be found here: 
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve
        ----------
        X_test, y_test : test data and labels
        n_bins : int
            Number of bins for calibration curve. Default 10.
        strategy : str
            Strategy for binning — "uniform" or "quantile". Default "uniform".
            - "uniform" : bins have equal width in [0, 1]
            - "quantile" : bins have equal number of samples
        """
        self._check_is_fitted()

        probs = self.predict_proba(X_test)[:, 1]
        # y_test is the true y
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=n_bins, strategy=strategy) # type: ignore

        fig, axes = plt.subplots(1, 2)

        # Left: calibration curve
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.plot(prob_pred, prob_true, "s-", label="Model")
        ax1.set_xlabel("Predicted probability (Class: 1)")
        ax1.set_ylabel("Fraction of positives (Class: 1)")
        ax1.set_title("Calibration curve")
        ax1.legend()

        # Right: probability distribution
        ax2 = axes[1]
        ax2.hist(probs[y_test == 0], label="Negative", color="steelblue")
        ax2.hist(probs[y_test == 1], label="Positive", color="coral")
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Probability distribution by class")
        ax2.legend()

        plt.tight_layout()
        
        return fig