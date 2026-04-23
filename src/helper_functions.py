import pandas as pd
import re
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from xgb_opt_clf import XGBOptClf
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import logging  
# -----------------------------------------------------------------------------
# Helper Functions 
# 'report_performance' is built to report our results in a pretty manner
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

def report_performance(obj):
    print(f"Threshold used:   {obj['applied_threshold']:.4f}")

    splits = ['dev_report', 'test_report']
    for split in splits:
        report = obj[split].copy()
        acc_value = report.pop('accuracy')
        df = pd.DataFrame(report).transpose().round(4)

        print(f"\n{'='*10} {split.upper()} {'='*10}")
        print(f"OVERALL ACCURACY: {acc_value:.4f}")
        print(f"AUC SCORE:        {obj[split.replace('_report', '_auc')]:.4f}")
        print("-" * 35)
        print(df)
  
  
  
def plot_roc_curve(clf, X_test, y_test, title="ROC curve"):
    """Plot ROC curve for a fitted XGBOptClf."""
    probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"XGBOptClf (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig

def nested_cv_score(clf, X, y, n_outer=5, stratified=True):
    """Return estimates via nested cross-validation.

    Parameters
    ----------
    clf : XGBOptClf
        A configured but unfitted instance. Its hyperparameters are
        cloned for each outer fold via get_params().
    n_outer : int
        Number of outer folds. Default 5.
    stratified : bool
        If True, uses StratifiedKFold to preserve class balance across
        folds. Recommended for imbalanced datasets. Default True.

    Returns
    -------
    dict with keys:
        - scores             : np.ndarray of shape (n_outer,) — AUC per outer fold
        - best_params        : list of dicts, one per outer fold
        - n_trees            : list of ints, one per outer fold
        - optimal_thresholds : np.ndarray of shape (n_outer,) — Youden's J threshold per outer fold (computed on train fold)
        - mccs               : np.ndarray of shape (n_outer,) — MCC per outer fold

    Notes
    -----
    The `stratified` parameter here controls the outer CV splits only.
    The inner CV stratification is controlled by `clf.is_stratified`.
    Both default to True — ensure they are set consistently.
    Computational cost scales as n_outer x n_trials x n_inner_folds x boost_rounds.
    Consider reducing n_trials on clf for large datasets.
    """
    if stratified:
        outer_cv = StratifiedKFold(
            n_splits=n_outer, shuffle=True, random_state=clf.random_state
        )
    else:
        outer_cv = KFold(
            n_splits=n_outer, shuffle=True, random_state=clf.random_state
        )

    scores, tmp_best_params, n_trees, optimal_thresholds, mccs = [], [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info(f"Nested CV: outer fold {fold_idx + 1}/{n_outer}")

        fold_clf = XGBOptClf(**clf.get_params())
        fold_clf.fit(X[train_idx], y[train_idx])

        # eval() computes threshold on train fold, applies to test fold
        results = fold_clf.eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

        fold_score        = results["test_auc"]
        optimal_threshold = results["applied_threshold"]
        fold_mcc          = results["test_mcc"]

        scores.append(fold_score)
        optimal_thresholds.append(optimal_threshold)
        mccs.append(fold_mcc)
        tmp_best_params.append(fold_clf.best_params_)
        n_trees.append(fold_clf.best_num_boost_round_)

        logger.info(
            f"Nested CV: fold {fold_idx + 1} — "
            f"AUC = {fold_score:.4f}, "
            f"MCC = {fold_mcc:.4f}, "
            f"threshold = {optimal_threshold:.4f}"
        )

    scores             = np.array(scores)
    optimal_thresholds = np.array(optimal_thresholds)
    mccs               = np.array(mccs)

    logger.info(
        f"Nested CV complete — "
        f"mean AUC: {scores.mean():.4f} ± {scores.std():.4f} | "
        f"mean MCC: {mccs.mean():.4f} ± {mccs.std():.4f} | "
        f"mean threshold: {optimal_thresholds.mean():.4f} ± {optimal_thresholds.std():.4f}"
    )

    return {
        "scores":             scores,
        "best_params":        tmp_best_params,
        "n_trees":            n_trees,
        "optimal_thresholds": optimal_thresholds,
        "mccs":               mccs,
    }