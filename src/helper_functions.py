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

def nested_cv_score(clf, X, y, n_outer=5, stratified=True, preprocessor=None, tau=None):
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
    preprocessor : callable or None
        Optional preprocessing function applied inside each outer fold
        before fitting. Must have the signature:

            X_dev, X_test = preprocessor(X_dev, y_dev, X_test)

        The function is fit on X_dev only and applied to X_test,
        ensuring no leakage across folds. Default None.
    tau: float or None
        Optional fixed threshold for classification. If None, the optimal
        threshold is determined on the dev fold using the default method in XGBOptClf.eval(). Default None.

    Notes
    -----
    The `stratified` parameter here controls the outer CV splits only.
    The inner CV stratification is controlled by `clf.is_stratified`.
    Both default to True — ensure they are set consistently.
    """
    if stratified:
        outer_cv = StratifiedKFold(
            n_splits=n_outer, shuffle=True, random_state=clf.random_state
        )
    else:
        outer_cv = KFold(
            n_splits=n_outer, shuffle=True, random_state=clf.random_state
        )

    test_aucs, best_params, n_trees, optimal_thresholds = [], [], [], []
    test_mccs, dev_mccs, dev_aucs, reports = [], [], [], []
    test_briers, dev_briers = [], []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info(f"Nested CV: outer fold {fold_idx + 1}/{n_outer}")

        X_dev,  y_dev  = X[dev_idx],  y[dev_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if preprocessor is not None:
            X_dev, X_test = preprocessor(X_dev, y_dev, X_test)

        fold_clf = XGBOptClf(**clf.get_params())
        fold_clf.fit(X_dev, y_dev, )

        # eval() computes threshold on dev fold, applies to test fold
        results = fold_clf.eval(X_dev, y_dev, X_test, y_test, threshold=tau)

        fold_test_auc   = results["test_auc"]
        optimal_threshold  = results["applied_threshold"]
        fold_test_mcc      = results["test_mcc"]
        fold_dev_auc       = results["dev_auc"]
        fold_dev_mcc       = results["dev_mcc"]
        fold_test_brier    = results["test_brier"]
        fold_dev_brier     = results["dev_brier"]


        test_aucs.append(fold_test_auc)
        optimal_thresholds.append(optimal_threshold)
        test_mccs.append(fold_test_mcc)
        dev_aucs.append(fold_dev_auc)
        dev_mccs.append(fold_dev_mcc)
        test_briers.append(fold_test_brier)
        dev_briers.append(fold_dev_brier)
        best_params.append(fold_clf.best_params_)
        n_trees.append(fold_clf.best_num_boost_round_)
        reports.append({
            "dev_report":  results["dev_report"],
            "test_report": results["test_report"],
        })

        logger.info(
            f"Nested CV: fold {fold_idx + 1} — "
            f"test AUC = {fold_test_auc:.4f}, "
            f"dev AUC = {fold_dev_auc:.4f}, "
            f"MCC = {fold_test_mcc:.4f}, "
            f"Brier = {fold_test_brier:.4f}, "
            f"threshold = {optimal_threshold:.4f}"
        )

    test_aucs      = np.array(test_aucs)
    optimal_thresholds = np.array(optimal_thresholds)
    test_mccs          = np.array(test_mccs)
    dev_aucs           = np.array(dev_aucs)
    dev_mccs           = np.array(dev_mccs)
    test_briers        = np.array(test_briers)
    dev_briers         = np.array(dev_briers)


    logger.info(
        f"Nested CV complete — "
        f"mean test AUC: {test_aucs.mean():.4f} ± {test_aucs.std():.4f} | "
        f"mean dev AUC: {dev_aucs.mean():.4f} ± {dev_aucs.std():.4f} | "
        f"mean test MCC: {test_mccs.mean():.4f} ± {test_mccs.std():.4f} | "
        f"mean dev MCC: {dev_mccs.mean():.4f} ± {dev_mccs.std():.4f} | "
        f"mean test Brier: {test_briers.mean():.4f} ± {test_briers.std():.4f} | "
        f"mean dev Brier: {dev_briers.mean():.4f} ± {dev_briers.std():.4f} | "
        f"mean threshold: {optimal_thresholds.mean():.4f} ± {optimal_thresholds.std():.4f}"
    )

    return {
        "scores":             test_aucs,
        "dev_aucs":           dev_aucs,
        "best_params":        best_params,
        "n_trees":            n_trees,
        "optimal_thresholds": optimal_thresholds,
        "test_mccs":          test_mccs,
        "dev_mccs":           dev_mccs,
        "test_briers":        test_briers,
        "dev_briers":         dev_briers,
        "reports":            reports,
    }