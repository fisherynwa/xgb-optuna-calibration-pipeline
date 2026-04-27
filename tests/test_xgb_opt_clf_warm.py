import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.xgb_opt_clf_warm import XGBOptClfWarm
import unittest

X, y = make_classification(n_samples=200, n_features=10, random_state=42)

def test_warm_start_enqueues_first_trial():
    """First trial params should be close to initial_params."""
    initial_params = {
        "lambda": 0.1, "alpha": 0.01, "learning_rate": 0.05,
        "max_depth": 4, "min_child_weight": 5, "gamma": 0.01,
        "subsample": 0.8, "colsample_bytree": 0.7,
    }
    clf = XGBOptClfWarm(n_trials=5, initial_params=initial_params)
    clf.fit(X, y)
    first_trial = clf.study_.trials[0].params

    for key, expected in initial_params.items():
        assert first_trial[key] == pytest.approx(expected, rel=1e-3), \
            f"{key}: expected {expected}, got {first_trial[key]}"

def test_no_initial_params_behaves_like_parent():
    """Without initial_params, should behave identically to XGBOptClf."""
    clf = XGBOptClfWarm(n_trials=5, initial_params=None)
    clf.fit(X, y)
    assert clf.best_params_ is not None
    assert clf.best_num_boost_round_ > 0

def test_predict_proba_shape():
    clf = XGBOptClfWarm(n_trials=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)

def test_predict_binary():
    clf = XGBOptClfWarm(n_trials=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})

def test_score_returns_auc():
    clf = XGBOptClfWarm(n_trials=5)
    clf.fit(X, y)
    auc = clf.score(X, y)
    assert 0.5 <= auc <= 1.0
    
    
if __name__ == "__main__":
    unittest.main()