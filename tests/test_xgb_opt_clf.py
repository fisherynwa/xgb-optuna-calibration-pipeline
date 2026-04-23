import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgb_opt_clf import XGBOptClf
import os

class TestXGBOptClf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate synthetic data once for all tests."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            random_state=42
        )
        cls.X_dev, cls.X_test, cls.y_dev, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
##############
# Test interface
#############
    def test_get_params(self):
        """get_params() returns all expected keys with correct defaults."""
        clf = XGBOptClf()
        params = clf.get_params()

        self.assertIn("n_trials", params)
        self.assertIn("nfold", params)
        self.assertIn("random_state", params)
        self.assertIn("n_jobs", params)
        self.assertIn("std_penalty", params)
        self.assertIn("eval_metric", params)
        self.assertIn("use_multivariate", params)
        self.assertIn("scale_pos_weight", params)
        
        self.assertEqual(params["n_trials"], 20)
        self.assertEqual(params["nfold"], 5)
        self.assertEqual(params["random_state"], 42)
        self.assertIsNone(params["n_jobs"]) 
        self.assertEqual(params["std_penalty"], 0.5)
        self.assertEqual(params["eval_metric"], "auc")
        self.assertEqual(params["use_multivariate"], False)
        self.assertIsNone(params["scale_pos_weight"])

    def test_check_is_fitted_raises_before_fit(self):
        """check_is_fitted raises NotFittedError before fit() is called."""
        from sklearn.exceptions import NotFittedError
        clf = XGBOptClf()
        with self.assertRaises(NotFittedError):
            clf.predict(self.X_test)
    
    def test_resolve_n_jobs_default(self):
        """_resolve_n_jobs() returns -1 locally and 1 in Docker."""
        clf = XGBOptClf()
        resolved = clf._resolve_n_jobs()
        
        if os.path.exists("/.dockerenv"):
            self.assertEqual(resolved, 1)   # inside Docker
        else:
            self.assertEqual(resolved, -1)  # local machine

    def test_resolve_n_jobs_manual(self):
        """_resolve_n_jobs() returns user-specified value when set explicitly."""
        clf = XGBOptClf(n_jobs=4)
        self.assertEqual(clf._resolve_n_jobs(), 4)
    
    
    def test_get_params_custom(self):
        """get_params() reflects custom values passed to __init__."""
        clf = XGBOptClf(n_trials=10, nfold=3, std_penalty=0.1)
        params = clf.get_params()

        self.assertEqual(params["n_trials"], 10)
        self.assertEqual(params["nfold"], 3)
        self.assertEqual(params["std_penalty"], 0.1)

    ### TESTS: predict()
    def test_predict_binary_output(self):
        """predict() returns only 0s and 1s."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        preds = clf.predict(self.X_test)
        self.assertTrue(set(preds).issubset({0, 1}))
        
    def test_predict_custom_threshold(self):
        """predict() with custom threshold produces different results than default 0.5."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        preds_default = clf.predict(self.X_test)              # threshold=0.5
        preds_low     = clf.predict(self.X_test, threshold=0.1)  # lower threshold
        # lower threshold should predict more positives
        self.assertGreater(preds_low.sum(), preds_default.sum())
        
    def test_predict_threshold_zero_all_positive(self):
        """predict() threshold=0.0 classifies everything as positive."""   
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        preds = clf.predict(self.X_test, threshold=0.0)
        self.assertTrue(np.all(preds == 1))             

    def test_predict_proba_in_range(self):
        """predict_proba() values are in [0, 1]."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        proba = clf.predict_proba(self.X_test)
        self.assertTrue((proba >= 0).all() and (proba <= 1).all())
        
    def test_predict_proba_shape(self):
        """predict_proba() its shape"""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 2))

    def test_predict_proba_sums_to_one(self):
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        proba = clf.predict_proba(self.X_test)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
        
    ####################################### 
    ### TESTS: eval()
    #######################################
    def test_eval_keys(self):
        """eval() returns dict with expected keys."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        results = clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test)
        self.assertIn("dev_auc", results)
        self.assertIn("test_auc", results)
        self.assertIn("dev_report", results)
        self.assertIn("test_report", results)
        
    def test_eval_mcc_in_valid_range(self):
        """MCC values are within valid range [-1, 1]."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        results = clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test)
        self.assertGreaterEqual(results["dev_mcc"],  -1.0)
        self.assertLessEqual(results["dev_mcc"],      1.0)
        self.assertGreaterEqual(results["test_mcc"], -1.0)
        self.assertLessEqual(results["test_mcc"],     1.0)
    
    def test_eval_fixed_threshold(self):
        """eval() with fixed threshold uses that exact threshold."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        results = clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test, threshold = 0.5)
        self.assertAlmostEqual(results["applied_threshold"], 0.5, places=5)
    
    def test_eval_invalid_threshold_raises(self):
        """eval() raises ValueError when threshold is not a float."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        with self.assertRaises(ValueError):
            clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test, threshold="bad")

    def test_eval_invalid_method_raises(self):
        """eval() raises ValueError when method is unrecognized."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        with self.assertRaises(ValueError):
            clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test, method="bad_method")

    def test_eval_threshold_computed_from_dev(self):
        """Threshold is computed from dev data — changing test data does not affect it."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        r1 = clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test)
        # use random test data — threshold should stay the same
        X_test_alt = np.random.default_rng(0).random(self.X_test.shape)
        y_test_alt = np.random.default_rng(0).integers(0, 2, len(self.y_test))
        r2 = clf.eval(self.X_dev, self.y_dev, X_test_alt, y_test_alt)
        self.assertAlmostEqual(r1["applied_threshold"], r2["applied_threshold"], places=5)

    def test_eval_auc_in_valid_range(self):
        """AUC scores are within valid range [0, 1]."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        results = clf.eval(self.X_dev, self.y_dev, self.X_test, self.y_test)
        self.assertGreaterEqual(results["dev_auc"],  0.0)
        self.assertLessEqual(results["dev_auc"],     1.0)
        self.assertGreaterEqual(results["test_auc"], 0.0)
        self.assertLessEqual(results["test_auc"],    1.0)
    
    #######################################
    ### TEST: score()
    #######################################   
    
    def test_score_in_valid_range(self):
        """test the plausibility of score()"""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        score_auc = float(clf.score(self.X_test, self.y_test))
        self.assertGreaterEqual(score_auc, 0.0) 
        self.assertLessEqual(score_auc,    1.0)
        
    #######################################
    ### TESTS: scale_pos_weight()
    #######################################   
    def test_scale_pos_weight_auto(self):
        """scale_pos_weight='auto' computes correct neg/pos ratio."""
        clf = XGBOptClf(n_trials=5, random_state=42, scale_pos_weight="auto")
        clf.fit(self.X_dev, self.y_dev)
        expected = np.sum(self.y_dev == 0) / np.sum(self.y_dev == 1)
        self.assertAlmostEqual(clf.scale_pos_weight_, expected, places=5)

    def test_scale_pos_weight_manual(self):
        """scale_pos_weight set manually is passed through correctly."""
        clf = XGBOptClf(n_trials=5, random_state=42, scale_pos_weight=5.0)
        clf.fit(self.X_dev, self.y_dev)
        self.assertEqual(clf.scale_pos_weight_, 5.0)

    def test_scale_pos_weight_none(self):
        """scale_pos_weight=None does not inject scale_pos_weight into params."""
        clf = XGBOptClf(n_trials=5, random_state=42, scale_pos_weight=None)
        clf.fit(self.X_dev, self.y_dev)
        self.assertNotIn("scale_pos_weight", clf.final_params_)   
    
     
    #######################################
    ## TESTS: fit()
    #######################################
    
    def test_fit_cv_results_keys(self):
        """fit() stores cv_results_ with expected keys."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        self.assertIn(f"test_{clf.eval_metric}_mean", clf.cv_results_)
        self.assertIn(f"test_{clf.eval_metric}_std",  clf.cv_results_)
        self.assertIn("n_estimators",                 clf.cv_results_)

    def test_fit_stores_best_params(self):
        """fit() stores best_params_ with expected hyperparameter keys."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        expected_keys = {
            "lambda",
            "alpha",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "gamma",
            "subsample",
            "colsample_bytree",
        }
        self.assertEqual(expected_keys, set(clf.best_params_.keys()))
     
    def test_fit_best_params_plausibility(self):
        """best_params_ contains finite, sensible values after fit."""
        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)
        p = clf.best_params_

        # all values are finite numbers, not NaN or inf
        for key, value in p.items():
            self.assertTrue(np.isfinite(value), msg=f"{key} is not finite: {value}")

        # n_estimators is a positive integer
        self.assertGreater(clf.best_num_boost_round_, 0)

        # learning rate is not degenerate
        self.assertGreater(p["learning_rate"], 0)
            
        
    #######################################
    ## TESTS: reproducibility
    #######################################
    def test_reproducibility(self):
        """Two classifiers with the same random_state produce identical results."""
        clf1 = XGBOptClf(n_trials=5, random_state=42)
        clf2 = XGBOptClf(n_trials=5, random_state=42)

        clf1.fit(self.X_dev, self.y_dev)
        clf2.fit(self.X_dev, self.y_dev)

        self.assertEqual(clf1.best_params_, clf2.best_params_)
        self.assertEqual(clf1.best_num_boost_round_, clf2.best_num_boost_round_)

        np.testing.assert_array_almost_equal(
            clf1.predict_proba(self.X_test),
            clf2.predict_proba(self.X_test)
        )

        self.assertAlmostEqual(
            clf1.eval(X_dev = self.X_dev, 
                      y_dev=self.y_dev,
                      X_test=self.X_test,
                      y_test=self.y_test)["test_auc"], 
            clf2.eval(X_dev = self.X_dev, 
                      y_dev=self.y_dev, 
                      X_test=self.X_test, 
                      y_test=self.y_test)["test_auc"], places=5)

    def test_different_random_state(self):
        """Two classifiers with different random_state may produce different results."""
        clf1 = XGBOptClf(n_trials=10, random_state=42)
        clf2 = XGBOptClf(n_trials=10, random_state=99)

        clf1.fit(self.X_dev, self.y_dev)
        clf2.fit(self.X_dev, self.y_dev)

        proba1 = clf1.predict_proba(self.X_test)
        proba2 = clf2.predict_proba(self.X_test)
        self.assertFalse(np.allclose(proba1, proba2))
    
    def test_final_model_matches_xgboost(self):
        """Retraining XGBoost with final_params_ and best_num_boost_round_
        produces identical predictions to best_model_."""
        import xgboost as xgb

        clf = XGBOptClf(n_trials=5, random_state=42)
        clf.fit(self.X_dev, self.y_dev)

        dtrain = xgb.DMatrix(self.X_dev, label=self.y_dev)
        dtest  = xgb.DMatrix(self.X_test)

        manual_model = xgb.train(
            clf.final_params_,
            dtrain,
            num_boost_round=clf.best_num_boost_round_
        )

        np.testing.assert_array_almost_equal(
            clf.best_model_.predict(dtest),
            manual_model.predict(dtest)
        )

if __name__ == "__main__":
    unittest.main()