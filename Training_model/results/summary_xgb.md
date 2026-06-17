# XGBoost: clean vs noisy

Same windowing, features, folds, and aggregation as `train_compare.py`
(see `results/summary.md`); only the classifier is swapped from
`SVC(rbf)` to `XGBClassifier`.

## XGBoost hyperparameters

```
{
  "objective": "multi:softprob",
  "num_class": 3,
  "n_estimators": 400,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "min_child_weight": 2,
  "reg_lambda": 1.0,
  "tree_method": "hist",
  "eval_metric": "mlogloss",
  "random_state": 42,
  "n_jobs": -1
}
```

## Results (per-track accuracy, mean +/- std across 5 folds)

| experiment       | accuracy            | jazz | metal | pop |
|------------------|---------------------|------|-------|-----|
| clean -> clean   | **0.980 +/- 0.025** | 0.99 | 0.99 | 0.96 |
| noisy -> noisy   | **0.852 +/- 0.030** | 0.87 | 0.83 | 0.86 |

- **Absolute accuracy drop from noise:** +0.128
- **Relative accuracy drop from noise:** +13.1 %

Window-level accuracy (informational):
- clean -> clean: 0.962 +/- 0.017
- noisy -> noisy: 0.764 +/- 0.045

Confusion matrices: `confusion_clean_xgb.png`, `confusion_noisy_xgb.png`.
