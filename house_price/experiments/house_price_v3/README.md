# House Price v3

This experiment follows `PIPELINE.md` and keeps all new artifacts inside this directory.

## Workflow and findings

- **Prepare Problem:** Ames House Prices regression; Kaggle's official metric is RMSLE on `SalePrice`.
- **Summarize Data:** 1,460 labeled rows, mixed numeric/categorical columns, missing values, and right-skewed prices/features.
- **Prepare Data:** v2 preprocessing was reproduced for `EXP-01`. Other experiments use train-fitted median imputation, one-hot encoding, standard scaling, and justified aggregate features (`TotalSF`, bathrooms, porch area, ages, and quality-area interactions). No target or test values are used to fit validation preprocessing.
- **Evaluate Algorithms:** fixed 80/20 split with `random_state=42`, PyTorch MLP, RMSLE as the selection metric, and raw RMSE as a secondary diagnostic.
- **Improve Accuracy:** target transformation and normalized target scale were tested separately from feature engineering and regularization.
- **Finalize Model:** the selected model is retrained on all labeled rows with the selected configuration, saved, and used for the Kaggle-format submission.

## Experiment comparison

| ID | Main change | Validation RMSLE | Raw RMSE | Conclusion |
| --- | --- | ---: | ---: | --- |
| v2 baseline | Original notebook, raw target | n/a | 40278.375 | Reference reproduced from v2 |
| EXP-01 | v2 reproduction in runner | 0.202642 | 39656.75 | Reproduces the pipeline and is slightly better on this run |
| EXP-02 | Scaled features + log target | 0.693002 | 110544.95 | Rejected |
| EXP-03 | Engineered/scaled features + regularization | 0.902982 | 144614.39 | Rejected |
| EXP-04 | Regularized architecture tuning | 0.881250 | 90584.94 | Rejected |
| EXP-05 | Engineered/scaled features + normalized log target | **0.135129** | **24119.70** | Selected |

The v2 notebook's official raw-price RMSE was `40278.3750`. EXP-05 is selected by Kaggle-aligned RMSLE and also improves the secondary raw RMSE diagnostic.

## Reproduction

```powershell
python house_price/experiments/house_price_v3/train.py
```

Artifacts:

- `models/best_model.pt`
- `configs/best_params.json`
- `results/experiment_results.csv`
- `results/best_validation_history.csv`
- `submission.csv`