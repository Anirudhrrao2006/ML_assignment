# Polynomial Regression: Geothermal Plant Assignment (IMT2024083)

Predicts the target `y` for two problems using polynomial regression:
- **var1** - steam turbine Net Power Score (6 features, max degree 10)
- **var2** - subterranean Thermal Anomaly Score (3 features, max degree 20)

## Structure
```
data/          train/test CSVs (IMT2024083_{train,test}_var{1,2}.csv)
src/common.py  shared config, data loading, model definition
src/train.py   5-fold CV over degree and alpha, refit, save model
src/predict.py loads saved models, writes test predictions
src/analysis.py regenerates the report figures
models/        saved models (.joblib) and CV results (.json)
predictions/   IMT2024083_pred_var1.csv, IMT2024083_pred_var2.csv
figures/       plots used in the report
```

## How to run
```bash
pip install -r requirements.txt
python src/train.py      # selects degree/alpha and saves models (~2-3 min)
python src/predict.py    # writes predictions/ files
python src/analysis.py   # optional: regenerates figures
```

## Method (short)
`PolynomialFeatures(degree)` -> `StandardScaler` -> `Ridge(alpha)`. Degree and alpha are
chosen by 5-fold cross-validation (seed 42) on the training set only. The test set is never
used for model selection.

| Problem | Degree | Alpha | CV MSE | CV R2 |
|---|---|---|---|---|
| var1 | 5 | 30 | 0.497 | 0.951 |
| var2 | 11 | 1 | 0.262 | 0.994 |
