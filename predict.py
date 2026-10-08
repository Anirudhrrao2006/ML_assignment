"""Inference: load the saved models and write predictions for the test sets.
Run:  python src/predict.py   (after train.py)"""
import joblib, pandas as pd
from common import *

for var in MAX_DEGREE:
    model = joblib.load(MODELS / f"model_var{var}.joblib")
    te = load(var, "test")
    out = pd.DataFrame({"y_pred": model.predict(te.values)})
    out.to_csv(PRED / f"{ROLL}_pred_var{var}.csv", index=False)
    print(f"var{var}: wrote {len(out)} predictions")
