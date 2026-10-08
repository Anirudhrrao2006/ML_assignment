"""Polynomial regression for IMT2024083 (var1: 6 features, var2: 3 features).
Usage: put the 4 CSVs in the same folder as this script, then run:  python poly_regression.py
Outputs: IMT2024083_pred_var1.csv, IMT2024083_pred_var2.csv
"""
import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

ROLL = "IMT2024083"
CFG = {1: range(1, 11), 2: range(1, 21)}        # max degree 10 for var1, 20 for var2
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100, 300]

def make(deg, alpha):
    return make_pipeline(PolynomialFeatures(deg, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))

for v, degs in CFG.items():
    tr = pd.read_csv(f"{ROLL}_train_var{v}.csv")
    te = pd.read_csv(f"{ROLL}_test_var{v}.csv")
    X, y = tr.drop(columns="y").values, tr["y"].values
    kf = KFold(5, shuffle=True, random_state=42)

    best = (np.inf, None, None)
    print(f"\n=== var{v} : 5-fold CV (MSE) ===")
    for d in degs:
        for a in ALPHAS:
            pred = cross_val_predict(make(d, a), X, y, cv=kf)
            mse = mean_squared_error(y, pred)
            if mse < best[0]:
                best = (mse, d, a)
        print(f"degree {d:2d} done, best so far: deg={best[1]} alpha={best[2]} MSE={best[0]:.6f}")

    mse, d, a = best
    pred = cross_val_predict(make(d, a), X, y, cv=kf)
    print(f"--> var{v} chosen degree={d}, alpha={a}, CV MSE={mse:.6f}, CV R2={r2_score(y, pred):.4f}")

    model = make(d, a).fit(X, y)               # refit on all training data
    out = pd.DataFrame({"y_pred": model.predict(te.values)})
    out.to_csv(f"{ROLL}_pred_var{v}.csv", index=False)
    print(f"saved {ROLL}_pred_var{v}.csv ({len(out)} rows)")
