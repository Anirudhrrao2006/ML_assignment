"""Training: select (degree, alpha) by 5-fold CV for each problem, refit on all
training data, and save the model + CV results.   Run:  python src/train.py"""
import json, warnings
import numpy as np, joblib
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from common import *

warnings.filterwarnings("ignore")

for var, max_deg in MAX_DEGREE.items():
    tr = load(var, "train")
    X, y = tr.drop(columns="y").values, tr["y"].values
    kf = KFold(5, shuffle=True, random_state=SEED)

    grid = {}                                   # grid[degree][alpha] = CV MSE
    for d in range(1, max_deg + 1):
        grid[d] = {}
        for a in ALPHAS:
            p = cross_val_predict(make_model(d, a), X, y, cv=kf)
            grid[d][a] = mean_squared_error(y, p)
        print(f"var{var} degree {d:2d}: best CV MSE {min(grid[d].values()):.4f}")

    d_best, a_best = min(((d, a) for d in grid for a in grid[d]), key=lambda t: grid[t[0]][t[1]])
    p = cross_val_predict(make_model(d_best, a_best), X, y, cv=kf)
    res = dict(degree=d_best, alpha=a_best, cv_mse=mean_squared_error(y, p),
               cv_r2=r2_score(y, p), grid={str(d): {str(a): v for a, v in g.items()} for d, g in grid.items()})
    model = make_model(d_best, a_best).fit(X, y)
    res["train_mse"] = mean_squared_error(y, model.predict(X))
    res["train_r2"] = r2_score(y, model.predict(X))
    joblib.dump(model, MODELS / f"model_var{var}.joblib")
    (MODELS / f"cv_results_var{var}.json").write_text(json.dumps(res, indent=1))
    print(f"==> var{var}: degree={d_best} alpha={a_best} CV MSE={res['cv_mse']:.4f} CV R2={res['cv_r2']:.4f}")
