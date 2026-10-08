"""Generates the figures + extra numbers used in the report. Run after train.py."""
import json, warnings
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from math import comb
from common import *
warnings.filterwarnings("ignore")

summary = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, ax2, var in zip(axes, axes2, MAX_DEGREE):
    res = json.loads((MODELS / f"cv_results_var{var}.json").read_text())
    tr = load(var, "train"); X, y = tr.drop(columns="y").values, tr["y"].values
    nf = X.shape[1]; degs = sorted(int(d) for d in res["grid"])
    best_reg = [min(res["grid"][str(d)].values()) for d in degs]
    kf = KFold(5, shuffle=True, random_state=SEED)
    # unregularised OLS where #terms < #samples (otherwise it is under-determined)
    ols = {}
    for d in degs:
        if comb(nf + d, d) - 1 < 0.8 * len(y):
            p = cross_val_predict(make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression()), X, y, cv=kf)
            ols[d] = mean_squared_error(y, p)
    ax.semilogy(degs, best_reg, "o-", label="Ridge (alpha tuned per degree)")
    ax.semilogy(list(ols), list(ols.values()), "s--", label="OLS (no regularisation)")
    ax.axvline(res["degree"], color="gray", ls=":", label=f"chosen degree = {res['degree']}")
    ax.set_xlabel("polynomial degree"); ax.set_ylabel("5-fold CV MSE"); ax.set_title(f"var{var}"); ax.legend(fontsize=7)
    p = cross_val_predict(make_model(res["degree"], res["alpha"]), X, y, cv=kf)
    ax2.scatter(y, p, s=6, alpha=.5); lim = [y.min(), y.max()]; ax2.plot(lim, lim, "r--")
    ax2.set_xlabel("actual y"); ax2.set_ylabel("CV-predicted y"); ax2.set_title(f"var{var}: out-of-fold predictions")
    summary[var] = dict(n=len(y), n_features=nf, ols_cv_mse={str(k): v for k, v in ols.items()},
                        best_per_degree={str(d): v for d, v in zip(degs, best_reg)},
                        n_terms_chosen=comb(nf + res["degree"], res["degree"]) - 1,
                        y_mean=float(y.mean()), y_std=float(y.std()), y_min=float(y.min()), y_max=float(y.max()),
                        corr_x_y=[float(np.corrcoef(X[:, i], y)[0, 1]) for i in range(nf)])
fig.tight_layout(); fig.savefig(FIGS / "cv_mse_vs_degree.png", dpi=170)
fig2.tight_layout(); fig2.savefig(FIGS / "cv_pred_vs_actual.png", dpi=170)
(MODELS / "analysis_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({v: {k: s[k] for k in ("n_terms_chosen", "ols_cv_mse")} for v, s in summary.items()}, indent=1))
