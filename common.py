"""Shared helpers: data loading and model construction."""
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pandas as pd

ROLL = "IMT2024083"
ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, PRED, FIGS = (ROOT / d for d in ("data", "models", "predictions", "figures"))
for d in (MODELS, PRED, FIGS):
    d.mkdir(exist_ok=True)

MAX_DEGREE = {1: 10, 2: 20}                      # limits given in the assignment
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100, 300]
SEED = 42


def load(var, split):
    return pd.read_csv(DATA / f"{ROLL}_{split}_var{var}.csv")


def make_model(degree, alpha):
    """Polynomial features (total degree <= `degree`) -> standardise -> Ridge."""
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))
