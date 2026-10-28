"""Tests run on synthetic data, so they work without the (unpublished) Glassdoor dataset."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import dominance, descriptives, vif
from src.config import FEATURES, TARGET
from src.data import load_data


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(3.6, 0.3, n)
    df = pd.DataFrame({f: np.clip(base + rng.normal(0, 0.2, n), 1, 5) for f in FEATURES})
    df[TARGET] = np.clip(df[FEATURES].mean(axis=1) + rng.normal(0, 0.05, n), 1, 5)
    df.insert(0, "ticker", [f"T{i}" for i in range(n)])
    return df


def test_general_dominance_sums_to_r2():
    df = _synthetic()
    res = dominance(df[FEATURES].to_numpy(), df[TARGET].to_numpy())
    assert res["general"].sum() == pytest.approx(res["r2"], abs=1e-10)


def test_orthogonal_predictors_get_their_squared_correlation():
    # With uncorrelated predictors, each general dominance weight equals r^2 with y.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 3))
    X = np.linalg.qr(X - X.mean(axis=0))[0]
    y = X @ np.array([3.0, 2.0, 1.0]) + rng.normal(0, 0.01, 5000)
    res = dominance(X, y)
    r2 = np.array([np.corrcoef(X[:, j], y)[0, 1] ** 2 for j in range(3)])
    assert np.allclose(res["general"], r2, atol=1e-4)


def test_unique_r2_is_never_larger_than_general():
    df = _synthetic()
    res = dominance(df[FEATURES].to_numpy(), df[TARGET].to_numpy())
    assert np.all(res["unique"] <= res["general"] + 1e-12)


def test_whiskers_stay_inside_range():
    d = descriptives(_synthetic())
    assert (d["lower_whisker"] >= d["min"]).all()
    assert (d["upper_whisker"] <= d["max"]).all()
    assert (d["lower_whisker"] <= d["Q1"]).all()


def test_vif_at_least_one():
    assert (vif(_synthetic())["VIF"] >= 1).all()


def test_load_data_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    _synthetic().drop(columns=["Culture"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_data(p)


def test_load_data_rejects_out_of_range(tmp_path):
    df = _synthetic()
    df.loc[0, "Comp"] = 6
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="1 to 5"):
        load_data(p)
