"""Statistical analyses: descriptives, collinearity, OLS, dominance analysis, validation."""
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import CV_FOLDS, CV_REPEATS, FEATURES, LABELS, N_BOOTSTRAP, SEED, TARGET


def _metrics(y, pred):
    return {
        "R2": r2_score(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": mean_absolute_error(y, pred),
    }


# ---------------------------------------------------------------- descriptives
def descriptives(df):
    """Five-number summary with Tukey whiskers (most extreme value within 1.5 x IQR)."""
    rows = []
    for col in [TARGET] + FEATURES:
        x = df[col]
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "variable": LABELS[col], "n": len(x), "mean": x.mean(), "sd": x.std(),
            "min": x.min(), "lower_whisker": x[x >= lo_fence].min(), "Q1": q1,
            "median": med, "Q3": q3, "upper_whisker": x[x <= hi_fence].max(),
            "max": x.max(), "IQR": iqr,
            "n_outliers": int(((x < lo_fence) | (x > hi_fence)).sum()),
        })
    return pd.DataFrame(rows)


def correlations(df):
    cols = [TARGET] + FEATURES
    return df[cols].corr().rename(index=LABELS, columns=LABELS)


def partial_correlations(df):
    """Partial correlation of each pair of sub-ratings, controlling for the other four."""
    c = df[FEATURES].corr().to_numpy()
    p = np.linalg.inv(c)
    pc = -p / np.sqrt(np.outer(np.diag(p), np.diag(p)))
    np.fill_diagonal(pc, 1.0)
    names = [LABELS[f] for f in FEATURES]
    return pd.DataFrame(pc, index=names, columns=names)


def vif(df):
    X = sm.add_constant(df[FEATURES]).to_numpy()
    out = pd.DataFrame({
        "variable": [LABELS[f] for f in FEATURES],
        "VIF": [variance_inflation_factor(X, i + 1) for i in range(len(FEATURES))],
    })
    out["tolerance"] = 1 / out["VIF"]
    return out.sort_values("VIF", ascending=False)


# ------------------------------------------------------------------------ OLS
def ols_full(df):
    """Full-sample OLS: fit statistics plus raw and standardized coefficients."""
    X, y = df[FEATURES], df[TARGET]
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    z = lambda s: (s - s.mean()) / s.std()
    betas = sm.OLS(z(y), X.apply(z)).fit().params
    coefs = pd.DataFrame({
        "variable": ["Intercept"] + [LABELS[f] for f in FEATURES],
        "b": fit.params.values,
        "beta": [np.nan] + list(betas.values),
        "se": fit.bse.values, "t": fit.tvalues.values, "p": fit.pvalues.values,
        "ci_low": fit.conf_int()[0].values, "ci_high": fit.conf_int()[1].values,
    })
    summary = {"n": int(fit.nobs), "R2": fit.rsquared, "adj_R2": fit.rsquared_adj,
               "F": fit.fvalue, "F_p": fit.f_pvalue,
               "residual_se": float(np.sqrt(fit.mse_resid)), **_metrics(y, fit.fittedvalues)}
    return summary, coefs


# --------------------------------------------------------- dominance analysis
def _all_subset_r2(X, y):
    """R-squared for every subset of predictors, keyed by a tuple of column indices."""
    sst = ((y - y.mean()) ** 2).sum()
    r2 = {(): 0.0}
    k = X.shape[1]
    for size in range(1, k + 1):
        for s in combinations(range(k), size):
            A = np.column_stack([np.ones(len(y)), X[:, s]])
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            e = y - A @ b
            r2[s] = 1 - (e @ e) / sst
    return r2


def _add(s, j):
    return tuple(sorted(s + (j,)))


def dominance(X, y):
    """General, conditional and complete dominance (Budescu, 1993; Azen & Budescu, 2003)."""
    k = X.shape[1]
    r2 = _all_subset_r2(X, y)
    cond = np.zeros((k, k))  # cond[j, size] = mean increment of j over subsets of that size
    for j in range(k):
        others = [i for i in range(k) if i != j]
        for size in range(k):
            cond[j, size] = np.mean([r2[_add(s, j)] - r2[s] for s in combinations(others, size)])
    general = cond.mean(axis=1)
    full = r2[tuple(range(k))]
    unique = np.array([full - r2[tuple(i for i in range(k) if i != j)] for j in range(k)])
    complete = {}
    for a, b in permutations(range(k), 2):
        rest = [i for i in range(k) if i not in (a, b)]
        diffs = [r2[_add(s, a)] - r2[_add(s, b)] for size in range(k - 1) for s in combinations(rest, size)]
        complete[(a, b)] = all(d > 0 for d in diffs)
    return {"r2": full, "general": general, "conditional": cond, "unique": unique, "complete": complete}


def dominance_tables(df):
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    res = dominance(X, y)
    names = [LABELS[f] for f in FEATURES]
    gen = pd.DataFrame({
        "variable": names,
        "general_dominance": res["general"],
        "share_of_R2": res["general"] / res["r2"],
        "unique_R2": res["unique"],
    }).sort_values("general_dominance", ascending=False)
    gen.insert(0, "rank", range(1, len(gen) + 1))
    cond = pd.DataFrame(res["conditional"], index=names,
                        columns=[f"k={i}" for i in range(len(FEATURES))])
    k = len(FEATURES)
    pairs = []
    for a, b in permutations(range(k), 2):
        pairs.append({
            "dominant": names[a], "dominated": names[b],
            "complete": res["complete"][(a, b)],
            "conditional": bool(np.all(res["conditional"][a] > res["conditional"][b])),
            "general": bool(res["general"][a] > res["general"][b]),
        })
    pairs = pd.DataFrame(pairs)
    pairs = pairs[pairs["general"]].reset_index(drop=True)
    return res["r2"], gen, cond, pairs


def bootstrap_dominance(df, n_boot=N_BOOTSTRAP, seed=SEED):
    """Resample companies with replacement and recompute shares of R-squared."""
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(y)
    shares = np.empty((n_boot, len(FEATURES)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        res = dominance(X[idx], y[idx])
        shares[b] = res["general"] / res["r2"]
    names = [LABELS[f] for f in FEATURES]
    ci = pd.DataFrame({
        "variable": names,
        "share_ci_low": np.percentile(shares, 2.5, axis=0),
        "share_ci_high": np.percentile(shares, 97.5, axis=0),
    })
    idx = {f: i for i, f in enumerate(FEATURES)}
    top = [idx["Culture"], idx["SeniorMgmt"], idx["Career"]]
    bottom = [idx["Comp"], idx["WLB"], idx["DI"]]
    rank_tests = {
        "P(Culture > Senior Management)": float(np.mean(shares[:, idx["Culture"]] > shares[:, idx["SeniorMgmt"]])),
        "P(Senior Management > Career)": float(np.mean(shares[:, idx["SeniorMgmt"]] > shares[:, idx["Career"]])),
        "P(top three all > bottom three)": float(np.mean(shares[:, top].min(axis=1) > shares[:, bottom].max(axis=1))),
    }
    return ci, rank_tests


# ------------------------------------------------------------------ validation
def train_test_results(df, train, test):
    model = LinearRegression().fit(df.loc[train, FEATURES], df.loc[train, TARGET])
    rows = []
    for name, mask in [("Train", train), ("Test", test)]:
        pred = model.predict(df.loc[mask, FEATURES])
        rows.append({"set": name, "n": int(mask.sum()), **_metrics(df.loc[mask, TARGET], pred)})
    return pd.DataFrame(rows)


def loocv_results(df):
    X, y = df[FEATURES], df[TARGET]
    pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return {"method": "Leave-One-Out CV", "folds": len(y), **_metrics(y, pred)}


def repeated_kfold_results(df, folds=CV_FOLDS, repeats=CV_REPEATS, seed=SEED):
    X, y = df[FEATURES], df[TARGET]
    rows = []
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(X)):
        m = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
        rows.append({"repeat": i // folds + 1, "fold": i % folds + 1,
                     **_metrics(y.iloc[te], m.predict(X.iloc[te]))})
    per_fold = pd.DataFrame(rows)
    summary = {"method": f"{folds}-fold CV x {repeats} repeats", "folds": len(per_fold),
               "R2": per_fold["R2"].mean(), "R2_sd": per_fold["R2"].std(),
               "RMSE": per_fold["RMSE"].mean(), "MAE": per_fold["MAE"].mean()}
    return summary, per_fold


# ----------------------------------------------------------- group comparison
def culture_pay_groups(df):
    """Overall rating for companies in the top/bottom quartiles of Culture and Comp."""
    c, p = df["Culture"], df["Comp"]
    hc, lc = c >= c.quantile(0.75), c <= c.quantile(0.25)
    hp, lp = p >= p.quantile(0.75), p <= p.quantile(0.25)
    groups = {
        "Top quartile Culture, top quartile Comp": hc & hp,
        "Top quartile Culture, bottom quartile Comp": hc & lp,
        "Bottom quartile Culture, top quartile Comp": lc & hp,
        "Bottom quartile Culture, bottom quartile Comp": lc & lp,
        "All companies": pd.Series(True, index=df.index),
    }
    summary = pd.DataFrame([{"group": g, "n": int(m.sum()),
                             "mean_overall": df.loc[m, TARGET].mean(),
                             "median_overall": df.loc[m, TARGET].median()} for g, m in groups.items()])
    keep = ["ticker", "company", "sector"]  # ratings omitted: see data/README.md
    keep = [k for k in keep if k in df.columns]
    members = pd.concat([df.loc[m, keep].assign(group=g) for g, m in list(groups.items())[1:3]])
    return summary, members
