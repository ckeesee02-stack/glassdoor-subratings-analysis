"""Run the full analysis and write every table to results/ and every chart to figures/.

Usage:
    python run_all.py              # full run (about a minute; includes 1,000 bootstrap resamples)
    python run_all.py --quick      # skips the bootstrap
"""
import argparse
import json

import pandas as pd

from src import analysis as A
from src import charts
from src.config import FIGURES_DIR, RESULTS_DIR
from src.data import load_data, split_masks


def main(quick=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load_data()
    train, test = split_masks(df)
    save = lambda t, name, **kw: t.round(4).to_csv(RESULTS_DIR / name, **kw)

    save(A.descriptives(df), "descriptives.csv", index=False)
    corr = A.correlations(df)
    save(corr, "correlations.csv")
    save(A.partial_correlations(df), "partial_correlations.csv")
    save(A.vif(df), "vif.csv", index=False)

    ols_summary, coefs = A.ols_full(df)
    save(coefs, "ols_coefficients.csv", index=False)

    r2, gen, cond, pairs = A.dominance_tables(df)
    save(cond, "dominance_conditional.csv")
    pairs.to_csv(RESULTS_DIR / "dominance_pairs.csv", index=False)

    rank_tests = {}
    if not quick:
        ci, rank_tests = A.bootstrap_dominance(df)
        gen = gen.merge(ci, on="variable")
        charts.dominance_chart(gen, FIGURES_DIR / "03_dominance_analysis.png")
    save(gen, "dominance_general.csv", index=False)

    tt = A.train_test_results(df, train, test)
    loo = A.loocv_results(df)
    kf, folds = A.repeated_kfold_results(df)
    validation = pd.concat([tt.rename(columns={"set": "method"}), pd.DataFrame([loo, kf])], ignore_index=True)
    save(validation, "validation.csv", index=False)
    save(folds, "cv_folds.csv", index=False)

    groups, members = A.culture_pay_groups(df)
    save(groups, "culture_pay_groups.csv", index=False)
    save(members, "culture_pay_group_members.csv", index=False)

    summary = {"full_sample_ols": ols_summary, "bootstrap_rank_tests": rank_tests}
    (RESULTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    charts.correlation_heatmap(corr, FIGURES_DIR / "01_correlation_matrix.png")
    charts.box_plot(df, FIGURES_DIR / "02_rating_distributions.png")

    print(f"Full-sample OLS: R2 = {ols_summary['R2']:.3f}, RMSE = {ols_summary['RMSE']:.3f}, MAE = {ols_summary['MAE']:.3f}")
    print(validation.round(3).to_string(index=False))
    print(gen.round(3).to_string(index=False))
    for k, v in rank_tests.items():
        print(f"{k}: {v:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="skip the bootstrap")
    main(p.parse_args().quick)
