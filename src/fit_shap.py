"""
Phase 2 -- SHAP explainability for the XGBoost model (task queue item 4,
CLAUDE.md), and specifically resolving the open question fit_xgboost.py
flagged: whether the raw provider-identifier features (ORG_NPI_NUM,
TAX_NUM, CARR_CLM_BLG_NPI_NUM, PRF_PHYSN_UPIN, PRVDR_NUM) that rank highly
by gain-based importance are a real, generalizable "this provider is
denied more often" signal, or the tree memorizing individual providers'
small-sample noise (gain-based importance alone can't distinguish these).

REUSES THE EXACT FITTED MODEL, DOES NOT REFIT. fit_xgboost.py now saves
the fitted booster (reports/xgboost_model.json, native XGBoost format) and
its column/category metadata (reports/xgboost_model_meta.json -- colnames,
which columns are categorical, and train's exact category list per column,
same train-governs-val discipline as everywhere else in this project) right
after training. This script loads both and RE-VERIFIES against
reports/xgboost_fit.json's already-reported val metrics (must match to
1e-6) before doing anything else -- ruling out any risk of SHAP explaining
a subtly different model than the one whose results are already written
up (a stale model file, a category-list mismatch, etc. would show up here
as a metrics mismatch, not a silent wrong answer).

TREE EXPLAINER, TESTED ON SYNTHETIC DATA FIRST (this project's standing
rule for new modeling-adjacent code): shap.TreeExplainer with
feature_perturbation="tree_path_dependent" is the natural choice for a
boosted-tree model and handles xgboost's native categorical splits and
missing values the same way training did. Verified on synthetic data with
a categorical column, NaN, and planted signal + noise that SHAP values sum
to (prediction margin - expected_value) to ~1e-6 before running on real
data (see the 2026-09-25 session log for the exact check).

SAMPLE, NOT ALL OF VAL: SHAP is expensive per row (its cost scales with
tree count x average path length x rows). Computed on a stratified sample
of val (default 20,000 rows, same label-stratification idea as
fit_chow_test.py's streaming sampler, but drawn from the already-loaded
val DataFrame in memory rather than re-streamed from disk).

METRICS/IMPORTANCE, NEVER ONLY OVERALL (this project's standing rule):
mean |SHAP| feature importance is reported overall AND per claim type,
since a feature could matter enormously in one claim type and not at all
in another -- gain-based importance (fit_xgboost.py) can't distinguish
that either.

THE PROVIDER-ID QUESTION, ANSWERED QUANTITATIVELY, NOT BY EYEBALLING A
BAR CHART: for each of the 5 flagged ID columns, this script computes each
category's mean SHAP value in the sample and its claim COUNT in train,
then reports the correlation between |mean SHAP| and log(claim count). A
genuine "this provider has a denial history" signal should be at least as
strong (or stronger) for providers with MANY claims, since that's a
stable estimate of their denial history; if instead the largest |SHAP|
values are concentrated on providers with very few claims, that's the
signature of the tree fitting small-sample noise rather than a
generalizable pattern.

Run (repo root, venv active):
    python src/fit_shap.py
    python src/fit_shap.py --sample-size 2000   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from fit_baseline_model import _format_block, _metrics_block
from fit_xgboost import (
    MODEL_META_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    VAL_PATH,
    _apply_categories,
    _columns_to_load,
    _split_xy,
)

TRAIN_PATH = PROCESSED_DIR / "train_model.parquet"

_CLAIM_TYPES = ("carrier", "outpatient", "dme")
_PROVIDER_ID_COLS = ("ORG_NPI_NUM", "TAX_NUM", "CARR_CLM_BLG_NPI_NUM", "PRF_PHYSN_UPIN", "PRVDR_NUM")

_METRICS_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP explainability for the Phase 2 XGBoost model.")
    p.add_argument("--sample-size", type=int, default=20_000, help="Rows of val to compute SHAP on (stratified by label).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_model_and_meta() -> tuple[XGBClassifier, dict]:
    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} / {MODEL_META_PATH} not found -- run `python src/fit_xgboost.py` "
            "(full size, no --sample-frac) first so there's a real model to explain."
        )
    meta = json.loads(MODEL_META_PATH.read_text())
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, meta


def _rebuild_val(meta: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Reload val_model.parquet and apply train's exact column order +
    category lists from `meta` -- mirrors fit_xgboost.py's own val-prep
    exactly, but from saved metadata rather than a live `categories_by_col`
    (this script never refits, so it has no live one)."""
    columns = _columns_to_load(TRAIN_PATH)
    val_raw = pd.read_parquet(VAL_PATH, columns=columns)
    y_val, X_val = _split_xy(val_raw)
    del val_raw
    gc.collect()
    X_val = _apply_categories(X_val, meta["cat_cols"], meta["categories_by_col"])
    X_val = X_val.reindex(columns=meta["colnames"])
    return y_val, X_val


def _verify_against_reported_metrics(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
    """Predict on the full val set and check against reports/xgboost_fit.json's
    already-reported overall metrics, to 1e-6 -- if the loaded model/meta
    don't reproduce those numbers exactly, something about the save/load
    round-trip is wrong and SHAP would be explaining the wrong model."""
    p_val = model.predict_proba(X_val)[:, 1]
    y_val_arr = y_val.to_numpy(dtype=np.float64)
    recomputed = _metrics_block(y_val_arr, p_val)

    reported_path = REPORTS_DIR / "xgboost_fit.json"
    reported = json.loads(reported_path.read_text())["val_metrics_overall"]

    mismatches = []
    for key in ("pr_auc", "roc_auc", "brier", "mean_predicted"):
        if abs(recomputed[key] - reported[key]) > _METRICS_TOL:
            mismatches.append(f"{key}: loaded model gives {recomputed[key]!r}, reported {reported[key]!r}")
    if mismatches:
        raise RuntimeError(
            "Loaded xgboost_model.json does NOT reproduce reports/xgboost_fit.json's "
            "val metrics -- refusing to run SHAP on a model that isn't verified to be "
            "the one already reported. Mismatches:\n  " + "\n  ".join(mismatches)
        )
    print(f"    verified: loaded model reproduces reported val metrics to {_METRICS_TOL:g} -- {_format_block('OVERALL', recomputed)}")
    return p_val


def _stratified_index_sample(y: pd.Series, n: int, seed: int) -> np.ndarray:
    """Stratified sample of row POSITIONS (not labels/index values) from an
    already-in-memory y, same proportional-by-label idea as
    fit_chow_test._stratified_sample_streaming, just applied in memory since
    val already fits."""
    rng = np.random.RandomState(seed)
    positions = np.arange(len(y))
    y_arr = y.to_numpy()
    chosen = []
    for val in np.unique(y_arr):
        idx = positions[y_arr == val]
        n_take = min(int(round(len(idx) * n / len(y))), len(idx))
        chosen.append(rng.choice(idx, size=n_take, replace=False))
    chosen = np.sort(np.concatenate(chosen))
    return chosen


def main() -> None:
    args = parse_args()

    print("=== Loading fitted model + metadata (not refitting) ===")
    model, meta = _load_model_and_meta()
    print(f"    model: {meta['best_iteration']} trees, {len(meta['colnames'])} features ({len(meta['cat_cols'])} categorical)")

    print("\n=== Rebuilding val with train's saved categories, verifying against reported metrics ===")
    y_val, X_val = _rebuild_val(meta)
    p_val_full = _verify_against_reported_metrics(model, X_val, y_val)

    print(f"\n=== Sampling {args.sample_size:,} val rows (stratified by label) for SHAP ===")
    sample_pos = _stratified_index_sample(y_val, args.sample_size, args.seed)
    X_sample = X_val.iloc[sample_pos].reset_index(drop=True)
    y_sample = y_val.iloc[sample_pos].reset_index(drop=True)
    p_sample = p_val_full[sample_pos]
    print(f"    sampled {len(X_sample):,} rows, {y_sample.mean():.2%} positive (val overall: {y_val.mean():.2%})")

    print("\n=== Computing SHAP values (TreeExplainer, tree_path_dependent) ===")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sample)
    recon_err = np.max(np.abs(
        shap_values.sum(axis=1) + explainer.expected_value
        - model.predict(X_sample, output_margin=True)
    ))
    print(f"    max |sum(SHAP) + base_value - predicted margin| over the sample: {recon_err:.3e} (sanity check)")
    if recon_err > 1e-3:
        raise RuntimeError(f"SHAP values do not reconstruct the model's predictions (max error {recon_err:.3e}) -- something is wrong, not proceeding.")

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)

    val_types = pd.read_parquet(VAL_PATH, columns=["claim_type_carrier", "claim_type_outpatient", "claim_type_dme"])
    val_types_sample = val_types.iloc[sample_pos].reset_index(drop=True)

    by_claim_type_importance: dict[str, pd.Series] = {}
    for ct in _CLAIM_TYPES:
        mask = (val_types_sample[f"claim_type_{ct}"] == 1).to_numpy()
        by_claim_type_importance[ct] = pd.Series(
            np.abs(shap_values[mask]).mean(axis=0), index=X_sample.columns
        ).sort_values(ascending=False)

    print("\n=== Provider-identifier question: generalizable signal, or small-sample memorization? ===")
    train_counts: dict[str, pd.Series] = {}
    for col in _PROVIDER_ID_COLS:
        train_counts[col] = pd.read_parquet(TRAIN_PATH, columns=[col])[col].value_counts()

    provider_id_report: dict[str, dict] = {}
    for i, col in enumerate(X_sample.columns):
        if col not in _PROVIDER_ID_COLS:
            continue
        col_shap = shap_values[:, i]
        cat_vals = X_sample[col].astype(object)
        df = pd.DataFrame({"cat": cat_vals, "abs_shap": np.abs(col_shap)})
        per_cat = df.groupby("cat", observed=True)["abs_shap"].mean()
        counts = train_counts[col]
        joined = per_cat.to_frame("mean_abs_shap").join(counts.rename("train_claim_count"), how="inner")
        joined = joined[joined["train_claim_count"] > 0]
        if len(joined) < 3:
            provider_id_report[col] = {"note": "too few categories present in sample to correlate"}
            continue
        log_count = np.log10(joined["train_claim_count"].to_numpy())
        corr = float(np.corrcoef(log_count, joined["mean_abs_shap"].to_numpy())[0, 1])
        low_n = joined[joined["train_claim_count"] <= joined["train_claim_count"].median()]
        high_n = joined[joined["train_claim_count"] > joined["train_claim_count"].median()]
        provider_id_report[col] = {
            "mean_abs_shap_overall": float(mean_abs_shap.get(col, float("nan"))),
            "n_categories_in_sample": int(len(joined)),
            "corr_log_train_count_vs_mean_abs_shap": corr,
            "mean_abs_shap_low_train_count_half": float(low_n["mean_abs_shap"].mean()),
            "mean_abs_shap_high_train_count_half": float(high_n["mean_abs_shap"].mean()),
        }

    lines = [
        "Phase 2 XGBoost -- SHAP explainability",
        "=" * 62,
        f"Model: {meta['best_iteration']} trees (loaded from {MODEL_PATH.name}, verified against reported val metrics)",
        f"SHAP sample: {len(X_sample):,} of {len(y_val):,} val rows (stratified by label, seed={args.seed})",
        "",
        "Top 20 features by mean |SHAP| (overall):",
    ]
    for name, val in mean_abs_shap.head(20).items():
        lines.append(f"  {name:40s} {val:.4f}")

    lines.append("")
    lines.append("Top 10 by mean |SHAP|, per claim type (never only overall):")
    for ct in _CLAIM_TYPES:
        lines.append(f"  {ct.upper()}:")
        for name, val in by_claim_type_importance[ct].head(10).items():
            lines.append(f"    {name:38s} {val:.4f}")

    lines.append("")
    lines.append("Provider-identifier question -- correlation of |mean SHAP| with log(train claim count),")
    lines.append("per category of each flagged ID column. Positive/near-zero => bigger claim history gives")
    lines.append("at least as much SHAP weight (consistent with a real provider-risk signal). Negative =>")
    lines.append("low-claim-count categories carry more SHAP weight (consistent with memorizing small-sample")
    lines.append("noise rather than a generalizable pattern).")
    for col in _PROVIDER_ID_COLS:
        r = provider_id_report.get(col, {})
        if "note" in r:
            lines.append(f"  {col:24s} {r['note']}")
            continue
        lines.append(
            f"  {col:24s} corr={r['corr_log_train_count_vs_mean_abs_shap']:+.3f}  "
            f"low-claim-count half mean|SHAP|={r['mean_abs_shap_low_train_count_half']:.4f}  "
            f"high-claim-count half mean|SHAP|={r['mean_abs_shap_high_train_count_half']:.4f}  "
            f"(n_categories={r['n_categories_in_sample']}, overall mean|SHAP|={r['mean_abs_shap_overall']:.4f})"
        )

    report = "\n".join(lines)
    print("\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = REPORTS_DIR / "shap_results.txt"
    out_txt.write_text(report + "\n")

    out_json = REPORTS_DIR / "shap_fit.json"
    out_json.write_text(json.dumps({
        "sample_size": len(X_sample),
        "seed": args.seed,
        "shap_reconstruction_max_error": float(recon_err),
        "mean_abs_shap_overall_top20": {k: float(v) for k, v in mean_abs_shap.head(20).items()},
        "mean_abs_shap_by_claim_type_top10": {
            ct: {k: float(v) for k, v in by_claim_type_importance[ct].head(10).items()} for ct in _CLAIM_TYPES
        },
        "provider_id_question": provider_id_report,
    }, indent=2))
    print(f"\nWritten to {out_txt} and {out_json}")


if __name__ == "__main__":
    main()
