"""
Phase 2 -- fit XGBoost (task queue item 3, CLAUDE.md), on the same
train_model.parquet / val_model.parquet as the baseline mixed logistic
model, with the same metrics (PR-AUC primary, plus ROC-AUC/Brier, overall
and per claim type -- this project's standing rule, never only overall).

DOES XGBOOST NEED THE BASELINE'S ENCODING? CONFIRMED, NOT ASSUMED
(CLAUDE.md task 3 explicitly asks this be checked, not skipped past):
  - Categorical cardinality: build_chow_design_matrix.py's top-n encoding
    (HCPCS/diagnosis -> top 20 + __OTHER__/__MISSING__) and frequency
    encoding (PRVDR_NUM) exist ONLY because a linear model needs a fixed,
    small number of dummy columns. A tree can split on a raw categorical
    value directly and can have as many leaves as the data supports, so
    none of that reduction is needed here -- provider_state, HCPCS_CD,
    PRNCPAL_DGNS_CD, PRVDR_NUM, CARR_NUM all go in at FULL cardinality via
    pandas 'category' dtype and xgboost's native categorical split support
    (tree_method="hist", enable_categorical=True). No information lost.
  - The rank-deficiency fixes (build_chow_design_matrix.py fixes A-D) are
    ALSO linear-model-only: constant-within-claim-type columns and exact
    linear identities cost a linear model its rank, but cost a tree
    nothing (a constant column just never gets split on; a linear
    identity gives the tree two equally good, redundant features). None
    of that removal is reused either.
  - Claim-type interaction terms (build_mixed_design_matrix, or the
    Chow test's per-claim-type __x__ dummies): also linear-model-only. A
    tree splits on claim_type_* directly and can then split differently
    per branch, which already gives every covariate an implicit
    claim-type-specific effect -- no *__x__claim_type columns are built.
  - What XGBoost DOES newly get to use, that the baseline never could:
    the ~130 "pending encoding" columns fit_chow_test.py's _PENDING_*
    groups exclude (secondary/tertiary diagnosis & procedure codes,
    legacy provider ID strings, other CMS categorical codes, line
    sequence numbers) -- excluded from the baseline only because a linear
    model had no defensible cardinality-reduction scheme picked yet for
    them. They cost a tree nothing to include at full cardinality, so
    they're in this run's feature set (see _PENDING_INCLUDE_AS_CATEGORY
    below) -- a genuine capability difference worth calling out in the
    model comparison, not just a fitting-mechanics one.
  - Kept EXCLUDED, same as the baseline, because no Phase 2 date
    transform has been decided for either model yet (raw strings like
    "01-Apr-2015" aren't a defensible tree feature either without some
    transform -- day count, weekday, etc. -- and picking one here would be
    inventing a new encoding decision mid-task, not confirming an existing
    one): CLM_FROM_DT, CLM_THRU_DT, NCH_WKLY_PROC_DT, LINE_1ST_EXPNS_DT,
    LINE_LAST_EXPNS_DT, REV_CNTR_DT, PRCDR_DT1-24.

TRAIN/VAL CATEGORY CONSISTENCY. Still needed even without top-n/frequency
encoding: xgboost's categorical split support keys off the pandas
CategoricalDtype's category LIST, so if train and val independently called
.astype("category") they could disagree on which codes exist and, worse,
which integer code a given category maps to -- silently corrupting
predictions rather than raising. Fixed the same way as the baseline:
train's category list per column is recorded and reused verbatim on val
(a val-only category becomes NaN, which xgboost treats as a normal missing
value, not an error) -- see _to_categorical_matching_train().

MISSING VALUES. Every remaining NaN (structural claim-type absence, e.g.
carrier-only fields on an outpatient row) is passed through AS NaN, not
zero-filled -- xgboost's hist method learns a default split direction for
missing values per split, natively, which is a real advantage over the
baseline's forced zero-fill-plus-presence-dummy pattern.

VALIDATION USE. val is used for early stopping (eval_set) here, same
train/val split as the baseline, with test_model.parquet held out
completely untouched -- consistent with a standard train/val/test split
where val is for model selection and test is reserved for the final,
one-time comparison across models later in the project.

Run (repo root, venv active):
    python src/fit_xgboost.py
    python src/fit_xgboost.py --sample-frac 0.02   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fit_baseline_model import _format_block, _metrics_block
from fit_chow_test import (
    LABEL_COL,
    _ID_LABEL_COLS,
    _DATE_COLS_NOT_YET_FEATURIZED,
    _PENDING_DATES,
    _stratified_sample_streaming,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
TRAIN_PATH = PROCESSED_DIR / "train_model.parquet"
VAL_PATH = PROCESSED_DIR / "val_model.parquet"
MODEL_PATH = REPORTS_DIR / "xgboost_model.json"
MODEL_META_PATH = REPORTS_DIR / "xgboost_model_meta.json"

_CLAIM_TYPES = ("carrier", "outpatient", "dme")

# Only ID/label columns and un-transformed raw dates are excluded -- see
# module docstring for why everything else (including the baseline's
# "pending encoding" columns) is kept for this model.
_EXCLUDE_COLS = _ID_LABEL_COLS | _DATE_COLS_NOT_YET_FEATURIZED | _PENDING_DATES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit Phase 2 XGBoost, same metrics as the baseline model.")
    p.add_argument("--sample-frac", type=float, default=None, help="Stratified subsample of TRAIN only (smoke test). val is always used in full.")
    p.add_argument("--stream-batch-size", type=int, default=50_000)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    return p.parse_args()


def _columns_to_load(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(parquet_path).schema_arrow.names
    return [c for c in names if c not in _EXCLUDE_COLS or c == LABEL_COL]


def _load(path: Path, columns: list[str], sample_frac: float | None, batch_size: int) -> pd.DataFrame:
    if sample_frac is None:
        print(f"Loading {path.name} (full; {len(columns)} columns)...")
        return pd.read_parquet(path, columns=columns)
    print(f"Streaming a --sample-frac {sample_frac} stratified subsample of {path.name}...")
    return _stratified_sample_streaming(path, sample_frac, LABEL_COL, batch_size, columns=columns)


def _split_xy(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])
    return y, X


def _object_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def _to_train_categories(X: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, dict[str, list]]:
    """Cast each object column to pandas 'category', fitting the category
    list fresh from this (train) data. Returns (X, categories_by_col) so
    the caller can pass categories_by_col into _apply_categories() for val."""
    X = X.copy()
    categories_by_col: dict[str, list] = {}
    for col in cat_cols:
        X[col] = X[col].astype("category")
        categories_by_col[col] = list(X[col].cat.categories)
    return X, categories_by_col


def _apply_categories(X: pd.DataFrame, cat_cols: list[str], categories_by_col: dict[str, list]) -> pd.DataFrame:
    """Cast each object column to 'category' using TRAIN's exact category
    list, not this split's own -- a val-only value becomes NaN (xgboost's
    native missing-value handling), never a category the model was never
    fit against. Mirrors build_chow_design_matrix.py's top_n_encode/
    frequency_encode discipline for the baseline model.

    Masks any val-only value to NaN BEFORE casting (rather than casting
    then letting pandas coerce out-of-category values) -- pandas 4 treats
    that coercion path as deprecated and warns on every call."""
    X = X.copy()
    for col in cat_cols:
        cats = categories_by_col[col]
        dtype = pd.CategoricalDtype(categories=cats)
        masked = X[col].where(X[col].isin(cats), other=np.nan)
        X[col] = masked.astype(dtype)
    return X


def main() -> None:
    args = parse_args()
    columns = _columns_to_load(TRAIN_PATH)

    print("=== Loading train ===")
    train_raw = _load(TRAIN_PATH, columns, args.sample_frac, args.stream_batch_size)
    y_train, X_train = _split_xy(train_raw)
    del train_raw
    gc.collect()

    cat_cols = _object_columns(X_train)
    print(f"train: {X_train.shape[0]:,} rows x {X_train.shape[1]} columns ({len(cat_cols)} categorical), {y_train.mean():.2%} positive")
    X_train, categories_by_col = _to_train_categories(X_train, cat_cols)

    print("\n=== Loading val (train's categories reused) ===")
    val_raw = pd.read_parquet(VAL_PATH, columns=columns)
    y_val, X_val = _split_xy(val_raw)
    del val_raw
    gc.collect()
    X_val = _apply_categories(X_val, cat_cols, categories_by_col)
    X_val = X_val.reindex(columns=list(X_train.columns))
    print(f"val:   {X_val.shape[0]:,} rows x {X_val.shape[1]} columns, {y_val.mean():.2%} positive")

    print(f"\n=== Fitting XGBoost (n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}) ===")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = model.best_iteration
    print(f"    early-stopped at {best_iter} trees (of {args.n_estimators} allowed, patience {args.early_stopping_rounds})")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps({
        "colnames": list(X_train.columns),
        "cat_cols": cat_cols,
        "categories_by_col": {k: list(v) for k, v in categories_by_col.items()},
        "best_iteration": int(best_iter),
    }, indent=2))
    print(f"    saved fitted model to {MODEL_PATH} (+ column/category metadata to {MODEL_META_PATH})")

    print("\n=== Evaluating on val ===")
    p_val = model.predict_proba(X_val)[:, 1]
    y_val_arr = y_val.to_numpy(dtype=np.float64)

    overall = _metrics_block(y_val_arr, p_val)
    lines = [
        "Phase 2 XGBoost -- val set results",
        "=" * 62,
        f"Trees: {best_iter} (early-stopped; max_depth={args.max_depth}, learning_rate={args.learning_rate})",
        f"Features: {X_train.shape[1]} ({len(cat_cols)} native categorical, full cardinality -- no top-n/frequency reduction)",
        f"Train: {len(y_train):,} rows",
        "",
        _format_block("OVERALL", overall),
    ]

    by_claim_type = {}
    val_types = pd.read_parquet(VAL_PATH, columns=["claim_type_carrier", "claim_type_outpatient", "claim_type_dme"])
    val_types = val_types.loc[y_val.index]
    for ct in _CLAIM_TYPES:
        mask = (val_types[f"claim_type_{ct}"] == 1).to_numpy()
        m = _metrics_block(y_val_arr[mask], p_val[mask])
        by_claim_type[ct] = m
        lines.append(_format_block(ct.upper(), m))

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    lines.append("")
    lines.append("Top 20 features by gain-based importance:")
    for name, val in importances.head(20).items():
        lines.append(f"  {name:40s} {val:.4f}")

    report = "\n".join(lines)
    print("\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = REPORTS_DIR / "xgboost_results.txt"
    out_txt.write_text(report + "\n")

    out_json = REPORTS_DIR / "xgboost_fit.json"
    out_json.write_text(json.dumps({
        "n_estimators_allowed": args.n_estimators,
        "best_iteration": int(best_iter),
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_features": X_train.shape[1],
        "n_categorical_features": len(cat_cols),
        "train_nobs": int(len(y_train)),
        "val_metrics_overall": overall,
        "val_metrics_by_claim_type": by_claim_type,
        "top_20_feature_importance": {k: float(v) for k, v in importances.head(20).items()},
    }, indent=2))
    print(f"\nWritten to {out_txt} and {out_json}")


if __name__ == "__main__":
    main()
