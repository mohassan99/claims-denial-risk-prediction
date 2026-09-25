"""
Phase 2 -- fit the baseline mixed logistic model (task queue item 1,
CLAUDE.md), the model Chow Stage 2 actually selected
(reports/chow_stage2_results__firth.txt, 2026-09-24 corrected-label run):
5 of 11 genuinely-shared covariates kept as claim-type interaction terms
(provider_state, HCPCS_CD, PRNCPAL_DGNS_CD, prvdr_num_freq,
CARR_CLM_CASH_DDCTBL_APLD_AMT), the other 6 pooled to one coefficient each
(build_chow_design_matrix.build_mixed_design_matrix -- see its docstring
for exactly how; MIXED_INTERACT_NUMERIC/MIXED_POOL_NUMERIC there must be
kept in sync with Stage 2's variable list if it's ever rerun on a changed
label or feature set).

TRAIN/VAL COLUMN CONSISTENCY. This is the first script in this project to
need two splits encoded into the SAME feature space -- the Chow test only
ever needed train. Two risks that would otherwise silently corrupt val
metrics, both fixed here:
  1. Categorical vocabulary drift: top-n HCPCS/diagnosis codes and the
     provider frequency map, if fit fresh on val, would define "top" and
     "frequent" differently than train did -- e.g. a code just outside
     train's top-20 but inside val's would flip from __OTHER__ to its own
     dummy, changing what the model's train-fit coefficients are being
     multiplied against. Fixed: build_chow_design_matrix(train, return_
     encoders=True) fits the categories/frequency map once; the same
     `encoders` dict is passed into the val call, which reuses them
     unchanged (see top_n_encode/frequency_encode docstrings).
  2. Per-split constancy decisions (rank-deficiency fix A: columns constant
     within one claim type are dropped) could still differ between splits
     even with the vocabulary fixed -- e.g. a field constant in train's DME
     rows by chance but not val's. Fixed the same way: train's
     `determined_dropped` list is carried in `encoders` and reused on val
     rather than recomputed (build_chow_design_matrix's encoders path).
  3. Belt and suspenders: after both matrices are built, val's X is
     reindexed to train's exact column LIST (fill_value=0) before
     prediction -- so even a residual per-split interaction-skip
     disagreement (build_mixed_design_matrix's docstring) can't produce a
     column mismatch silently.

FITTING. Standard MLE via chunked_logit.ChunkedLogit (penalty_weight=0) --
this design matrix has none of the Chow test's structural separation (that
was a claim-type x HCPCS/state cross, gone since the label fix; see
TARGET_DEFINITION.md's 2026-09-24 addendum), so a Firth penalty isn't
needed by default. Falls back to Firth automatically (--method is still a
flag, in case a future rerun disagrees) only if standard MLE doesn't
converge, with a clear note that coefficients are then bias-reduced, not
raw MLE.

METRICS on val, never only overall (this project's standing rule): PR-AUC
(primary -- positive rate 14.9%, so ROC-AUC alone is optimistic), ROC-AUC,
Brier score, and the same breakdown per claim type, since a pooled metric
could hide one claim type the model handles badly.

Run (repo root, venv active):
    python src/fit_baseline_model.py
    python src/fit_baseline_model.py --sample-frac 0.02   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from build_chow_design_matrix import PROCESSED_DIR, build_chow_design_matrix, build_mixed_design_matrix
from chunked_logit import ChunkedLogit
from fit_chow_test import LABEL_COL, _columns_to_load, _prepare_xy, _stratified_sample_streaming

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
TRAIN_PATH = PROCESSED_DIR / "train_model.parquet"
VAL_PATH = PROCESSED_DIR / "val_model.parquet"

_CLAIM_TYPES = ("carrier", "outpatient", "dme")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the Phase 2 baseline mixed logistic model.")
    p.add_argument("--method", choices=["standard", "firth"], default="standard")
    p.add_argument("--sample-frac", type=float, default=None, help="Stratified subsample of TRAIN only (smoke test). val is always used in full.")
    p.add_argument("--stream-batch-size", type=int, default=50_000)
    p.add_argument("--max-iter", type=int, default=300)
    return p.parse_args()


def _load(path: Path, columns: list[str], sample_frac: float | None, batch_size: int) -> pd.DataFrame:
    if sample_frac is None:
        print(f"Loading {path.name} (full; {len(columns)} columns)...")
        return pd.read_parquet(path, columns=columns)
    print(f"Streaming a --sample-frac {sample_frac} stratified subsample of {path.name}...")
    return _stratified_sample_streaming(path, sample_frac, LABEL_COL, batch_size, columns=columns)


def _predict(X: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return expit(X.to_numpy(dtype=np.float64) @ beta)


def _metrics_block(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
    }


def _format_block(label: str, m: dict) -> str:
    return (
        f"{label:14s} n={m['n']:>8,}  pos_rate={m['positive_rate']:.4f}  "
        f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
        f"Brier={m['brier']:.4f}  mean_p={m['mean_predicted']:.4f}"
    )


def main() -> None:
    args = parse_args()
    columns = _columns_to_load(TRAIN_PATH)

    print("=== Building train design matrix ===")
    train_raw = _load(TRAIN_PATH, columns, args.sample_frac, args.stream_batch_size)
    train_intermediate, encoders = build_chow_design_matrix(train_raw, return_encoders=True)
    del train_raw
    gc.collect()
    train_mixed = build_mixed_design_matrix(train_intermediate)
    del train_intermediate
    gc.collect()
    y_train, X_train = _prepare_xy(train_mixed)
    del train_mixed
    gc.collect()
    print(f"train: {X_train.shape[0]:,} rows x {X_train.shape[1]} columns, {y_train.mean():.2%} positive")

    colnames = list(X_train.columns)

    print(f"\n=== Fitting ({args.method}) ===")
    y_arr = y_train.to_numpy(dtype=np.float64)
    X_arr = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64))
    # MEMORY (2026-09-25, following fit_chow_test.py's discipline): X_train
    # is dropped here, BEFORE val is built at all, not after. The first
    # full-size run held X_train (DataFrame), X_val (DataFrame), AND X_arr
    # (a fresh float64 copy of X_train, ~2.5 GB) alive at once and was
    # OOM-killed by the container's memory cgroup (peak anon-rss 6.1 GB).
    # Building val strictly after the fit, once X_arr/y_arr are also freed,
    # means the two large allocations (train's float64 array during/after
    # fitting; val's DataFrame+array during prediction) never overlap.
    del X_train
    gc.collect()

    penalty_weight = 0.5 if args.method == "firth" else 0.0
    model = ChunkedLogit(X_arr, y_arr, penalty_weight=penalty_weight)
    fit = model.fit(max_iter=args.max_iter)
    method_used = args.method
    if not fit.converged and args.method == "standard":
        print("    standard MLE did not converge -- retrying with Firth's penalized MLE...")
        model = ChunkedLogit(X_arr, y_arr, penalty_weight=0.5)
        fit = model.fit(max_iter=args.max_iter)
        method_used = "firth (fallback after standard MLE did not converge)"
    if not fit.converged:
        raise RuntimeError(f"Baseline fit did not converge (method={method_used}) -- not evaluating an unconverged fit.")
    del X_arr, y_arr, model
    gc.collect()
    print(f"    converged in {fit.n_iter} iterations, log-likelihood {fit.llf_unpenalized:.4f}")

    print("\n=== Building val design matrix (train's encoders reused) ===")
    val_raw = pd.read_parquet(VAL_PATH, columns=columns)
    val_intermediate = build_chow_design_matrix(val_raw, encoders=encoders)
    del val_raw
    gc.collect()
    val_mixed = build_mixed_design_matrix(val_intermediate)
    del val_intermediate
    gc.collect()
    y_val, X_val = _prepare_xy(val_mixed)
    del val_mixed
    gc.collect()

    extra_in_val = set(X_val.columns) - set(colnames)
    missing_in_val = set(colnames) - set(X_val.columns)
    if extra_in_val or missing_in_val:
        print(
            f"    [note] aligning val to train's {len(colnames)} columns: "
            f"{len(missing_in_val)} missing in val (added as all-0), "
            f"{len(extra_in_val)} extra in val (dropped): "
            f"missing={sorted(missing_in_val)[:10]}{'...' if len(missing_in_val) > 10 else ''}, "
            f"extra={sorted(extra_in_val)[:10]}{'...' if len(extra_in_val) > 10 else ''}"
        )
    X_val = X_val.reindex(columns=colnames, fill_value=0)
    print(f"val:   {X_val.shape[0]:,} rows x {X_val.shape[1]} columns, {y_val.mean():.2%} positive")

    print("\n=== Evaluating on val ===")
    p_val = _predict(X_val, fit.beta)
    y_val_arr = y_val.to_numpy(dtype=np.float64)
    del X_val
    gc.collect()

    overall = _metrics_block(y_val_arr, p_val)
    lines = [
        "Phase 2 baseline mixed logistic model -- val set results",
        "=" * 62,
        f"Method: {method_used}",
        f"Design: {len(colnames)} columns (5 Chow-selected variables interacted by claim type, "
        "6 pooled, all claim-type-exclusive covariates unaffected)",
        f"Train: {int(fit.nobs):,} rows, log-likelihood {fit.llf_unpenalized:.4f}, "
        f"{fit.n_iter} Newton iterations",
        "",
        _format_block("OVERALL", overall),
    ]

    by_claim_type = {}
    val_mixed_types = pd.read_parquet(VAL_PATH, columns=["claim_type_carrier", "claim_type_outpatient", "claim_type_dme"])
    val_mixed_types = val_mixed_types.loc[y_val.index]
    for ct in _CLAIM_TYPES:
        mask = (val_mixed_types[f"claim_type_{ct}"] == 1).to_numpy()
        m = _metrics_block(y_val_arr[mask], p_val[mask])
        by_claim_type[ct] = m
        lines.append(_format_block(ct.upper(), m))

    report = "\n".join(lines)
    print("\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = REPORTS_DIR / f"baseline_model_results__{method_used.split()[0]}.txt"
    out_txt.write_text(report + "\n")

    out_json = REPORTS_DIR / f"baseline_model_fit__{method_used.split()[0]}.json"
    out_json.write_text(json.dumps({
        "method": method_used,
        "colnames": colnames,
        "beta": fit.beta.tolist(),
        "train_llf_unpenalized": fit.llf_unpenalized,
        "train_nobs": int(fit.nobs),
        "n_iter": fit.n_iter,
        "val_metrics_overall": overall,
        "val_metrics_by_claim_type": by_claim_type,
        "encoders": {
            "state_columns": encoders["state_columns"],
            "hcpcs_categories": encoders["hcpcs_categories"],
            "hcpcs_columns": encoders["hcpcs_columns"],
            "dgns_categories": encoders["dgns_categories"],
            "dgns_columns": encoders["dgns_columns"],
            "determined_dropped": encoders["determined_dropped"],
        },
    }, indent=2))
    print(f"\nWritten to {out_txt} and {out_json}")


if __name__ == "__main__":
    main()
