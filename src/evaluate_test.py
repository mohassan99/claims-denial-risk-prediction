"""Phase 5: decision-focused evaluation of model v2, validation first, test once.

Two stages, run in order:

    python src/evaluate_test.py --stage val    # look, decide, freeze
    python src/evaluate_test.py --stage test   # score the held-out test set ONCE

Why two stages. Every choice (raw vs recalibrated scores, which cost scenarios
are the headline, which k values are reported) is made on validation data and
written to reports/phase5/protocol.json. The test stage reads that file, refuses
to run without it, and refuses to run a second time once
reports/phase5/test_results.json exists. That file records when the test set
was used and the exact model and data hashes, so "test used once" is checkable.

Unit of analysis. One row = one claim line, the same unit the label, the model
and every earlier metric use. A real review is per claim, so line-level counts
overstate how many separate reviews a flag list implies; the report says so.

Dollars at stake (A_i). Native Synthea payment on the line: LINE_NCH_PMT_AMT
for carrier and DME, REV_CNTR_PMT_AMT_AMT for outpatient. These are never
touched by the label build (only CLM_PMT_AMT is, and it is dropped), checked
2026-09-30 by joining val rows back to combined_claims_raw.parquet on _row_id:
100% identical. They are not model inputs for this purpose; they only weight
the decision.

Costs (all hypothetical; see the "Phase 5 Cost Assumptions" doc):
    r   = w * b * m / 60          review cost, w = $37.51/h (BLS, May 2025)
    L_i = A_i * (1 - rho) + c * rho * A_i
    review line i when p_i * L_i > r   (equivalently p_i > r / L_i)
Headline: b = 1.3, m = 20 min, c = 0.125, rho reported at BOTH 0.24 and 0.55.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from fit_xgboost import (
    MODEL_META_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    _apply_categories,
    _columns_to_load,
    _split_xy,
)

TRAIN_PATH = PROCESSED_DIR / "train_model.parquet"
SPLIT_PATHS = {"val": PROCESSED_DIR / "val_model.parquet", "test": PROCESSED_DIR / "test_model.parquet"}
OUT_DIR = REPORTS_DIR / "phase5"
PROTOCOL_PATH = OUT_DIR / "protocol.json"
TEST_RESULTS_PATH = OUT_DIR / "test_results.json"

EXPECTED_MODEL_HASH = "11f409f72fc4c45e"
CLAIM_TYPES = ("carrier", "outpatient", "dme")
METRICS_TOL = 1e-6

# Cost assumptions (hypothetical). Headline first in each list.
WAGE = 37.51
HEADLINE = {"b": 1.3, "m": 20.0, "c": 0.125}
RHO_SCENARIOS = (0.24, 0.55)
SWEEP = {"b": (1.3, 1.43), "m": (10.0, 20.0, 30.0), "c": (0.09, 0.125, 0.20), "rho": RHO_SCENARIOS}
K_FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)
DECISION_CURVE_T = tuple(np.round(np.arange(0.02, 0.62, 0.02), 2))
N_BINS = 10
# Pre-stated rule (written before looking at val calibration): recalibrate a
# claim type only if isotonic recalibration, fit on the first half of val by
# date and scored on the second half, lowers that half's Brier score by more
# than 1% relative. Otherwise raw scores are kept (simpler, nothing to freeze).
RECAL_MIN_REL_GAIN = 0.01


def review_cost(b: float, m: float) -> float:
    return WAGE * b * m / 60.0


def loss_if_missed(amount: np.ndarray, rho: float, c: float) -> np.ndarray:
    return amount * (1.0 - rho) + c * rho * amount


def model_hash() -> str:
    h = hashlib.sha256()
    h.update(MODEL_PATH.read_bytes())
    h.update(MODEL_META_PATH.read_bytes())
    return h.hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- scoring

def score_split(split: str) -> pd.DataFrame:
    """Score one split with model v2. Returns one row per line: _row_id,
    claim_type, date, y, p, amount."""
    meta = json.loads(MODEL_META_PATH.read_text())
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    path = SPLIT_PATHS[split]
    columns = _columns_to_load(TRAIN_PATH)
    raw = pd.read_parquet(path, columns=columns)
    y, X = _split_xy(raw)
    del raw
    X = _apply_categories(X, meta["cat_cols"], meta["categories_by_col"]).reindex(columns=meta["colnames"])
    p = model.predict_proba(X)[:, 1]
    del X

    side = pd.read_parquet(
        path,
        columns=["_row_id", "CLM_FROM_DT", "claim_type_carrier", "claim_type_outpatient", "claim_type_dme",
                 "LINE_NCH_PMT_AMT", "REV_CNTR_PMT_AMT_AMT"],
    )
    ct = np.select(
        [side["claim_type_carrier"] == 1, side["claim_type_outpatient"] == 1, side["claim_type_dme"] == 1],
        list(CLAIM_TYPES), default="unknown",
    )
    if (ct == "unknown").any():
        raise ValueError("rows with no claim type")
    amount = np.where(ct == "outpatient", side["REV_CNTR_PMT_AMT_AMT"], side["LINE_NCH_PMT_AMT"]).astype(float)
    if np.isnan(amount).any():
        raise ValueError("missing line payment amount; the dollar weighting assumes it is always present")
    return pd.DataFrame({
        "_row_id": side["_row_id"].to_numpy(),
        "claim_type": ct,
        "date": pd.to_datetime(side["CLM_FROM_DT"], format="%d-%b-%Y"),
        "y": y.to_numpy(dtype=np.int8),
        "p_raw": p,
        "amount": amount,
    })


def verify_val_reproduces(df: pd.DataFrame) -> None:
    """Model v2 must reproduce its recorded val metrics before anything else."""
    reported = json.loads((REPORTS_DIR / "xgboost_fit.json").read_text())
    checks = [("overall", df, reported["val_metrics_overall"])] + [
        (ct, df[df.claim_type == ct], reported["val_metrics_by_claim_type"][ct]) for ct in CLAIM_TYPES
    ]
    for name, g, rep in checks:
        got = {"pr_auc": average_precision_score(g.y, g.p_raw), "roc_auc": roc_auc_score(g.y, g.p_raw),
               "brier": brier_score_loss(g.y, g.p_raw)}
        for k, v in got.items():
            if abs(v - rep[k]) > METRICS_TOL:
                raise RuntimeError(f"val {name} {k}: got {v!r}, recorded {rep[k]!r}. Wrong model or data.")
    print("verified: model v2 reproduces recorded val PR-AUC/ROC-AUC/Brier (overall and per claim type) to 1e-6")


# ---------------------------------------------------------------- metrics

def calibration_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Equal-width score bins: mean score vs observed rate, with counts."""
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        mask = idx == b
        n = int(mask.sum())
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "n": n,
                     "mean_score": float(p[mask].mean()) if n else None,
                     "observed_rate": float(y[mask].mean()) if n else None})
    return rows


def ece(y: np.ndarray, p: np.ndarray) -> float:
    """Expected calibration error: count-weighted mean |mean score - observed rate| over the bins."""
    tab = calibration_table(y, p)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["mean_score"] - r["observed_rate"]) for r in tab if r["n"]))


def topk_table(y: np.ndarray, p: np.ndarray, amount: np.ndarray, value: np.ndarray) -> list[dict]:
    """precision@k, lift, recall and dollars captured, for two rankings:
    by score p alone, and by expected net value p*L - r."""
    n, base = len(y), y.mean()
    pos_dollars = float((amount * y).sum())
    out = []
    for frac in K_FRACS:
        k = max(1, int(round(frac * n)))
        row = {"k_frac": frac, "k": k}
        for name, key in (("by_score", p), ("by_value", value)):
            top = np.argsort(-key, kind="stable")[:k]
            prec = float(y[top].mean())
            row[name] = {
                "precision": prec,
                "lift": prec / base if base > 0 else None,
                "recall": float(y[top].sum() / y.sum()) if y.sum() else None,
                "dollars_captured": float((amount[top] * y[top]).sum()),
                "dollars_captured_share": float((amount[top] * y[top]).sum() / pos_dollars) if pos_dollars else None,
            }
        out.append(row)
    return out


def decision_outcome(y: np.ndarray, p: np.ndarray, amount: np.ndarray, b: float, m: float, c: float, rho: float) -> dict:
    """Apply the per-line rule p*L > r and count what it costs and saves.
    Net savings are relative to reviewing nothing: each flagged true denial
    saves L_i, each flag costs r."""
    r = review_cost(b, m)
    L = loss_if_missed(amount, rho, c)
    flag = p * L > r
    n_flag = int(flag.sum())
    tp = int((flag & (y == 1)).sum())
    saved = float((L * y)[flag].sum())
    spent = r * n_flag
    oracle = float(np.clip(L[y == 1] - r, 0, None).sum())
    review_all = float((L * y).sum() - r * len(y))
    return {
        "r": r, "rho": rho, "b": b, "m": m, "c": c,
        "n": int(len(y)), "n_flagged": n_flag, "flag_rate": n_flag / len(y),
        "precision": tp / n_flag if n_flag else None,
        "recall": tp / int(y.sum()) if y.sum() else None,
        "loss_avoided": saved, "review_cost": spent, "net_savings": saved - spent,
        "net_savings_per_1000_lines": (saved - spent) / len(y) * 1000,
        "oracle_net_savings": oracle,
        "share_of_oracle": (saved - spent) / oracle if oracle > 0 else None,
        "review_all_net_savings": review_all,
    }


def decision_curve(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Net benefit = TP/n - FP/n * t/(1-t) for the model, review-all and review-none (0)."""
    n, prev = len(y), y.mean()
    rows = []
    for t in DECISION_CURVE_T:
        flag = p > t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        w = t / (1 - t)
        rows.append({"t": float(t), "model": tp / n - fp / n * w, "review_all": prev - (1 - prev) * w, "review_none": 0.0})
    return rows


def block(df: pd.DataFrame, pcol: str) -> dict:
    """Every metric for one group of lines."""
    y, p, a = df.y.to_numpy(), df[pcol].to_numpy(), df.amount.to_numpy()
    r = review_cost(HEADLINE["b"], HEADLINE["m"])
    out = {
        "n": int(len(y)), "positive_rate": float(y.mean()), "mean_score": float(p.mean()),
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
        "calibration": calibration_table(y, p),
        "dollars_total": float(a.sum()), "dollars_on_denied": float((a * y).sum()),
        "decision_curve": decision_curve(y, p),
        "scenarios": {},
        "topk": {},
    }
    for rho in RHO_SCENARIOS:
        key = f"rho_{rho}"
        out["scenarios"][key] = decision_outcome(y, p, a, rho=rho, **HEADLINE)
        value = p * loss_if_missed(a, rho, HEADLINE["c"]) - r
        out["topk"][key] = topk_table(y, p, a, value)
    out["sweep"] = [decision_outcome(y, p, a, b=b, m=m, c=c, rho=rho)
                    for b, m, c, rho in product(SWEEP["b"], SWEEP["m"], SWEEP["c"], SWEEP["rho"])]
    return out


def all_blocks(df: pd.DataFrame, pcol: str) -> dict:
    return {"overall": block(df, pcol), **{ct: block(df[df.claim_type == ct], pcol) for ct in CLAIM_TYPES}}


# ---------------------------------------------------------------- stages

def recalibration_decision(val: pd.DataFrame) -> dict:
    """Apply the pre-stated rule per claim type (see RECAL_MIN_REL_GAIN)."""
    cut = val.date.sort_values().iloc[len(val) // 2]
    decisions = {}
    for ct in CLAIM_TYPES:
        g = val[val.claim_type == ct]
        fit, hold = g[g.date < cut], g[g.date >= cut]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit.p_raw, fit.y)
        raw_b = brier_score_loss(hold.y, hold.p_raw)
        cal_b = brier_score_loss(hold.y, iso.predict(hold.p_raw))
        gain = (raw_b - cal_b) / raw_b
        decisions[ct] = {"split_date": str(cut.date()), "n_fit": int(len(fit)), "n_holdout": int(len(hold)),
                         "holdout_brier_raw": raw_b, "holdout_brier_isotonic": cal_b,
                         "relative_gain": gain, "recalibrate": bool(gain > RECAL_MIN_REL_GAIN)}
    return decisions


def apply_recalibration(df: pd.DataFrame, val: pd.DataFrame, decisions: dict) -> pd.Series:
    """Isotonic map fit on ALL of val for the claim types the rule chose; raw otherwise."""
    p = df.p_raw.copy()
    for ct, d in decisions.items():
        if d["recalibrate"]:
            g = val[val.claim_type == ct]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(g.p_raw, g.y)
            mask = df.claim_type == ct
            p[mask] = iso.predict(df.loc[mask, "p_raw"])
    return p


def run_val() -> None:
    if TEST_RESULTS_PATH.exists():
        raise SystemExit(f"{TEST_RESULTS_PATH} exists: the test set has been used. Val choices are frozen; not re-running.")
    h = model_hash()
    if h != EXPECTED_MODEL_HASH:
        raise SystemExit(f"model hash {h} != expected {EXPECTED_MODEL_HASH}")
    val = score_split("val")
    verify_val_reproduces(val)
    decisions = recalibration_decision(val)
    val["p"] = apply_recalibration(val, val, decisions)
    results = {"split": "val", "model_hash": h, "data_hash": file_hash(SPLIT_PATHS["val"]),
               "recalibration_rule": f"recalibrate a claim type only if isotonic (fit on val first half by date) "
                                     f"cuts second-half Brier by > {RECAL_MIN_REL_GAIN:.0%} relative",
               "recalibration": decisions,
               "raw": all_blocks(val, "p_raw")}
    if any(d["recalibrate"] for d in decisions.values()):
        results["recalibrated"] = all_blocks(val, "p")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val_results.json").write_text(json.dumps(results, indent=2))
    val.to_parquet(OUT_DIR / "val_predictions.parquet", index=False)

    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_hash": h,
        "scores": {ct: ("isotonic fit on all of val" if d["recalibrate"] else "raw") for ct, d in decisions.items()},
        "rule": "flag line i when p_i * L_i > r; L_i = A_i*(1-rho) + c*rho*A_i; r = 37.51*b*m/60",
        "headline": {**HEADLINE, "rho_scenarios": list(RHO_SCENARIOS), "r": review_cost(HEADLINE["b"], HEADLINE["m"])},
        "sweep": {k: list(v) for k, v in SWEEP.items()},
        "k_fracs": list(K_FRACS),
        "decision_curve_t": list(DECISION_CURVE_T),
        "note": "Chosen on validation only. The test stage reads this file and changes nothing in it.",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2))
    print(f"wrote {OUT_DIR / 'val_results.json'} and {PROTOCOL_PATH}")
    print(json.dumps({ct: {k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()} for ct, d in decisions.items()}, indent=1))


def run_test() -> None:
    if not PROTOCOL_PATH.exists():
        raise SystemExit("no protocol.json: run --stage val first and commit it before touching test")
    if TEST_RESULTS_PATH.exists():
        raise SystemExit(f"{TEST_RESULTS_PATH} exists: the test set was already used once. Refusing to re-run.")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    h = model_hash()
    if h != protocol["model_hash"]:
        raise SystemExit(f"model hash {h} != protocol's {protocol['model_hash']}")
    val = pd.read_parquet(OUT_DIR / "val_predictions.parquet")
    decisions = {ct: {"recalibrate": protocol["scores"][ct] != "raw"} for ct in CLAIM_TYPES}
    test = score_split("test")
    test["p"] = apply_recalibration(test, val, decisions)
    results = {
        "split": "test",
        "used_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_hash": h, "data_hash": file_hash(SPLIT_PATHS["test"]),
        "protocol": protocol,
        "date_range": [str(test.date.min().date()), str(test.date.max().date())],
        "scored": all_blocks(test, "p"),
    }
    if any(d["recalibrate"] for d in decisions.values()):
        results["raw_for_comparison"] = all_blocks(test, "p_raw")
    test.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    TEST_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {TEST_RESULTS_PATH} -- the test set is now used; this stage will refuse to run again")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    args = ap.parse_args()
    run_val() if args.stage == "val" else run_test()


if __name__ == "__main__":
    main()
