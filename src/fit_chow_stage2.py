"""
Phase 2 -- Chow test STAGE 2: per-variable likelihood-ratio tests, run
because Stage 1 (src/fit_chow_test.py) rejected H0 on the full train set
(Firth, 2026-09-24: LR 4,732.5 on 158 df). FEATURE_ENGINEERING.md Section 3
calls for this step: a rejection says at least one shared covariate's
effect differs by claim type. It does NOT say which ones.

Stage 2 uses the same machinery as Stage 1 and adds no new
design-matrix logic. It builds the reparameterized full design
U' = [restricted columns, Z] (fit_chow_test._build_reparameterized_design,
span-verified) and fits the unrestricted model once. Then, for each
variable v, it refits with ONLY v's tested coefficients (its Z columns)
held at 0, and computes
    LR_v = 2 * (ll_full - ll_{Z_v = 0})   ~   chi2(|Z_v|)
With --method firth, each constrained fit keeps the full model's penalty,
so this is the Heinze & Schemper penalized LR test (the same form
firthmodels' own .lrt() uses, generalized from one coefficient to a block).

WHAT COUNTS AS A "VARIABLE". The test unit is the original variable, not
each one-hot dummy. The three genuinely-shared categorical groups --
provider_state (state_*), HCPCS_CD (hcpcs_*) and PRNCPAL_DGNS_CD (dgns_*)
-- are tested as whole variables, jointly over all their dummies' tested
coefficients. Every numeric shared covariate is tested on its own. This
matches Section 3's construction of the final mixed model, where a
variable either keeps one pooled representation or is replaced by its
claim-type interactions. It is an analyst choice, and a per-dummy
breakdown within a rejected categorical variable is a possible follow-up.

MULTIPLE TESTING. Raw p-values are reported alongside Holm-adjusted ones
(family = the variables tested in this run). Holm controls the
family-wise error rate without assuming the tests are independent (they
aren't: all share one fitted full model).

Resumable: results are appended to the output CSV one variable at a time,
and a rerun with the same flags skips variables already done. At full
size each constrained fit is warm-started from the full-model
coefficients (with v's block zeroed), so it takes a handful of Newton
iterations rather than a cold start's ~75.

Run (repo root, venv active):
    python src/fit_chow_stage2.py --method firth --sample-frac 0.02   # smoke test
    python src/fit_chow_stage2.py --method firth                      # full
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from build_chow_design_matrix import add_claim_type_interactions, build_restricted_design_matrix
from chunked_logit import ChunkedLogit
from fit_chow_test import (
    REPORTS_DIR,
    _build_intermediate,
    _build_reparameterized_design,
    _prepare_xy,
    _require_full_rank,
)

_CATEGORICAL_GROUPS = {"state_": "provider_state", "hcpcs_": "HCPCS_CD", "dgns_": "PRNCPAL_DGNS_CD"}


def variable_of(z_col: str) -> str:
    base = z_col.rsplit("__x__", 1)[0]
    for prefix, var in _CATEGORICAL_GROUPS.items():
        if base.startswith(prefix):
            return var
    return base


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chow test Stage 2: per-variable LR tests.")
    p.add_argument("--method", choices=["standard", "firth"], default="firth")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--stream-batch-size", type=int, default=50_000)
    p.add_argument("--exclude-separating-codes", action="store_true")
    p.add_argument("--max-iter", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.method + ("_excl" if args.exclude_separating_codes else "")
    if args.sample_frac is not None:
        suffix += f"_frac{args.sample_frac:g}"
    out_csv = REPORTS_DIR / f"chow_stage2_results__{suffix}.csv"
    out_txt = REPORTS_DIR / f"chow_stage2_results__{suffix}.txt"
    full_json = REPORTS_DIR / f"chow_stage2_fullfit__{suffix}.json"

    intermediate = _build_intermediate(args)
    y, X_R = _prepare_xy(build_restricted_design_matrix(intermediate), args.exclude_separating_codes)
    _require_full_rank(X_R, "restricted")
    unrestricted_df, _, _ = add_claim_type_interactions(intermediate)
    del intermediate
    gc.collect()
    y_u, X_U = _prepare_xy(unrestricted_df, args.exclude_separating_codes)
    del unrestricted_df
    gc.collect()
    if not y_u.equals(y):
        raise ValueError("Restricted and unrestricted matrices do not cover identical rows.")
    _require_full_rank(X_U, "unrestricted")
    A, colnames, z_mask, _ = _build_reparameterized_design(X_R, X_U)
    del X_R, X_U, y_u
    gc.collect()

    z_idx = np.flatnonzero(z_mask)
    groups: dict[str, list[int]] = defaultdict(list)
    for j in z_idx:
        groups[variable_of(colnames[j])].append(int(j))
    print(f"\n{len(groups)} variable(s) to test, covering all {len(z_idx)} tested coefficients")

    model = ChunkedLogit(A, y.to_numpy(dtype=np.float64), penalty_weight=0.5 if args.method == "firth" else 0.0)
    del y

    if full_json.exists():
        cached = json.loads(full_json.read_text())
        if cached["colnames"] != colnames:
            raise ValueError(f"{full_json} was fit on different columns -- delete it and rerun.")
        full_llf = cached["llf"]
        beta_full = np.asarray(cached["beta"])
        print(f"Loaded cached full fit (llf {full_llf:.4f}) from {full_json}")
    else:
        print("\nFitting the full (unrestricted) model once...")
        full = model.fit(max_iter=args.max_iter)
        if not full.converged:
            raise RuntimeError("Full model did not converge -- cannot run Stage 2.")
        full_llf, beta_full = full.llf, full.beta
        full_json.write_text(json.dumps({"llf": full_llf, "n_iter": full.n_iter, "colnames": colnames, "beta": beta_full.tolist()}))

    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["variable"])
    done_vars = set(done["variable"])
    for var in sorted(groups, key=lambda v: (-len(groups[v]), v)):
        if var in done_vars:
            continue
        idx = groups[var]
        fixed = np.zeros(len(colnames), dtype=bool)
        fixed[idx] = True
        init = beta_full.copy()
        init[idx] = 0.0
        print(f"\n  [{var}] {len(idx)} tested coefficient(s) held at 0")
        res = model.fit(fixed_zero=fixed, beta_init=init, max_iter=args.max_iter, verbose=False)
        lr = 2.0 * (full_llf - res.llf)
        row = {
            "variable": var,
            "df": len(idx),
            "llf_constrained": res.llf,
            "LR": max(lr, 0.0),
            "LR_raw": lr,
            "p_value": float(stats.chi2.sf(max(lr, 0.0), len(idx))) if res.converged else np.nan,
            "converged": res.converged,
            "n_iter": res.n_iter,
            "tested_terms": ";".join(colnames[j] for j in idx),
        }
        print(f"    converged {res.converged} ({res.n_iter} it)  LR {lr:.3f}  df {len(idx)}  p {row['p_value']:.4g}")
        if lr < -1e-6:
            raise RuntimeError(f"Negative LR for {var} -- a fit is not at its optimum.")
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)

    res_df = pd.read_csv(out_csv)
    ok = res_df["converged"].astype(bool)
    res_df["p_holm"] = np.nan
    res_df.loc[ok, "p_holm"] = holm(res_df.loc[ok, "p_value"].to_numpy())
    res_df["reject_holm_0.05"] = res_df["p_holm"] < 0.05
    res_df = res_df.sort_values(["p_value", "LR"], ascending=[True, False])
    res_df.to_csv(out_csv, index=False)

    n_rej = int(res_df["reject_holm_0.05"].sum())
    lines = [
        "Chow test STAGE 2 -- per-variable claim-type homogeneity tests",
        "=" * 64,
        f"Method: {args.method}{' (penalized LR, full-model penalty)' if args.method == 'firth' else ''}; "
        f"--sample-frac {args.sample_frac if args.sample_frac is not None else 'full'}; "
        f"separating-code exclusion {'ON' if args.exclude_separating_codes else 'OFF'}",
        f"Full-model log-likelihood: {full_llf:.4f}",
        f"Variables tested: {len(res_df)} (non-converged: {int((~ok).sum())})",
        f"Heterogeneous by claim type (Holm-adjusted p < 0.05): {n_rej}",
        f"Pooling adequate (Holm-adjusted p >= 0.05): {int(ok.sum()) - n_rej}",
        "",
        f"{'variable':42s} {'df':>4s} {'LR':>12s} {'p':>11s} {'p_holm':>11s}  verdict",
    ]
    for _, r in res_df.iterrows():
        verdict = "INTERACT" if r["reject_holm_0.05"] else "pool"
        lines.append(f"{r['variable']:42s} {int(r['df']):>4d} {r['LR']:>12.3f} {r['p_value']:>11.4g} {r['p_holm']:>11.4g}  {verdict}")
    report = "\n".join(lines)
    print("\n" + report)
    out_txt.write_text(report + "\n")
    print(f"\nWritten to {out_txt} and {out_csv}")


if __name__ == "__main__":
    main()
