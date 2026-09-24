"""
Turn check_rank.py's rank-deficiency findings into EXPLICIT linear
equations -- "column A = 1*B - 1*C + 0.8*D" -- rather than SVD null-space
vectors (which show co-involvement but aren't unique or sparse when
several dependencies share a subspace) or a bare QR drop list (which says
WHAT to drop, not WHY).

Three passes, on the same design matrix check_rank.py builds:

  1. Constant columns (one distinct value everywhere) -- already known
     from check_rank.py, listed again for completeness.

  2. Claim-type-indicator columns: any column that is CONSTANT WITHIN
     EACH claim type (possibly a different constant per claim type) is
     exactly  c_carrier*claim_type_carrier + c_outpatient*claim_type_
     outpatient + c_dme*claim_type_dme  -- an exact linear combination of
     the 3 protected dummies. check_rank.py's _find_constant_columns()
     only caught columns constant across the WHOLE dataset, so it missed
     this more general case. Hypotheses this tests directly (2026-09-24):
       - all 5 NPI presence flags are claim-type indicators in disguise
         (has_performing == carrier; has_referring == carrier + dme;
         has_attending == has_operating == has_rendering == outpatient).
         NOTE: this reinterprets check_rank.py's "67.8%, refuted" NPI-sum
         check -- that test was wrongly designed: if all three outpatient
         flags equal claim_type_outpatient, their SUM is 3 on outpatient
         rows, so "sum == claim_type_outpatient" can only hold on
         non-outpatient rows. 67.8% is then just the non-outpatient
         share, which is evidence FOR the hypothesis, not against it.
       - CLAIM_QUERY_CODE / REV_CNTR_UNIT_CNT / NCH_PROFNL_CMPNT_CHRG_AMT
         (x outpatient) being a fixed nonzero value on every outpatient
         claim.

  3. Everything else: after removing passes 1-2, take check_rank.py's
     protected QR drop list (never drops a claim_type dummy), then for
     each dropped column run orthogonal matching pursuit (greedy sparse
     regression) against the KEPT columns until the relative residual is
     ~0. This yields the sparsest equation it can find, e.g. an exact
     duplicate shows up as a single term with coefficient 1. Expected
     (from the SVD patterns): DME primary-payer/allowed-amount duplicates,
     outpatient claim-level vs revenue-center payment arithmetic,
     carr_num_freq as a function of the state dummies (Medicare carriers
     are assigned by state), and -- unrestricted matrix only -- an HCPCS
     dummy trap WITHIN claim type (drop_first's reference category never
     occurs in carrier or DME, so the remaining dummies x that claim type
     sum to exactly the claim_type dummy).

OMP's answer is exact but not guaranteed to be the UNIQUE sparsest one --
e.g. if B == C, "A = B" and "A = C" are both correct and it picks one.
Each equation is verified numerically (relative residual printed).

Writes a copy of the output to reports/rank_deficiency_explained__<matrix>
[_excl].txt for FEATURE_ENGINEERING.md.

Run with (from the repo root, inside the venv):
    python src/explain_dependencies.py
    python src/explain_dependencies.py --matrix unrestricted
    python src/explain_dependencies.py --exclude-separating-codes
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from build_chow_design_matrix import (
    add_claim_type_interactions,
    build_chow_design_matrix,
    build_restricted_design_matrix,
)
from check_rank import _PROTECTED_COLS, _find_constant_columns, _find_redundant_columns_via_qr
from fit_chow_test import REPORTS_DIR, _load_train, _prepare_xy

_CLAIM_TYPES = ("carrier", "outpatient", "dme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
        help="Default 0.2, matching the check_rank.py runs whose deficiency "
        "counts (25 restricted / 35 unrestricted) this explains. Smaller "
        "samples run faster but can add chance coincidences.",
    )
    parser.add_argument("--stream-batch-size", type=int, default=50_000)
    parser.add_argument(
        "--matrix",
        choices=["restricted", "unrestricted", "both"],
        default="both",
    )
    parser.add_argument("--exclude-separating-codes", action="store_true")
    parser.add_argument(
        "--max-terms",
        type=int,
        default=25,
        help="Maximum terms OMP will use per equation before reporting the "
        "dependency as 'not sparse' (default 25).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Relative-residual threshold for calling an equation exact "
        "(default 1e-7).",
    )
    return parser.parse_args()


def _fmt_num(c: float) -> str:
    r = round(c)
    if abs(c - r) < 1e-6:
        return str(int(r))
    return f"{c:.6g}"


def _format_equation(target: str, terms: list[tuple[str, float]]) -> str:
    if not terms:
        return f"{target} = 0"
    parts = []
    for name, c in terms:
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        coef = "" if abs(mag - 1) < 1e-6 else f"{_fmt_num(mag)}*"
        parts.append(f"{sign} {coef}{name}")
    if parts[0].startswith("+ "):
        parts[0] = parts[0][2:]
    elif parts[0].startswith("- "):
        parts[0] = "-" + parts[0][2:]
    return f"{target} = " + " ".join(parts)


def _claim_type_indicator_columns(A: np.ndarray, names: list[str]) -> dict[str, list[tuple[str, float]]]:
    """Columns constant within EACH claim type -> exact combination of the
    3 claim_type dummies. Excludes columns that are zero everywhere (those
    are pass-1 constants) and the protected dummies themselves."""
    idx = {n: i for i, n in enumerate(names)}
    masks = {ct: A[:, idx[f"claim_type_{ct}"]] == 1 for ct in _CLAIM_TYPES if f"claim_type_{ct}" in idx}
    found: dict[str, list[tuple[str, float]]] = {}
    for j, name in enumerate(names):
        if name in _PROTECTED_COLS:
            continue
        col = A[:, j]
        coefs: dict[str, float] = {}
        ok = True
        for ct, m in masks.items():
            vals = col[m]
            if vals.size == 0:
                continue
            if not np.all(vals == vals[0]):
                ok = False
                break
            coefs[ct] = float(vals[0])
        if ok and any(c != 0 for c in coefs.values()):
            found[name] = [(f"claim_type_{ct}", c) for ct, c in coefs.items() if c != 0]
    return found


def _sparse_explain(
    y: np.ndarray, Xk: np.ndarray, col_norms: np.ndarray, max_terms: int, tol: float
) -> tuple[list[int], np.ndarray, float]:
    """Orthogonal matching pursuit: greedily add the kept column most
    correlated with the current residual, refit by least squares on the
    selected set, stop when the relative residual drops below tol."""
    ynorm = np.linalg.norm(y)
    if ynorm == 0:
        return [], np.array([]), 0.0
    r = y.copy()
    sel: list[int] = []
    coef = np.array([])
    rel = 1.0
    for _ in range(max_terms):
        scores = np.abs(Xk.T @ r) / col_norms
        if sel:
            scores[sel] = -np.inf
        sel.append(int(np.argmax(scores)))
        coef, *_ = np.linalg.lstsq(Xk[:, sel], y, rcond=None)
        r = y - Xk[:, sel] @ coef
        rel = float(np.linalg.norm(r) / ynorm)
        if rel < tol:
            break
    return sel, coef, rel


def _explain(matrix_kind: str, args: argparse.Namespace) -> list[str]:
    out: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        out.append(line)

    excl = args.exclude_separating_codes
    emit("=" * 78)
    emit(f"{matrix_kind.upper()} matrix, {'WITH' if excl else 'WITHOUT'} --exclude-separating-codes (--sample-frac {args.sample_frac})")
    emit("=" * 78)

    train = _load_train(args.sample_frac, args.stream_batch_size)
    intermediate = build_chow_design_matrix(train)
    del train
    if matrix_kind == "restricted":
        design_df = build_restricted_design_matrix(intermediate)
    else:
        design_df, _, _ = add_claim_type_interactions(intermediate)
    del intermediate
    _, X = _prepare_xy(design_df, exclude_separating=excl)
    del design_df
    gc.collect()

    names = list(X.columns)
    A = X.to_numpy(dtype=np.float64)
    del X
    gc.collect()
    total_rank = np.linalg.matrix_rank(A)
    emit(f"\n{A.shape[1]} columns, rank {total_rank}, total deficiency {A.shape[1] - total_rank}")

    # --- Pass 1: whole-dataset constants ---

    constants = _find_constant_columns(pd.DataFrame(A, columns=names))
    emit(f"\nPASS 1 -- constant everywhere ({len(constants)}):")
    for c in constants:
        emit(f"  {c} = {_fmt_num(A[:, names.index(c)][0])}  (constant)")

    # --- Pass 2: constant within each claim type ---
    keep_mask = [n not in set(constants) for n in names]
    names2 = [n for n, k in zip(names, keep_mask) if k]
    A2 = A[:, keep_mask]
    del A
    gc.collect()
    indicators = _claim_type_indicator_columns(A2, names2)
    emit(f"\nPASS 2 -- constant WITHIN each claim type = exact combination of claim_type dummies ({len(indicators)}):")
    for name, terms in indicators.items():
        emit(f"  {_format_equation(name, terms)}")

    # --- Pass 3: sparse equations for everything else ---
    keep3 = [n not in indicators for n in names2]
    names3 = [n for n, k in zip(names2, keep3) if k]
    A3 = A2[:, keep3]
    del A2
    gc.collect()
    rank3 = np.linalg.matrix_rank(A3)
    deficiency3 = A3.shape[1] - rank3
    emit(f"\nAfter passes 1-2: {A3.shape[1]} columns, rank {rank3}, remaining deficiency {deficiency3}")
    if deficiency3 == 0:
        emit("  Passes 1-2 fully explain the deficiency.")
        return out

    X3 = pd.DataFrame(A3, columns=names3)
    dropped = _find_redundant_columns_via_qr(X3, deficiency3)
    del X3
    gc.collect()

    kept_idx = [i for i, n in enumerate(names3) if n not in set(dropped)]
    kept_names = [names3[i] for i in kept_idx]
    Xk = A3[:, kept_idx]
    col_norms = np.linalg.norm(Xk, axis=0)
    col_norms[col_norms == 0] = 1.0
    if np.linalg.matrix_rank(Xk) != len(kept_idx):
        emit("  [warning] kept set is not full rank -- equations below may not be unique")

    emit(f"\nPASS 3 -- {deficiency3} remaining dependencies as sparse exact equations (OMP against the kept columns):")
    for d in dropped:
        y = A3[:, names3.index(d)]
        sel, coef, rel = _sparse_explain(y, Xk, col_norms, args.max_terms, args.tol)
        terms = [(kept_names[j], float(c)) for j, c in zip(sel, coef) if abs(c) > 1e-9]
        if rel < args.tol:
            emit(f"  {_format_equation(d, terms)}    [exact, rel. residual {rel:.1e}]")
        else:
            top = sorted(terms, key=lambda t: -abs(t[1]) * col_norms[kept_names.index(t[0])])[:10]
            emit(
                f"  {d}: NOT sparse within {args.max_terms} terms (rel. residual {rel:.1e}) -- "
                f"largest contributors: {_format_equation(d, top)}"
            )
    return out


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["restricted", "unrestricted"] if args.matrix == "both" else [args.matrix]
    for kind in kinds:
        lines = _explain(kind, args)
        suffix = kind + ("_excl" if args.exclude_separating_codes else "")
        path = REPORTS_DIR / f"rank_deficiency_explained__{suffix}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"\nWritten to {path}\n")


if __name__ == "__main__":
    main()
