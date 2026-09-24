"""
Quick, targeted check: is the "weighted design matrix is rank deficient"
error from firthmodels (--method firth --sample-frac 0.02) caused by a raw
collinearity in X itself, or only by IRLS weight collapse under severe
separation (the 6 confirmed structurally-separating HCPCS codes)?

Computes np.linalg.matrix_rank() on the RAW (unweighted) restricted design
matrix, at a given --sample-frac, both WITH and WITHOUT
--exclude-separating-codes.

CONFIRMED 2026-09-23: rank deficient by EXACTLY 25 at BOTH --sample-frac
0.02 and 0.2 (10x more data, identical deficiency count), and unchanged by
--exclude-separating-codes in both cases. Two theories fully ruled out by
this: (1) separation-driven IRLS weight collapse (deficiency doesn't move
when the 6 separating codes are removed), and (2) a small-sample artifact
of computing the top-20 cardinality encoding on an already-subsampled
DataFrame (deficiency is IDENTICAL at 10x the sample size). This is a
real, sample-size-invariant structural issue.

CONFIRMED MECHANISM (2026-09-23, from the printed null-space vectors):
most of the 25 dependencies show ONE column sitting at a coefficient near
+-1.0 with every other column negligible (e.g. REV_CNTR_2ND_MSP_PD_AMT__x__
outpatient, REV_CNTR_BENE_PMT_AMT__x__outpatient,
REV_CNTR_1ST_MSP_PD_AMT__x__outpatient, DMERC_LINE_SCRN_SVGS_AMT__x__dme).
That's the signature of a single column that's LITERALLY CONSTANT (almost
certainly all-zero) across the entire dataset, not a genuine two-variable
relationship -- and it points at a real, specific gap in
build_chow_design_matrix.py: Pass 2 (the genuinely-3-of-3 categorical
dummy groups) has _interact_with_zero_variance_guard() to skip interaction
terms that come out constant-zero, but Pass 1 (the numeric claim-type-
exclusive/2-of-3 covariates) has NO equivalent guard -- its _null_pattern()
helper only checks whether a field is NULL per claim type, never whether
its NON-null values are all identically the same constant (e.g. always
exactly $0.00, plausible for niche real-world fields like secondary-payer-
coordination amounts or managed-care-paid switches that a synthetic
generator may simply never populate with a nonzero value). This would
explain why it doesn't resolve with more data -- a population-wide
constant field stays constant at any sample size. _find_constant_columns()
below checks this directly (a column's own distinct-value count), which is
unambiguous, rather than continuing to eyeball noisy SVD output -- SVD's
null-space basis is not unique when MULTIPLE independent all-zero columns
exist together, so it can visually "mix" unrelated constant columns into
the same printed vector (see dependencies 15/16 in the 0.2 run, which
likely aren't a real two-variable relationship at all, just two separately-
constant columns sharing a 2-D null subspace with an arbitrary basis).

Run with (from the repo root, inside the venv):
    python src/check_rank.py
    python src/check_rank.py --sample-frac 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from build_chow_design_matrix import build_chow_design_matrix, build_restricted_design_matrix
from fit_chow_test import _load_train, _prepare_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.02,
        help="Fraction of train_model.parquet to check (default 0.02). Pass a "
        "larger value (e.g. 0.2, or 1.0 for the full file) -- confirmed 2026-09-23 "
        "that the deficiency count does NOT change between 0.02 and 0.2, so this "
        "is no longer expected to resolve at a larger sample, but left "
        "configurable for further confirmation at the full population.",
    )
    parser.add_argument("--stream-batch-size", type=int, default=50_000)
    return parser.parse_args()


def _find_constant_columns(X: pd.DataFrame) -> list[str]:
    """Columns with exactly one distinct value across every row -- the
    definitive, unambiguous first check for rank deficiency, since
    build_chow_design_matrix.py's Pass 1 (numeric claim-type-exclusive/
    2-of-3 covariates) checks only whether a field is NULL per claim type,
    never whether its non-null values are all identically the SAME
    constant. A constant column (especially an all-zero one, which a
    value*claim_type_dummy interaction term would be if the underlying
    field is always exactly 0 within that claim type) trivially reduces
    rank by 1 regardless of sample size."""
    return [c for c in X.columns if X[c].nunique(dropna=False) <= 1]


def _identify_remaining_dependencies(X: pd.DataFrame, rank: int) -> None:
    """For each null-space direction of a (hopefully much smaller) residual
    problem, print the columns with the largest-magnitude coefficients --
    only called after constant columns are already removed, so any
    remaining deficiency here is a genuine multi-column relationship, not
    a constant column getting smeared across the printed output by SVD's
    non-unique basis for a degenerate subspace."""
    ncols = X.shape[1]
    deficiency = ncols - rank
    if deficiency <= 0:
        return
    print(f"\n  Identifying likely columns behind the {deficiency} REMAINING dependency(ies) (via SVD null space)...")
    _, _, vt = np.linalg.svd(X.to_numpy(dtype=np.float64), full_matrices=False)
    null_vectors = vt[rank:]
    colnames = X.columns.to_numpy()
    for i, vec in enumerate(null_vectors):
        order = np.argsort(-np.abs(vec))[:6]
        top = [(str(colnames[j]), round(float(vec[j]), 3)) for j in order]
        print(f"    dependency {i + 1}: {top}")


def _check(sample_frac: float, stream_batch_size: int, exclude_separating: bool) -> None:
    label = "WITH --exclude-separating-codes" if exclude_separating else "WITHOUT --exclude-separating-codes"
    print(f"\n{'=' * 78}\n{label}  (--sample-frac {sample_frac})\n{'=' * 78}")

    train = _load_train(sample_frac, stream_batch_size)
    intermediate = build_chow_design_matrix(train)
    del train

    restricted_df = build_restricted_design_matrix(intermediate)
    del intermediate
    y, X = _prepare_xy(restricted_df, exclude_separating=exclude_separating)
    del restricted_df

    print(f"\n  X shape: {X.shape}")
    ncols = X.shape[1]
    rank = np.linalg.matrix_rank(X.to_numpy(dtype=np.float64))
    print(f"  numpy.linalg.matrix_rank(X): {rank} (ncols: {ncols})")
    if rank == ncols:
        print("  FULL RANK -- no collinearity issue at this sample size.")
        return

    deficiency = ncols - rank
    print(f"  RANK DEFICIENT by {deficiency} -- a genuine collinearity in the RAW (unweighted) matrix.")

    constant_cols = _find_constant_columns(X)
    print(f"\n  Columns with exactly 1 distinct value (definite rank-deficiency contributors): {len(constant_cols)}")
    if constant_cols:
        for c in constant_cols:
            print(f"    {c}  (constant value: {X[c].iloc[0]!r})")

    if not constant_cols:
        _identify_remaining_dependencies(X, rank)
        return

    X_reduced = X.drop(columns=constant_cols)
    reduced_rank = np.linalg.matrix_rank(X_reduced.to_numpy(dtype=np.float64))
    reduced_ncols = X_reduced.shape[1]
    remaining_deficiency = reduced_ncols - reduced_rank
    print(
        f"\n  After dropping the {len(constant_cols)} constant column(s): "
        f"{reduced_ncols} columns, rank {reduced_rank}, remaining deficiency {remaining_deficiency}"
    )
    if remaining_deficiency > 0:
        _identify_remaining_dependencies(X_reduced, reduced_rank)
    else:
        print("  Constant columns fully explain the deficiency -- no further collinearity beyond that.")


def main() -> None:
    args = parse_args()
    _check(args.sample_frac, args.stream_batch_size, exclude_separating=False)
    _check(args.sample_frac, args.stream_batch_size, exclude_separating=True)


if __name__ == "__main__":
    main()
