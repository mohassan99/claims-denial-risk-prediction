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
--exclude-separating-codes in both cases -- ruling out separation-driven
IRLS weight collapse AND a small-sample cardinality-encoding artifact.
This is a real, sample-size-invariant structural issue.

STAGE 1, CONFIRMED: 11 of the 25 are LITERALLY CONSTANT columns (all
value 0.0), identical in both the with- and without-exclusion runs --
e.g. REV_CNTR_2ND_MSP_PD_AMT__x__outpatient, DMERC_LINE_SCRN_SVGS_AMT__x__
dme. Points at a real, specific gap in build_chow_design_matrix.py: Pass 2
(genuinely-3-of-3 categorical dummy groups) has
_interact_with_zero_variance_guard() to skip constant-zero interaction
terms; Pass 1 (numeric claim-type-exclusive/2-of-3 covariates) has NO
equivalent -- its _null_pattern() helper only checks NULLness per claim
type, never whether the non-null values are all identically the same
constant (plausible for niche real-world fields -- secondary-payer-
coordination amounts, managed-care-paid switches -- that a synthetic
generator may simply never populate with a nonzero value). Explains why
this doesn't resolve with more data: a population-wide constant field
stays constant at any sample size.

STAGE 2, IN PROGRESS: dropping the 11 constants leaves a REAL remaining
deficiency of 14, unchanged by --exclude-separating-codes. The NPI
presence flags (has_referring/performing/attending/operating/rendering_
physician) and claim_type_* dummies recur across nearly every printed
SVD null-space vector for this residual -- plausibly connected to an
already-documented finding (decisions-and-learnings.md, 2026-09-22): 4 of
the 5 NPI flags are empirically claim-type-exclusive (has_performing_
physician carrier-only; has_attending/operating/rendering_physician all
outpatient-only). HYPOTHESIS: if CMS's outpatient billing rules require
EXACTLY ONE of those three physician roles populated per outpatient
claim, their sum would equal claim_type_outpatient EXACTLY -- a genuine
accounting identity, not a bug. _check_npi_sum_identity() tests this
directly. Separately, a cluster of REV_CNTR_*/CLM_*_outpatient dollar-
amount interaction terms also recur together, plausibly a real CMS
billing arithmetic identity (a total defined as the sum of its
components) -- not yet identified by name. _find_redundant_columns_via_qr()
gives a directly actionable "drop exactly these columns" answer via QR
column pivoting, independent of whether every mechanism gets named.

Run with (from the repo root, inside the venv):
    python src/check_rank.py
    python src/check_rank.py --sample-frac 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.linalg

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
    definitive, unambiguous first check for rank deficiency. See module
    docstring's STAGE 1."""
    return [c for c in X.columns if X[c].nunique(dropna=False) <= 1]


def _check_npi_sum_identity(X: pd.DataFrame) -> None:
    """Direct test of a specific, substantive hypothesis (module docstring
    STAGE 2): if CMS's outpatient billing rules require EXACTLY ONE of
    has_attending/operating/rendering_physician populated per outpatient
    claim, their sum equals claim_type_outpatient EXACTLY for every row --
    a genuine accounting identity, not a coding bug. Checked directly
    rather than assumed. Also checks has_performing_physician against
    claim_type_carrier, the other empirically-claim-type-exclusive flag."""
    npi_cols = ["has_attending_physician", "has_operating_physician", "has_rendering_physician"]
    if all(c in X.columns for c in npi_cols) and "claim_type_outpatient" in X.columns:
        npi_sum = X[npi_cols].sum(axis=1)
        exact_match = (npi_sum == X["claim_type_outpatient"]).mean()
        verdict = "EXACT IDENTITY CONFIRMED" if exact_match == 1.0 else "not exact -- hypothesis refuted or only partial"
        print(
            f"\n  [NPI identity check] has_attending_physician + has_operating_physician + "
            f"has_rendering_physician == claim_type_outpatient for {exact_match:.4%} of rows ({verdict})"
        )
    else:
        print("\n  [NPI identity check] required columns not present in this X -- skipped.")

    if "has_performing_physician" in X.columns and "claim_type_carrier" in X.columns:
        exact_match2 = (X["has_performing_physician"] == X["claim_type_carrier"]).mean()
        verdict2 = "EXACT IDENTITY CONFIRMED" if exact_match2 == 1.0 else "not exact"
        print(
            f"  [NPI identity check] has_performing_physician == claim_type_carrier for "
            f"{exact_match2:.4%} of rows ({verdict2})"
        )


def _find_redundant_columns_via_qr(X: pd.DataFrame, deficiency: int) -> list[str]:
    """QR decomposition with column pivoting greedily orders columns by how
    much NEW (linearly independent) information each adds, given columns
    already selected. The LAST `deficiency` columns in pivot order are a
    valid, directly actionable set to drop to reach full rank -- unlike
    SVD's null-space view (which shows which columns are INVOLVED in a
    dependency, not which specific subset to remove to resolve it)."""
    _, _, pivot = scipy.linalg.qr(X.to_numpy(dtype=np.float64), mode="economic", pivoting=True)
    redundant_idx = pivot[-deficiency:]
    colnames = X.columns.to_numpy()
    return [str(colnames[i]) for i in redundant_idx]


def _identify_remaining_dependencies(X: pd.DataFrame, rank: int) -> None:
    """SVD null-space view -- which columns are INVOLVED in each remaining
    dependency (not necessarily a clean sparse combination; see module
    docstring). Kept alongside the QR-based approach below since the two
    give complementary information: this shows co-involvement, QR gives a
    directly actionable drop set."""
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

    print(f"\n  Directly actionable alternative (QR with column pivoting) -- drop these {deficiency} column(s) to reach full rank:")
    redundant = _find_redundant_columns_via_qr(X, deficiency)
    for c in redundant:
        print(f"    {c}")
    verify_rank = np.linalg.matrix_rank(X.drop(columns=redundant).to_numpy(dtype=np.float64))
    verify_ncols = X.shape[1] - deficiency
    print(
        f"  Verification: dropping these {deficiency} column(s) gives rank "
        f"{verify_rank} of {verify_ncols} columns "
        f"({'CONFIRMED full rank' if verify_rank == verify_ncols else 'STILL DEFICIENT -- QR pivot choice was not sufficient, investigate further'})"
    )

    _check_npi_sum_identity(X)


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
