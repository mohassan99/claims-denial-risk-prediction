"""
Quick, targeted check: is the "weighted design matrix is rank deficient"
error from firthmodels (--method firth --sample-frac 0.02) caused by a raw
collinearity in X itself, or only by IRLS weight collapse under severe
separation (the 6 confirmed structurally-separating HCPCS codes)?

Computes np.linalg.matrix_rank() on the RAW (unweighted) design matrix, at
a given --sample-frac, for BOTH the restricted and unrestricted matrices,
both WITH and WITHOUT --exclude-separating-codes (4 checks total).

CONFIRMED 2026-09-23, RESTRICTED matrix: rank deficient by EXACTLY 25 at
BOTH --sample-frac 0.02 and 0.2 (10x more data, identical deficiency
count), and unchanged by --exclude-separating-codes -- ruling out
separation-driven IRLS weight collapse AND a small-sample cardinality-
encoding artifact. Real, sample-size-invariant structural issue.

STAGE 1, CONFIRMED: 11 of the 25 are LITERALLY CONSTANT columns (all
value 0.0), identical across both exclusion settings -- e.g.
REV_CNTR_2ND_MSP_PD_AMT__x__outpatient, DMERC_LINE_SCRN_SVGS_AMT__x__dme.
Points at a real gap in build_chow_design_matrix.py: Pass 2 has
_interact_with_zero_variance_guard() to skip constant-zero interaction
terms; Pass 1 (numeric claim-type-exclusive/2-of-3 covariates) has no
equivalent -- its _null_pattern() helper checks NULLness only, never
whether the non-null values are all identically the same constant.

STAGE 2, CONFIRMED: dropping the 11 constants leaves a real remaining
deficiency of 14 in the RESTRICTED matrix, IDENTICAL as a column SET
(not just a count) across both exclusion settings once the QR search is
protected from touching the claim_type dummies (see CRITICAL CORRECTNESS
FIX below) -- a clean, reproducible, fully-verified result.
has_performing_physician == claim_type_carrier EXACTLY (100% of rows) --
a literal duplicate column, not just "carrier-only" as the 2026-09-22
finding already established. The competing hypothesis (has_attending/
operating/rendering_physician summing exactly to claim_type_outpatient)
is REFUTED at 67.8%. The remaining structure is a genuinely tangled,
multi-column relationship among REV_CNTR_*/CLM_*_outpatient dollar fields
(plausibly real CMS revenue-center billing arithmetic) entangled with all
5 NPI presence flags -- not fully named by mechanism, but fully resolved
by column identification (see the QR drop list, verified to reach full
rank).

NOT YET CHECKED as of the previous run: whether the UNRESTRICTED matrix
has an analogous problem. It has a DIFFERENT column structure for the
same covariates (separate __x__carrier/__x__outpatient/__x__dme terms
instead of one pooled column), so the same underlying identities could
show up differently -- e.g. has_performing_physician__x__carrier (the
only interaction term Pass 2's zero-variance guard keeps for that flag,
since it's confirmed carrier-only) equals claim_type_carrier *
claim_type_carrier = claim_type_carrier exactly, the identical redundancy
in interaction-term form. This matters beyond mere completeness: whatever
gets excluded must match between the two matrices at the level of the
underlying COVARIATE, or _compute_df_and_table()'s restricted/unrestricted
correspondence check in fit_chow_test.py would break. This version checks
both matrix kinds.

CRITICAL CORRECTNESS FIX (2026-09-23): the first version of
_find_redundant_columns_via_qr() ran QR pivoting on ALL columns
unconstrained, and it picked claim_type_carrier/outpatient/dme THEMSELVES
as part of the "redundant" set in one run. QR pivoting is not unique when
columns are entangled in a shared degenerate subspace -- nothing stops it
from choosing a structurally ESSENTIAL column over an equally-valid
alternative. Actually dropping a claim_type dummy would silently destroy
the cell-means, no-shared-intercept design this whole project depends on.
_PROTECTED_COLS below is now NEVER eligible to be named redundant:
everything else is orthogonalized against the protected block first, and
pivoted QR runs only on the residual.

Run with (from the repo root, inside the venv):
    python src/check_rank.py
    python src/check_rank.py --sample-frac 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.linalg

from build_chow_design_matrix import (
    add_claim_type_interactions,
    build_chow_design_matrix,
    build_restricted_design_matrix,
)
from fit_chow_test import _load_train, _prepare_xy

# Columns that must NEVER be flagged as "redundant" by the QR search below,
# no matter what -- see module docstring's CRITICAL CORRECTNESS FIX.
_PROTECTED_COLS = {"claim_type_carrier", "claim_type_outpatient", "claim_type_dme"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.02,
        help="Fraction of train_model.parquet to check (default 0.02). Pass a "
        "larger value (e.g. 0.2, or 1.0 for the full file) -- confirmed 2026-09-23 "
        "that the RESTRICTED matrix's deficiency count does NOT change between "
        "0.02 and 0.2, so this is no longer expected to resolve at a larger "
        "sample, but left configurable for further confirmation at the full "
        "population and for the not-yet-checked unrestricted matrix.",
    )
    parser.add_argument("--stream-batch-size", type=int, default=50_000)
    return parser.parse_args()


def _find_constant_columns(X: pd.DataFrame) -> list[str]:
    """Columns with exactly one distinct value across every row -- the
    definitive, unambiguous first check for rank deficiency. See module
    docstring's STAGE 1."""
    return [c for c in X.columns if X[c].nunique(dropna=False) <= 1]


def _check_npi_sum_identity(X: pd.DataFrame) -> None:
    """Direct test of two specific, substantive hypotheses (module
    docstring STAGE 2). Only meaningful for the RESTRICTED matrix, where
    these flags are plain columns -- skipped (columns won't be present
    under these exact names) for the unrestricted matrix."""
    npi_cols = ["has_attending_physician", "has_operating_physician", "has_rendering_physician"]
    if all(c in X.columns for c in npi_cols) and "claim_type_outpatient" in X.columns:
        npi_sum = X[npi_cols].sum(axis=1)
        exact_match = (npi_sum == X["claim_type_outpatient"]).mean()
        verdict = "EXACT IDENTITY CONFIRMED" if exact_match == 1.0 else "not exact -- hypothesis refuted or only partial"
        print(
            f"\n  [NPI identity check] has_attending_physician + has_operating_physician + "
            f"has_rendering_physician == claim_type_outpatient for {exact_match:.4%} of rows ({verdict})"
        )
    if "has_performing_physician" in X.columns and "claim_type_carrier" in X.columns:
        exact_match2 = (X["has_performing_physician"] == X["claim_type_carrier"]).mean()
        verdict2 = "EXACT IDENTITY CONFIRMED" if exact_match2 == 1.0 else "not exact"
        print(
            f"  [NPI identity check] has_performing_physician == claim_type_carrier for "
            f"{exact_match2:.4%} of rows ({verdict2})"
        )


def _find_redundant_columns_via_qr(
    X: pd.DataFrame, deficiency: int, protect: set[str] = _PROTECTED_COLS
) -> list[str]:
    """QR decomposition with column pivoting, applied ONLY to the
    non-protected columns after orthogonalizing them against the protected
    block (Q @ (Q.T @ other), subtracted off) -- see module docstring's
    CRITICAL CORRECTNESS FIX for why the protected set exists at all. A
    column in `protect` can therefore never appear in the returned list."""
    protected = [c for c in X.columns if c in protect]
    other = [c for c in X.columns if c not in protect]
    if not protected:
        _, _, pivot = scipy.linalg.qr(X.to_numpy(dtype=np.float64), mode="economic", pivoting=True)
        redundant_idx = pivot[-deficiency:]
        colnames = X.columns.to_numpy()
        return [str(colnames[i]) for i in redundant_idx]

    Xp = X[protected].to_numpy(dtype=np.float64)
    Xo = X[other].to_numpy(dtype=np.float64)
    Qp, _ = np.linalg.qr(Xp)
    Xo_orth = Xo - Qp @ (Qp.T @ Xo)

    _, _, pivot = scipy.linalg.qr(Xo_orth, mode="economic", pivoting=True)
    redundant_idx = pivot[-deficiency:]
    return [other[i] for i in redundant_idx]


def _identify_remaining_dependencies(X: pd.DataFrame, rank: int) -> None:
    """SVD null-space view (co-involvement) plus the protected QR-based
    approach (a directly actionable, SAFE drop set -- never a claim_type
    dummy), with a full-rank re-verification after dropping."""
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

    present_protected = sorted(set(X.columns) & _PROTECTED_COLS)
    print(
        f"\n  Directly actionable alternative (QR with column pivoting, "
        f"PROTECTED from being dropped: {present_protected}) -- "
        f"drop these {deficiency} column(s) to reach full rank:"
    )
    redundant = _find_redundant_columns_via_qr(X, deficiency)
    for c in redundant:
        print(f"    {c}")
    dropped_a_protected_col = bool(set(redundant) & _PROTECTED_COLS)
    if dropped_a_protected_col:
        raise RuntimeError(
            f"BUG: _find_redundant_columns_via_qr returned a protected column in {redundant} "
            "-- the orthogonal-projection guard failed. Do not trust this drop list."
        )
    verify_rank = np.linalg.matrix_rank(X.drop(columns=redundant).to_numpy(dtype=np.float64))
    verify_ncols = X.shape[1] - deficiency
    print(
        f"  Verification: dropping these {deficiency} column(s) gives rank "
        f"{verify_rank} of {verify_ncols} columns "
        f"({'CONFIRMED full rank, no protected column touched' if verify_rank == verify_ncols else 'STILL DEFICIENT -- investigate further'})"
    )

    _check_npi_sum_identity(X)


def _check(sample_frac: float, stream_batch_size: int, exclude_separating: bool, matrix_kind: str) -> None:
    label = (
        f"{matrix_kind.upper()} matrix, "
        f"{'WITH' if exclude_separating else 'WITHOUT'} --exclude-separating-codes"
    )
    print(f"\n{'=' * 78}\n{label}  (--sample-frac {sample_frac})\n{'=' * 78}")

    train = _load_train(sample_frac, stream_batch_size)
    intermediate = build_chow_design_matrix(train)
    del train

    if matrix_kind == "restricted":
        design_df = build_restricted_design_matrix(intermediate)
        del intermediate
    else:
        design_df, _interaction_cols, _dropped_zero_variance = add_claim_type_interactions(intermediate)
        del intermediate

    y, X = _prepare_xy(design_df, exclude_separating=exclude_separating)
    del design_df

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
    for matrix_kind in ("restricted", "unrestricted"):
        for exclude in (False, True):
            _check(args.sample_frac, args.stream_batch_size, exclude, matrix_kind)


if __name__ == "__main__":
    main()
