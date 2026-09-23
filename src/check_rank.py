"""
Quick, targeted check: is the "weighted design matrix is rank deficient"
error from firthmodels (--method firth --sample-frac 0.02) caused by a raw
collinearity in X itself, or only by IRLS weight collapse under severe
separation (the 6 confirmed structurally-separating HCPCS codes)?

Computes np.linalg.matrix_rank() on the RAW (unweighted) restricted design
matrix, at a given --sample-frac, both WITH and WITHOUT
--exclude-separating-codes.

CONFIRMED 2026-09-23 at --sample-frac 0.02: rank deficient by exactly 25
in BOTH cases (144->119 without exclusion, 138->113 with it) -- the
deficiency count is UNCHANGED by removing the 6 separating codes, ruling
out the separation-driven-IRLS-weight-collapse theory entirely. This is a
genuine collinearity in the raw design, unrelated to the HCPCS-separation
problem.

LEADING HYPOTHESIS (not yet confirmed -- see _identify_dependencies()
below, added same day): top_n_encode()'s "top-20 most frequent" cardinality
encoding (build_chow_design_matrix.py) is computed on whatever DataFrame is
passed to it -- at --sample-frac 0.02, that's the ALREADY-SUBSAMPLED
23,039-row train, not the full ~1.15M-row population. With DME's already-
small slice shrunk further by subsampling, it's plausible several
categorical dummies end up PERFECTLY coinciding with a claim_type dummy by
small-sample chance (e.g., a rare diagnosis/state category present in
every row of one claim type and absent from the others in this specific
draw, making that dummy column exactly equal to -- or the exact complement
of -- a claim_type dummy). If so, this should resolve at a larger
--sample-frac, where the top-20 list stabilizes and such coincidences
become far less likely.

This also likely explains something retroactively: the same deficiency was
probably present (silently) in every earlier statsmodels/LBFGS run too --
LBFGS doesn't need to invert anything to find the likelihood maximum, so
it pushed through, and the HessianInversionWarning seen on every prior run
(which only fires computing standard errors AFTER optimization) was likely
THIS issue, separate from the HCPCS-separation problem that caused the
degenerate identical-log-likelihoods finding. The log-likelihood VALUE
itself stays well-defined under rank deficiency even though the specific
coefficients achieving it aren't unique -- firthmodels is just stricter,
since Newton-Raphson needs to invert the weighted matrix at EVERY
iteration, not just once at the end for inference.

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
        help="Fraction of train_model.parquet to check (default 0.02, matching "
        "the run that first hit the firthmodels rank-deficiency error). Pass a "
        "larger value (e.g. 0.2, or 1.0 for the full file) to test whether the "
        "deficiency is a small-sample artifact of computing the top-20 "
        "cardinality encoding on an already-subsampled DataFrame.",
    )
    parser.add_argument("--stream-batch-size", type=int, default=50_000)
    return parser.parse_args()


def _identify_dependencies(X: pd.DataFrame, rank: int) -> None:
    """For each null-space direction (X's smallest singular values), print
    the columns with the largest-magnitude coefficients -- the strongest
    hint of which columns are involved in that specific linear dependency.
    Not a perfect sparse attribution (a null-space vector can in principle
    spread weight thinly across many columns with no single clean
    combination), but a strong first-pass diagnostic given the leading
    hypothesis above expects a simple 'one category dummy exactly cancels
    one claim_type dummy' pattern, which shows up as two large,
    opposite-signed coefficients in the same null vector."""
    ncols = X.shape[1]
    deficiency = ncols - rank
    if deficiency <= 0:
        return
    print(f"\n  Identifying likely columns behind the {deficiency} dependency(ies) (via SVD null space)...")
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
    X_arr = X.to_numpy(dtype=np.float64)
    rank = np.linalg.matrix_rank(X_arr)
    ncols = X_arr.shape[1]
    print(f"  numpy.linalg.matrix_rank(X): {rank} (ncols: {ncols})")
    if rank < ncols:
        print(f"  RANK DEFICIENT by {ncols - rank} -- a genuine collinearity in the RAW (unweighted) matrix.")
        _identify_dependencies(X, rank)
    else:
        print("  FULL RANK -- no collinearity issue at this sample size.")


def main() -> None:
    args = parse_args()
    _check(args.sample_frac, args.stream_batch_size, exclude_separating=False)
    _check(args.sample_frac, args.stream_batch_size, exclude_separating=True)


if __name__ == "__main__":
    main()
