"""
Quick, targeted check: is the "weighted design matrix is rank deficient"
error from firthmodels (--method firth --sample-frac 0.02) caused by a raw
collinearity in X itself, or only by IRLS weight collapse under severe
separation (the 6 confirmed structurally-separating HCPCS codes)?

Computes np.linalg.matrix_rank() on the RAW (unweighted) restricted design
matrix, at the exact same --sample-frac used in the failing run, both WITH
and WITHOUT --exclude-separating-codes:

  - If the raw matrix is already rank-deficient even WITHOUT excluding the
    6 codes, and stays deficient WITH exclusion too, that's a genuine,
    separate collinearity issue unrelated to separation -- needs its own
    investigation (which columns, via QR with pivoting or similar).
  - If the raw matrix is FULL RANK in both cases, the rank deficiency only
    shows up once IRLS weighting is applied -- confirming it's a
    separation-induced numerical issue, not a raw design problem, and
    that --exclude-separating-codes should let --method firth run cleanly.

Run with (from the repo root, inside the venv):
    python src/check_rank.py
"""

from __future__ import annotations

import numpy as np

from build_chow_design_matrix import build_chow_design_matrix, build_restricted_design_matrix
from fit_chow_test import _load_train, _prepare_xy

SAMPLE_FRAC = 0.02
STREAM_BATCH_SIZE = 50_000


def _check(exclude_separating: bool) -> None:
    label = "WITH --exclude-separating-codes" if exclude_separating else "WITHOUT --exclude-separating-codes (matches the failing run)"
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

    train = _load_train(SAMPLE_FRAC, STREAM_BATCH_SIZE)
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
    else:
        print("  FULL RANK -- the raw matrix itself is fine; a weighted-matrix rank deficiency "
              "would have to come from IRLS weight collapse under separation, not a raw design problem.")


def main() -> None:
    _check(exclude_separating=False)
    _check(exclude_separating=True)


if __name__ == "__main__":
    main()
