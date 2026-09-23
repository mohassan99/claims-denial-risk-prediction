"""
Phase 2 -- fit the Chow test.

The two design matrices (restricted/pooled and unrestricted/fully-
interacted) are already built and verified by
src/build_chow_design_matrix.py (see data/FEATURE_ENGINEERING.md Sections
3, 5, 6 for the full design/derivation history). What's left, per the
2026-09-22 handoff, is the fit itself:

    LR = -2 * (ll_restricted - ll_unrestricted)   ~   chi2(df)
    H0: every genuinely-shared covariate has one coefficient, pooled
        across claim types (the restricted model is adequate)
    H1: at least one genuinely-shared covariate's effect differs by
        claim type somewhere

DEGREES OF FREEDOM -- COMPUTED, NOT HAND-ENTERED. df = Sigma_i (n_i - 1)
over every genuinely-shared covariate is correct per
FEATURE_ENGINEERING.md Section 3/6, but hand-transcribing that from the
audit tables there (13 confirmed 2-of-3 covariates, ~161 nominal 3-of-3
covariates minus 39 empirically-zero-variance corrections) risks a
transcription error on top of an already-long correction history.
Instead _compute_df_and_table() below derives df directly from the
actual column names in the two already-built design matrices: a
genuinely-shared covariate is exactly one whose base name appears as
N>=1 "<name>__x__<claim_type>" columns in the unrestricted matrix and as
a single plain "<name>" column in the restricted matrix (see
build_chow_design_matrix.py's own docstrings -- a claim-type-EXCLUSIVE
field, by contrast, is named IDENTICALLY in both matrices, so it never
shows up as a difference here at all, correctly contributing 0 df
without needing special-casing). This is the same quantity
build_chow_design_matrix.py's own __main__ block already prints as
"Column count difference (unrestricted - restricted)" (166, as of the
last confirmed run: 446 unrestricted, 280 restricted) -- this script
cross-checks its per-covariate-derived df against that raw shape
difference and refuses to proceed if they disagree, since that would
mean the two matrices have silently diverged in some way neither this
script nor the shape-diff check alone would otherwise catch.

MEMORY. The design matrices are ~3.3 GB / ~3.0 GB just from the encoding
step (FEATURE_ENGINEERING.md Section 6, 2026-09-22 addendum) -- on a
machine that already hit its memory ceiling 3 separate times building
just the encodings. statsmodels needs its own working memory on top of
holding one matrix (gradient/line-search arrays roughly the size of the
exog array itself), so this script:
  - never holds `train`, `intermediate`, and BOTH design matrices at once
    -- restricted is built, fit, and its result reduced to a few scalars
    before the unrestricted matrix is even built (see main()).
  - never pre-casts X to float64 itself before handing it to sm.Logit()
    -- that would allocate a full-size float64 copy (roughly 1.15M rows
    x 446 cols x 8 bytes =~ 4.1 GB for the unrestricted matrix alone) IN
    ADDITION to the equivalent array statsmodels builds internally
    regardless. Passing the mixed-dtype (int8 + float64) DataFrame
    straight through lets statsmodels do that upcast exactly once.
  - explicitly `del`s each design matrix AND fit-result object (a fit
    result retains its own reference to the exog array via
    `result.model.exog`, so deleting only the DataFrame variable would
    NOT free that memory) plus `gc.collect()`, between the two fits.
  - supports --sample-frac for a fast, low-memory smoke test of the
    whole pipeline (stratified on is_denied) before committing to a full
    run, and --restricted-only / --skip-restricted so a crash on the
    (larger, riskier) unrestricted fit doesn't require re-fitting the
    restricted model too.

KNOWN, UNRESOLVED RISK: fully interacting many rare one-hot categories
(e.g. a specific top-N HCPCS/diagnosis code) against DME, the smallest
claim type (66,335 rows), creates real quasi-complete-separation risk
for some coefficients even after the zero-variance guard removes the
fully-zero cases -- e.g. a category that happens to be ~100% denied
within one claim type. method='lbfgs' (below) degrades more gracefully
than statsmodels' Newton-Raphson default under this kind of near-
singularity, but doesn't eliminate the risk. This script surfaces
non-convergence / separation warnings; it does not attempt to fix them.
Any individual coefficient reported with a very large magnitude and a
very large standard error is worth checking against its raw cell counts
before trusting it -- flagged here as an open item for whoever reads the
fit output next, not treated as resolved.

Run with (from the repo root, inside the venv):
    python src/fit_chow_test.py
    python src/fit_chow_test.py --sample-frac 0.02      # smoke test first
    python src/fit_chow_test.py --restricted-only       # stage 1 of 2
    python src/fit_chow_test.py --skip-restricted       # stage 2 of 2
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from build_chow_design_matrix import (
    PROCESSED_DIR,
    add_claim_type_interactions,
    build_chow_design_matrix,
    build_restricted_design_matrix,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RESULTS_PATH = REPORTS_DIR / "chow_test_results.txt"
RESTRICTED_SUMMARY_PATH = REPORTS_DIR / "chow_restricted_fit_summary.json"

LABEL_COL = "is_denied"

# Columns that are never features -- identifiers and the label. These sit
# in build_chow_design_matrix.py's _NEVER_INTERACT set, so they pass
# through both design matrices completely untouched and must be dropped
# here, never fed to Logit.
_ID_LABEL_COLS = {"BENE_ID", "CLM_ID", "_row_id", LABEL_COL}

# Also in _NEVER_INTERACT, also passed through untouched, also not usable
# as a Logit feature as-is: raw datetime columns. No numeric transform of
# these exists yet for Phase 2 (the EDA's denial-rate-over-time chart used
# the raw date directly; nothing in FEATURE_ENGINEERING.md or
# TARGET_DEFINITION.md describes a Phase-2 date feature) -- dropped here,
# loudly, rather than silently passed to Logit where they'd crash with a
# far less specific error.
_DATE_COLS_NOT_YET_FEATURIZED = {"CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT"}

_NON_FEATURE_COLS = _ID_LABEL_COLS | _DATE_COLS_NOT_YET_FEATURIZED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the Chow test (restricted vs. unrestricted logistic "
        "regression) over the claim-type-interaction design matrices."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Subsample train_model.parquet by this fraction (stratified on "
        "is_denied) before building either design matrix -- for a fast, "
        "low-memory correctness check of this whole script before "
        "committing to the full ~1.15M-row fit, which hasn't been run "
        "before. Omit for the full run. Must match between a "
        "--restricted-only run and the --skip-restricted run that follows "
        "it, or the two fits won't be nested over the same rows.",
    )
    parser.add_argument(
        "--restricted-only",
        action="store_true",
        help="Fit only the restricted (pooled) model, write its summary to "
        f"{RESTRICTED_SUMMARY_PATH.name}, and exit. Use this first given "
        "the real memory risk of building+fitting both matrices in one "
        "process -- then rerun with --skip-restricted for the second half.",
    )
    parser.add_argument(
        "--skip-restricted",
        action="store_true",
        help="Load a previously-saved restricted fit summary instead of "
        "rebuilding and refitting the restricted model. Use after a prior "
        "--restricted-only run.",
    )
    return parser.parse_args()


def _stratified_sample(df: pd.DataFrame, frac: float, label_col: str, seed: int = 42) -> pd.DataFrame:
    """Subsample while preserving the (very unbalanced) is_denied ratio --
    a plain df.sample(frac=...) risks a smoke-test run that happens to
    draw few or zero denied claims, which would make the fit meaningless
    rather than just small. Deterministic given the same input file, frac,
    and seed -- required so a --restricted-only run and the
    --skip-restricted run that follows it see the identical row set.

    CORRECTED 2026-09-23: the first version used
    df.groupby(label_col).apply(lambda g: g.sample(...)). That broke on
    real data -- newer pandas excludes the grouping column itself
    (label_col) from the sub-frame `g` passed into an .apply() callback
    (the "operating on the grouping columns" behavior change), so the
    result silently lost `is_denied` entirely. The groupby+apply call
    itself doesn't error; the KeyError only surfaced two steps later, in
    _prepare_xy, which made the actual cause easy to misread as a
    Chow-test/design-matrix bug rather than a sampling-helper one.
    Rewritten to iterate the GroupBy object directly instead of calling
    .apply() on it -- plain iteration over a GroupBy always yields the
    full sub-frame, grouping column included, in every pandas version;
    only the function-based .apply() path has the version-dependent
    exclusion behavior.
    """
    parts = [group.sample(frac=frac, random_state=seed) for _, group in df.groupby(label_col)]
    return pd.concat(parts, ignore_index=True)


def _prepare_xy(design_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Split a design matrix into (y, X): drop ID/label/not-yet-featurized-
    date columns, then fail loudly on anything statsmodels would otherwise
    choke on deep inside its own code with a much less specific error --
    a leftover non-numeric column, or a NaN that shouldn't exist given
    every claim-type-specific field is supposed to already be zero-filled
    by build_chow_design_matrix.py."""
    present_non_feature = _NON_FEATURE_COLS & set(design_df.columns)
    print(f"    dropping {len(present_non_feature)} non-feature column(s) from X: {sorted(present_non_feature)}")

    y = design_df[LABEL_COL].astype(int)
    X = design_df.drop(columns=list(present_non_feature))

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            "Non-numeric column(s) leaked into the design matrix -- fix the "
            f"upstream build in build_chow_design_matrix.py, not here: {non_numeric}"
        )

    na_counts = X.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        raise ValueError(
            "Unexpected NaN(s) remain in the design matrix -- every "
            "claim-type-specific field should already be zero-filled. "
            f"Offending column(s):\n{na_counts}"
        )

    return y, X


def _fit_logit(y: pd.Series, X: pd.DataFrame, label: str):
    """Fit one Logit model. method='lbfgs', not statsmodels' Newton-Raphson
    default: full interaction against many rare one-hot categories (see
    module docstring's KNOWN RISK section) creates real near-singularity
    risk for some coefficients even after the zero-variance guard removes
    the fully-zero cases, and LBFGS degrades more gracefully than Newton's
    direct Hessian inversion when that happens -- at the cost of needing
    more iterations, hence the raised maxiter. Passing X as a DataFrame
    (not pre-cast to a numpy array) so statsmodels does the float64 upcast
    exactly once, itself -- see module docstring's MEMORY section."""
    print(f"\n  fitting {label}: {X.shape[0]:,} rows x {X.shape[1]} columns")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.Logit(y, X).fit(method="lbfgs", maxiter=500, disp=False)
    for w in caught:
        print(f"    [warning] {w.category.__name__}: {w.message}")

    converged = result.mle_retvals.get("converged", "unknown")
    print(f"    converged: {converged}")
    print(f"    log-likelihood: {result.llf:.4f}")
    print(f"    nobs: {int(result.nobs):,}")
    return result


def _compute_df_and_table(
    unrestricted_colnames: set[str], restricted_colnames: set[str]
) -> tuple[int, list[tuple[str, int]]]:
    """df = Sigma_i (n_i - 1), derived directly from the two matrices'
    actual column names rather than the FEATURE_ENGINEERING.md audit
    tables. A genuinely-shared covariate is exactly a base name that
    appears as one or more '<base>__x__<claim_type>' columns ONLY in the
    unrestricted set and as a single plain '<base>' column ONLY in the
    restricted set -- claim-type-exclusive fields are named identically
    in both matrices (build_chow_design_matrix.py's own docstrings
    confirm this "IDENTICAL treatment" by design), so they never appear
    in either "only in X" set and are correctly never counted here, with
    no special-casing needed. Raises loudly on anything that doesn't fit
    that pattern, rather than silently mis-summing df."""
    only_unrestricted = unrestricted_colnames - restricted_colnames
    only_restricted = restricted_colnames - unrestricted_colnames

    n_i_by_base: dict[str, int] = defaultdict(int)
    for col in only_unrestricted:
        if "__x__" not in col:
            raise ValueError(
                f"{col!r} exists only in the unrestricted matrix but isn't "
                "an interaction term -- the two matrices have diverged "
                "unexpectedly; do not trust df until this is investigated."
            )
        base = col.rsplit("__x__", 1)[0]
        n_i_by_base[base] += 1

    unaccounted_restricted = only_restricted - set(n_i_by_base.keys())
    if unaccounted_restricted:
        raise ValueError(
            "Column(s) present only in the restricted matrix with no "
            f"matching interaction terms in the unrestricted matrix: "
            f"{unaccounted_restricted} -- do not trust df until investigated."
        )
    missing_pooled = set(n_i_by_base.keys()) - only_restricted
    if missing_pooled:
        raise ValueError(
            "Covariate(s) interacted in the unrestricted matrix with no "
            f"pooled column in the restricted matrix: {missing_pooled} -- "
            "do not trust df until investigated."
        )

    df_table = sorted(n_i_by_base.items(), key=lambda kv: (-kv[1], kv[0]))
    df_total = sum(n_i - 1 for _, n_i in df_table)
    return df_total, df_table


def _save_restricted_summary(result, ncols: int, colnames: list[str], sample_frac: float | None) -> None:
    summary = {
        "llf": float(result.llf),
        "nobs": int(result.nobs),
        "converged": bool(result.mle_retvals.get("converged", False)),
        "ncols": ncols,
        "colnames": colnames,
        "sample_frac": sample_frac,
    }
    RESTRICTED_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n  restricted fit summary written to {RESTRICTED_SUMMARY_PATH}")


def _load_restricted_summary() -> dict:
    if not RESTRICTED_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"--skip-restricted was passed but {RESTRICTED_SUMMARY_PATH} "
            "doesn't exist -- run with --restricted-only first."
        )
    return json.loads(RESTRICTED_SUMMARY_PATH.read_text())


def main() -> None:
    args = parse_args()
    if args.restricted_only and args.skip_restricted:
        raise SystemExit("--restricted-only and --skip-restricted are mutually exclusive.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_restricted:
        print(f"Loading cached restricted fit summary from {RESTRICTED_SUMMARY_PATH}...")
        restricted_summary = _load_restricted_summary()
        ll_restricted = restricted_summary["llf"]
        n_obs = restricted_summary["nobs"]
        restricted_ncols = restricted_summary["ncols"]
        restricted_colnames = set(restricted_summary["colnames"])
        cached_sample_frac = restricted_summary.get("sample_frac")
        if cached_sample_frac != args.sample_frac:
            print(
                f"  [warning] restricted fit was built with --sample-frac="
                f"{cached_sample_frac!r}, this run passed {args.sample_frac!r} "
                "-- the row-count check below will catch a real mismatch, "
                "but pass the same value explicitly to avoid relying on that."
            )

        print("\nBuilding unrestricted (fully-interacted) design matrix...")
        train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
        if args.sample_frac is not None:
            print(f"  --sample-frac {args.sample_frac}: subsampling before anything else")
            train = _stratified_sample(train, args.sample_frac, LABEL_COL)
        intermediate = build_chow_design_matrix(train)
        del train
        gc.collect()
    else:
        print("Loading train_model.parquet...")
        train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
        if args.sample_frac is not None:
            print(f"  --sample-frac {args.sample_frac}: subsampling before anything else")
            train = _stratified_sample(train, args.sample_frac, LABEL_COL)
            print(f"  sampled shape: {train.shape}")

        intermediate = build_chow_design_matrix(train)
        del train
        gc.collect()

        print("\nBuilding restricted (pooled) design matrix...")
        restricted_df = build_restricted_design_matrix(intermediate)
        y_restricted, X_restricted = _prepare_xy(restricted_df)
        restricted_ncols = X_restricted.shape[1]
        restricted_colnames = set(X_restricted.columns)
        del restricted_df
        gc.collect()

        restricted_result = _fit_logit(y_restricted, X_restricted, "restricted (pooled)")
        ll_restricted = restricted_result.llf
        n_obs = int(restricted_result.nobs)
        _save_restricted_summary(restricted_result, restricted_ncols, sorted(restricted_colnames), args.sample_frac)

        del X_restricted, y_restricted, restricted_result
        gc.collect()

        if args.restricted_only:
            print("\n--restricted-only: stopping here. Rerun with --skip-restricted to finish the test.")
            return

        print("\nBuilding unrestricted (fully-interacted) design matrix...")

    unrestricted_df, interaction_cols, dropped_zero_variance = add_claim_type_interactions(intermediate)
    del intermediate
    gc.collect()

    y_unrestricted, X_unrestricted = _prepare_xy(unrestricted_df)
    if len(y_unrestricted) != n_obs:
        raise ValueError(
            f"Unrestricted design matrix has {len(y_unrestricted):,} rows "
            f"but the restricted fit used {n_obs:,} -- the two models were "
            "built from different data (e.g. a different --sample-frac "
            "between a --restricted-only and a --skip-restricted run), "
            "which breaks the likelihood-ratio test's nesting assumption. "
            "Rerun both stages with the same --sample-frac (or omit it for "
            "the full data both times)."
        )
    unrestricted_ncols = X_unrestricted.shape[1]
    unrestricted_colnames = set(X_unrestricted.columns)
    del unrestricted_df
    gc.collect()

    unrestricted_result = _fit_logit(y_unrestricted, X_unrestricted, "unrestricted (fully-interacted)")
    ll_unrestricted = unrestricted_result.llf

    del X_unrestricted, y_unrestricted, unrestricted_result
    gc.collect()

    # --- Degrees of freedom, computed and cross-checked ---
    df_chow, df_table = _compute_df_and_table(unrestricted_colnames, restricted_colnames)
    shape_diff = unrestricted_ncols - restricted_ncols
    if df_chow != shape_diff:
        raise ValueError(
            f"Computed df ({df_chow}) does not match the raw column-count "
            f"difference ({shape_diff}) between the two design matrices -- "
            "something about their structure doesn't match this script's "
            "assumptions. Do not trust the LR test until this is resolved."
        )

    n_i_counts = Counter(n_i for _, n_i in df_table)
    breakdown = ", ".join(
        f"{n_covariates} covariate(s) with n_i={n_i} (contributing {n_i - 1} df each)"
        for n_i, n_covariates in sorted(n_i_counts.items())
    )

    # --- The test itself ---
    LR = -2 * (ll_restricted - ll_unrestricted)
    p_value = stats.chi2.sf(LR, df_chow)

    lines = [
        "Chow test -- claim-type homogeneity of shared covariates",
        "=" * 60,
        f"n obs:                               {n_obs:,}",
        f"restricted (pooled) columns:         {restricted_ncols}",
        f"unrestricted (interacted) columns:   {unrestricted_ncols}",
        f"degrees of freedom (Sigma n_i - 1):  {df_chow}  [{breakdown}]",
        "",
        f"log-likelihood, restricted:    {ll_restricted:.4f}",
        f"log-likelihood, unrestricted:  {ll_unrestricted:.4f}",
        f"LR statistic:                  {LR:.4f}",
        f"p-value (chi2, df={df_chow}):   {p_value:.6g}",
        "",
        f"H0 (pooled coefficients adequate for every shared covariate) "
        f"{'REJECTED' if p_value < 0.05 else 'NOT REJECTED'} at alpha=0.05",
        "",
        "Reminder: a rejection means at least one shared covariate has a "
        "real claim-type interaction -- it does NOT mean every shared "
        "covariate does. Stage 2 (per-variable testing) localizes which "
        "ones, per FEATURE_ENGINEERING.md Section 3.",
        "",
        f"Interaction terms skipped for zero variance (0 df, correctly "
        f"excluded from the hypothesis): {len(dropped_zero_variance)}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_PATH.write_text(report + "\n")
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
