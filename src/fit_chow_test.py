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
step (FEATURE_ENGINEERING.md Section 6, 2026-09-22 addendum) -- on an
8 GB machine that already hit its memory ceiling several separate times
building just the encodings. statsmodels needs its own working memory on
top of holding one matrix (gradient/line-search arrays roughly the size
of the exog array itself), so this script:
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
  - CORRECTED 2026-09-23: --sample-frac originally loaded the ENTIRE
    ~1.15M-row / 192-column train_model.parquet into memory before
    throwing away (1-frac) of it -- completely defeating the point of a
    "low-memory smoke test" flag, and a real pyarrow.lib.ArrowMemoryError
    was hit doing exactly that at --sample-frac 0.2. _load_train()/
    _stratified_sample_streaming() below now pick the sampled row indices
    from a single-column (label-only) pass, then stream the rest of the
    file in --stream-batch-size-row batches via pyarrow's iter_batches(),
    filtering each batch down to its kept rows before converting more
    than one batch's worth to pandas at a time.

KNOWN, CONFIRMED SEPARATION (2026-09-23): fully interacting many rare
one-hot categories against DME (the smallest claim type) was originally
flagged as a theoretical risk; it turned out to be real and larger than
expected. diagnose_separation.py and verify_provider_outlier_mechanism.py
traced it to a genuine, confirmed, population-wide zero: 6 HCPCS codes
(94010, 96156, 99401, 99408, 99495, M1069) have an EXACT 0.000 denial
rate within claim_type=carrier across the full ~1.15M-row population (up
to 45,788 claims for one code alone). Mechanism, confirmed directly from
denial_rules.py: 4 of 6 risk factors are structurally excluded for these
codes by design (dx_procedure_mismatch only fires for its explicitly-
mapped codes -- these are documented as "deliberately left unmapped";
missing_prior_auth only for E/K-prefix codes; missing_hcpcs only when
HCPCS is absent; deprecated_code only for the 10 specific consult codes),
and PRVDR_NUM is NaN for these specific claims, which pandas'
groupby(...).size() silently drops by default -- meaning these claims
were never even a CANDIDATE for rule_provider_outlier's volume-outlier
flag, not evaluated and found low-volume. duplicate_claim's independent
dataset-wide rarity covers the remainder. This produced bit-identical
log-likelihoods between the restricted and unrestricted fits (LR=0,
p=1) at every sample size tried, from 23,039 rows up to the full
population, with exp-overflow/log-divide-by-zero/HessianInversionWarning
on both fits.

TWO FIXES, BOTH IMPLEMENTED, MEANT TO BE COMPARED (2026-09-23):
  --method firth: Firth's penalized (bias-reduced) logistic regression
      instead of ordinary MLE (see _fit_logit_firth's docstring for the
      implementation and, importantly, its asymptotic caveat -- the
      chi-square justification for a penalized LR test under TRUE
      separation is empirically well-validated but not as rigorously
      settled as the ordinary LRT's).
  --exclude-separating-codes: drop the 6 confirmed columns entirely from
      both matrices before fitting (see _SEPARATING_HCPCS_CODES below).
      With no separation left in the design, the ordinary --method
      standard LRT needs no caveat at all -- classic Wilks asymptotics
      apply cleanly.
  Run both (--method firth alone, and --method standard
  --exclude-separating-codes together) and compare the saved reports --
  output filenames are suffixed by method/exclusion (see _output_paths)
  specifically so the two runs don't overwrite each other.

Run with (from the repo root, inside the venv):
    python src/fit_chow_test.py
    python src/fit_chow_test.py --sample-frac 0.02      # smoke test first
    python src/fit_chow_test.py --restricted-only       # stage 1 of 2
    python src/fit_chow_test.py --skip-restricted       # stage 2 of 2
    python src/fit_chow_test.py --method firth
    python src/fit_chow_test.py --exclude-separating-codes
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy import stats

from build_chow_design_matrix import (
    PROCESSED_DIR,
    add_claim_type_interactions,
    build_chow_design_matrix,
    build_restricted_design_matrix,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
TRAIN_PARQUET_PATH = PROCESSED_DIR / "train_model.parquet"

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

# --- Columns not yet given ANY Phase 2 encoding decision -----------------
# Confirmed present in real train_model.parquet by running this script and
# hitting a real ValueError (2026-09-23) -- not a hardcoded guess made in
# advance. Neither build_features.py nor build_chow_design_matrix.py drops
# or encodes these: build_chow_design_matrix.py only ever touches (1) the
# 5 named cardinality covariates, (2) numeric claim-type-exclusive/2-of-3
# fields, and (3) the specific dummy-prefix groups -- everything else rides
# through both design matrices completely untouched.
#
# Excluded from the Phase 2 BASELINE/Chow-test feature set HERE, not from
# train_model.parquet itself -- XGBoost (Phase 2's other model) can use or
# encode several of these natively and shouldn't inherit a baseline-
# specific exclusion decision.

# A. Already documented elsewhere as fixed/near-constant, carrying zero
#    signal.
_PENDING_CONSTANT = {
    "CARR_CLM_PMT_DNL_CD", "CLM_DISP_CD", "CLM_MDCR_NON_PMT_RSN_CD",
}

# B. Raw dates with no Phase 2 numeric transform yet.
_PENDING_DATES = {
    "LINE_1ST_EXPNS_DT", "LINE_LAST_EXPNS_DT", "REV_CNTR_DT",
    *(f"PRCDR_DT{i}" for i in range(1, 25)),
}

# C. Legacy/secondary provider-identifier strings beyond the 6 NPI fields
#    FEATURE_ENGINEERING.md Section 1 already resolved.
_PENDING_LEGACY_IDS = {
    "RFR_PHYSN_UPIN", "PRF_PHYSN_UPIN", "AT_PHYSN_UPIN", "OP_PHYSN_UPIN",
    "RNDRNG_PHYSN_UPIN", "CARR_CLM_RFRNG_PIN_NUM", "CARR_PRFRNG_PIN_NUM",
    "CARR_CLM_BLG_NPI_NUM", "ORG_NPI_NUM", "TAX_NUM",
}

# D. Secondary/tertiary diagnosis & procedure codes.
_PENDING_SECONDARY_CODES = {
    *(f"ICD_DGNS_CD{i}" for i in range(1, 26)),
    *(f"ICD_DGNS_E_CD{i}" for i in range(1, 13)),
    "FST_DGNS_E_CD",
    *(f"ICD_PRCDR_CD{i}" for i in range(1, 25)),
    "LINE_ICD_DGNS_CD", "LINE_ICD_DGNS_VRSN_CD",
}

# E. Other CMS categorical/indicator code fields with no cardinality check
#    or encoding decision made yet.
_PENDING_OTHER_CODES = {
    "CARR_CLM_ENTRY_CD", "CARR_CLM_PRVDR_ASGNMT_IND_SW",
    "CARR_CLM_HCPCS_YR_CD", "CARR_LINE_PRVDR_TYPE_CD", "PRTCPTNG_IND_CD",
    "LINE_PLACE_OF_SRVC_CD", "CARR_LINE_PRCNG_LCLTY_CD", "BETOS_CD",
    "LINE_BENE_PRMRY_PYR_CD", "LINE_PRCSG_IND_CD", "HPSA_SCRCTY_IND_CD",
    "LINE_HCT_HGB_TYPE_CD", "CARR_LINE_CLIA_LAB_NUM", "CLM_FAC_TYPE_CD",
    "CLM_SRVC_CLSFCTN_TYPE_CD", "CLM_FREQ_CD", "NCH_PRMRY_PYR_CD",
    "PTNT_DSCHRG_STUS_CD", "REV_CNTR_PMT_MTHD_IND_CD",
    "REV_CNTR_STUS_IND_CD", "DMERC_LINE_PRCNG_STATE_CD",
    "DMERC_LINE_SUPPLR_TYPE_CD", "DMERC_LINE_MTUS_CD",
}

# F. Sequence/line-position numbers -- not real predictive features.
_PENDING_SEQUENCE_NUMS = {"LINE_NUM", "CLM_LINE_NUM"}

# G. Numeric-sounding but object dtype -- a real data-quality question.
_PENDING_DATA_QUALITY = {"LINE_HCT_HGB_RSLT_NUM"}

_PENDING_ENCODING_COLS = (
    _PENDING_CONSTANT
    | _PENDING_DATES
    | _PENDING_LEGACY_IDS
    | _PENDING_SECONDARY_CODES
    | _PENDING_OTHER_CODES
    | _PENDING_SEQUENCE_NUMS
    | _PENDING_DATA_QUALITY
)

_NON_FEATURE_COLS = _ID_LABEL_COLS | _DATE_COLS_NOT_YET_FEATURIZED | _PENDING_ENCODING_COLS

_DROP_GROUPS: list[tuple[str, set[str]]] = [
    ("id/label", _ID_LABEL_COLS),
    ("dates (no Phase 2 transform)", _DATE_COLS_NOT_YET_FEATURIZED | _PENDING_DATES),
    ("pending -- fixed/constant (candidate for build_features.py DROP_COLUMNS)", _PENDING_CONSTANT),
    ("pending -- legacy provider IDs (UPIN/PIN/2nd-tier NPI/tax)", _PENDING_LEGACY_IDS),
    ("pending -- secondary diagnosis/procedure codes", _PENDING_SECONDARY_CODES),
    ("pending -- other CMS categorical/indicator codes", _PENDING_OTHER_CODES),
    ("pending -- sequence/line-position numbers", _PENDING_SEQUENCE_NUMS),
    ("pending -- data-quality question (numeric-looking, object dtype)", _PENDING_DATA_QUALITY),
]

# --- Confirmed structurally-separating HCPCS codes (2026-09-23) ----------
# Exact 0.000 denial rate within claim_type=carrier across the FULL
# population (up to 45,788 claims for one code) -- confirmed via
# diagnose_separation.py Steps 3/4 and verify_provider_outlier_mechanism.py.
# See the module docstring's "KNOWN, CONFIRMED SEPARATION" section for the
# full mechanism. Used by --exclude-separating-codes below.
_SEPARATING_HCPCS_CODES = {"94010", "96156", "99401", "99408", "99495", "M1069"}


def _separating_hcpcs_columns(colnames: set[str]) -> set[str]:
    """Every column derived from a confirmed structurally-separating HCPCS
    code that's actually present in `colnames` -- the plain pooled column
    in the restricted matrix ('hcpcs_<code>') and/or every interaction
    term in the unrestricted matrix ('hcpcs_<code>__x__<claim_type>').
    Only returns what's actually there, so this is safe to call on either
    matrix (they have different column shapes for the same covariate) or
    even if a code's zero-variance-guarded interaction terms mean fewer
    columns exist than expected."""
    return {
        c for c in colnames
        if any(c == f"hcpcs_{code}" or c.startswith(f"hcpcs_{code}__x__") for code in _SEPARATING_HCPCS_CODES)
    }


class _SimpleFitResult:
    """Minimal stand-in for a statsmodels fit result, so main() can treat
    a Firth fit and a statsmodels fit identically -- both just need .llf,
    .nobs, and .mle_retvals.get(...)."""

    def __init__(self, llf: float, nobs: int, converged, n_iter=None):
        self.llf = llf
        self.nobs = nobs
        self.mle_retvals = {"converged": converged, "iterations": n_iter}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the Chow test (restricted vs. unrestricted logistic "
        "regression) over the claim-type-interaction design matrices."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Stream a stratified (on is_denied) subsample of this fraction "
        "of train_model.parquet, without ever loading the full file into "
        "memory. Omit for the full run. Must match between a "
        "--restricted-only run and the --skip-restricted run that follows "
        "it, or the two fits won't be nested over the same rows.",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=50_000,
        help="Rows per pyarrow batch when streaming a --sample-frac subsample "
        "(default 50,000). No effect when --sample-frac is omitted.",
    )
    parser.add_argument(
        "--restricted-only",
        action="store_true",
        help="Fit only the restricted (pooled) model, write its summary, and "
        "exit. Use this first given the real memory risk of building+fitting "
        "both matrices in one process -- then rerun with --skip-restricted "
        "for the second half.",
    )
    parser.add_argument(
        "--skip-restricted",
        action="store_true",
        help="Load a previously-saved restricted fit summary instead of "
        "rebuilding and refitting the restricted model. Use after a prior "
        "--restricted-only run with the SAME --method/--exclude-separating-"
        "codes/--sample-frac -- mismatches are checked and warned about.",
    )
    parser.add_argument(
        "--method",
        choices=["standard", "firth"],
        default="standard",
        help="'standard': ordinary MLE via statsmodels (method='lbfgs') -- "
        "degenerate (identical log-likelihoods, LR=0) under the confirmed "
        "separation unless combined with --exclude-separating-codes. "
        "'firth': Firth's penalized (bias-reduced) logistic regression via "
        "the firthmodels package (pip install firthmodels) -- handles "
        "separation directly, at the cost of a less rigorously-settled "
        "asymptotic justification for the resulting LR test (see "
        "_fit_logit_firth's docstring). Run both this and --method standard "
        "--exclude-separating-codes, and compare the saved reports.",
    )
    parser.add_argument(
        "--exclude-separating-codes",
        action="store_true",
        help="Drop the 6 confirmed structurally-separating HCPCS codes "
        "(94010, 96156, 99401, 99408, 99495, M1069 -- see module docstring) "
        "entirely from both design matrices before fitting. With this set, "
        "--method standard should no longer show degenerate log-likelihoods, "
        "and the resulting LR test needs no asymptotic caveat at all.",
    )
    return parser.parse_args()


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Results/summary file paths, suffixed by --method and
    --exclude-separating-codes -- the whole point of running more than one
    combination is to compare their saved output, so they must not
    overwrite each other."""
    suffix = args.method + ("_excl" if args.exclude_separating_codes else "")
    return (
        REPORTS_DIR / f"chow_test_results__{suffix}.txt",
        REPORTS_DIR / f"chow_restricted_fit_summary__{suffix}.json",
    )


def _stratified_sample_streaming(
    parquet_path: Path, frac: float, label_col: str, batch_size: int, seed: int = 42
) -> pd.DataFrame:
    """Pick a stratified sample of row positions from a SINGLE-COLUMN
    (label-only) read of the parquet file, then stream the rest of the
    file in `batch_size`-row pyarrow batches via iter_batches(), filtering
    each batch down to just its kept rows before converting more than one
    batch's worth to pandas at a time. Deterministic given the same file,
    frac, and seed."""
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"    {parquet_path.name}: {total_rows:,} rows across {pf.num_row_groups} row group(s)")
    if pf.num_row_groups <= 2:
        print(
            "    [note] very few row groups -- pyarrow decompresses at "
            "row-group granularity internally regardless of batch_size, so "
            "streaming still helps at the pandas-conversion step but may not "
            "fully avoid a crash if a single row group alone is too large."
        )

    label_values = pf.read(columns=[label_col]).column(label_col).to_pandas().to_numpy()
    if len(label_values) != total_rows:
        raise ValueError(
            f"Label column read back {len(label_values):,} values but the "
            f"file's own metadata reports {total_rows:,} rows -- do not "
            "trust the sample until this is investigated."
        )
    rng = np.random.RandomState(seed)
    keep_mask = np.zeros(total_rows, dtype=bool)
    kept_counts = {}
    for val in np.unique(label_values):
        idx = np.flatnonzero(label_values == val)
        n_keep = min(int(round(len(idx) * frac)), len(idx))
        keep_mask[rng.choice(idx, size=n_keep, replace=False)] = True
        kept_counts[val] = (n_keep, len(idx))
    del label_values
    gc.collect()
    print(
        "    stratified target: "
        + ", ".join(f"{label_col}={v}: {k}/{n}" for v, (k, n) in sorted(kept_counts.items()))
    )

    parts = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n = batch.num_rows
        batch_mask = keep_mask[row_offset : row_offset + n]
        row_offset += n
        if batch_mask.any():
            parts.append(batch.to_pandas().loc[batch_mask].copy())
        del batch

    if row_offset != total_rows:
        raise ValueError(
            f"Streamed {row_offset:,} rows via iter_batches but the file's "
            f"metadata reports {total_rows:,} -- do not trust the sample "
            "until this is investigated."
        )

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"    kept {len(result):,} of {total_rows:,} rows (shape: {result.shape})")
    return result


def _load_train(sample_frac: float | None, stream_batch_size: int) -> pd.DataFrame:
    """Load train_model.parquet, or a streamed stratified subsample of it."""
    if sample_frac is None:
        print("Loading train_model.parquet (full)...")
        return pd.read_parquet(TRAIN_PARQUET_PATH)
    print(f"Streaming a --sample-frac {sample_frac} stratified subsample of train_model.parquet...")
    return _stratified_sample_streaming(TRAIN_PARQUET_PATH, sample_frac, LABEL_COL, stream_batch_size)


def _prepare_xy(design_df: pd.DataFrame, exclude_separating: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Split a design matrix into (y, X): drop ID/label/not-yet-encoded
    columns, printing which group each dropped column belongs to, then
    fail loudly on anything LEFT OVER. When exclude_separating is True,
    also drops the confirmed structurally-separating HCPCS columns (see
    _SEPARATING_HCPCS_CODES) -- computed fresh per call since the
    restricted and unrestricted matrices have different column shapes for
    the same covariate (one plain column vs. one-or-more interaction
    terms)."""
    present_non_feature = _NON_FEATURE_COLS & set(design_df.columns)
    drop_groups = list(_DROP_GROUPS)
    if exclude_separating:
        sep_cols = _separating_hcpcs_columns(set(design_df.columns))
        drop_groups.append(("EXCLUDED -- confirmed structurally-separating HCPCS codes", sep_cols))
        present_non_feature = present_non_feature | sep_cols

    print(f"    dropping {len(present_non_feature)} non-feature column(s) from X:")
    for group_label, group_cols in drop_groups:
        present_in_group = sorted(present_non_feature & group_cols)
        if present_in_group:
            print(f"      {group_label} ({len(present_in_group)}): {present_in_group}")

    y = design_df[LABEL_COL].astype(int)
    X = design_df.drop(columns=list(present_non_feature))

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            "Non-numeric column(s) leaked into the design matrix, NOT "
            "accounted for by any of the _PENDING_* groups above -- this "
            "is a genuinely new gap, investigate before adding it to a "
            f"group or fixing it upstream: {non_numeric}"
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


def _fit_logit_standard(y: pd.Series, X: pd.DataFrame, label: str) -> _SimpleFitResult | object:
    """Fit one Logit model via ordinary MLE. method='lbfgs', not
    statsmodels' Newton-Raphson default: full interaction against many
    rare one-hot categories creates real near-singularity risk for some
    coefficients, and LBFGS degrades more gracefully than Newton's direct
    Hessian inversion when that happens. Passing X as a DataFrame (not
    pre-cast to a numpy array) so statsmodels does the float64 upcast
    exactly once, itself."""
    print(f"\n  fitting {label} (standard MLE, lbfgs): {X.shape[0]:,} rows x {X.shape[1]} columns")
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


def _fit_logit_firth(y: pd.Series, X: pd.DataFrame, label: str) -> _SimpleFitResult:
    """Fit via Firth's penalized (bias-reduced) logistic regression instead
    of ordinary MLE -- the standard statistical fix for the separation
    confirmed in diagnose_separation.py / verify_provider_outlier_mechanism.py.
    Uses the `firthmodels` package (pip install firthmodels) -- the
    successor to `firthlogist`, which was archived by its author (Dec
    2025) and pulled from PyPI entirely. This function was written and
    verified against firthmodels 0.8.2's ACTUAL installed API (checked via
    inspect.signature() and the class docstring directly, 2026-09-23 --
    not assumed from search results, after firthlogist's disappearance
    was itself a lesson in not trusting an unverified package name/API).

    fit_intercept=False (confirmed real constructor parameter): our
    design has NO shared intercept by construction -- the
    claim_type_carrier/outpatient/dme dummies (cell-means coding) serve
    that role. An automatically added intercept would break that
    structure and make the restricted/unrestricted models differ in more
    than just the tested covariates.

    No wald=True or similar flag needed here (unlike the now-defunct
    firthlogist): firthmodels' .fit() only computes the cheap Wald-based
    bse_/pvalues_ automatically. The expensive per-coefficient
    profile-likelihood computation (lrt_pvalues_/lrt_bse_) lives behind a
    SEPARATE .lrt() method call, which this function never makes -- so
    there's no risk of the catastrophic per-parameter refitting that
    would be infeasible at the 140-310 parameters this design has.

    max_iter raised from the package default (25) to 100: Newton-Raphson
    converges quadratically near the optimum so 25 is often enough, but
    given this design's history of convergence trouble under separation, a
    larger safety margin costs little (each iteration is one Newton step,
    not a full re-fit).

    IMPORTANT CAVEAT, repeat this wherever these results get used: the
    chi-square justification for a PENALIZED likelihood-ratio test is
    well-established AWAY from separation (Firth's correction term is
    O(1) against an O(n) log-likelihood, so it vanishes asymptotically and
    standard Wilks theory applies in the well-behaved regime) but is NOT a
    rigorously-proven exact result under TRUE separation -- the general
    theory of LR tests near a parameter-space boundary (Self & Liang 1987;
    the chi-bar-squared literature) shows the null distribution can be a
    MIXTURE of chi-squares with reduced effective df in boundary cases,
    and Firth's penalty being asymptotically negligible doesn't repair
    that. Firth + penalized-LRT's good behavior under separation (Heinze &
    Schemper 2002) is validated empirically via simulation, not proven as
    an exact asymptotic result. The --exclude-separating-codes run has no
    separation left at all and needs no such caveat -- compare the two.
    """
    from firthmodels import FirthLogisticRegression

    print(f"\n  fitting {label} (Firth's penalized MLE via firthmodels): {X.shape[0]:,} rows x {X.shape[1]} columns")
    print("    this can be slower than the standard lbfgs fit -- watch for a long run.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FirthLogisticRegression(fit_intercept=False, max_iter=100)
        model.fit(X, y)
    for w in caught:
        print(f"    [warning] {w.category.__name__}: {w.message}")

    llf = float(model.loglik_)
    n_iter = int(model.n_iter_)
    converged = bool(model.converged_)
    print(f"    converged: {converged}")
    print(f"    penalized log-likelihood: {llf:.4f}")
    print(f"    Newton-Raphson iterations: {n_iter}")
    print(f"    nobs: {len(y):,}")
    return _SimpleFitResult(llf=llf, nobs=len(y), converged=converged, n_iter=n_iter)


def _fit_logit(y: pd.Series, X: pd.DataFrame, label: str, method: str):
    if method == "firth":
        return _fit_logit_firth(y, X, label)
    return _fit_logit_standard(y, X, label)


def _compute_df_and_table(
    unrestricted_colnames: set[str], restricted_colnames: set[str]
) -> tuple[int, list[tuple[str, int]]]:
    """df = Sigma_i (n_i - 1), derived directly from the two matrices'
    actual column names. A genuinely-shared covariate is exactly a base
    name that appears as one or more '<base>__x__<claim_type>' columns
    ONLY in the unrestricted set and as a single plain '<base>' column
    ONLY in the restricted set. Raises loudly on anything that doesn't
    fit that pattern, rather than silently mis-summing df."""
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


def _save_restricted_summary(
    path: Path, result, ncols: int, colnames: list[str], sample_frac: float | None, method: str, exclude_separating: bool
) -> None:
    summary = {
        "llf": float(result.llf),
        "nobs": int(result.nobs),
        "converged": result.mle_retvals.get("converged", False),
        "ncols": ncols,
        "colnames": colnames,
        "sample_frac": sample_frac,
        "method": method,
        "exclude_separating_codes": exclude_separating,
    }
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n  restricted fit summary written to {path}")


def _load_restricted_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"--skip-restricted was passed but {path} doesn't exist -- run "
            "with --restricted-only (with the SAME --method/"
            "--exclude-separating-codes) first."
        )
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    if args.restricted_only and args.skip_restricted:
        raise SystemExit("--restricted-only and --skip-restricted are mutually exclusive.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path, restricted_summary_path = _output_paths(args)

    if args.skip_restricted:
        print(f"Loading cached restricted fit summary from {restricted_summary_path}...")
        restricted_summary = _load_restricted_summary(restricted_summary_path)
        ll_restricted = restricted_summary["llf"]
        n_obs = restricted_summary["nobs"]
        restricted_ncols = restricted_summary["ncols"]
        restricted_colnames = set(restricted_summary["colnames"])
        for field, current in (("sample_frac", args.sample_frac), ("method", args.method), ("exclude_separating_codes", args.exclude_separating_codes)):
            cached = restricted_summary.get(field)
            if cached != current:
                print(
                    f"  [warning] restricted fit was built with {field}={cached!r}, "
                    f"this run passed {current!r} -- results will not be comparable "
                    "unless these match. The row-count check below catches a "
                    "--sample-frac mismatch specifically, but not a --method or "
                    "--exclude-separating-codes mismatch."
                )

        print("\nBuilding unrestricted (fully-interacted) design matrix...")
        train = _load_train(args.sample_frac, args.stream_batch_size)
        intermediate = build_chow_design_matrix(train)
        del train
        gc.collect()
    else:
        train = _load_train(args.sample_frac, args.stream_batch_size)

        intermediate = build_chow_design_matrix(train)
        del train
        gc.collect()

        print("\nBuilding restricted (pooled) design matrix...")
        restricted_df = build_restricted_design_matrix(intermediate)
        y_restricted, X_restricted = _prepare_xy(restricted_df, exclude_separating=args.exclude_separating_codes)
        restricted_ncols = X_restricted.shape[1]
        restricted_colnames = set(X_restricted.columns)
        del restricted_df
        gc.collect()

        restricted_result = _fit_logit(y_restricted, X_restricted, "restricted (pooled)", args.method)
        ll_restricted = restricted_result.llf
        n_obs = int(restricted_result.nobs)
        _save_restricted_summary(
            restricted_summary_path, restricted_result, restricted_ncols, sorted(restricted_colnames),
            args.sample_frac, args.method, args.exclude_separating_codes,
        )

        del X_restricted, y_restricted, restricted_result
        gc.collect()

        if args.restricted_only:
            print("\n--restricted-only: stopping here. Rerun with --skip-restricted to finish the test.")
            return

        print("\nBuilding unrestricted (fully-interacted) design matrix...")

    unrestricted_df, interaction_cols, dropped_zero_variance = add_claim_type_interactions(intermediate)
    del intermediate
    gc.collect()

    y_unrestricted, X_unrestricted = _prepare_xy(unrestricted_df, exclude_separating=args.exclude_separating_codes)
    if len(y_unrestricted) != n_obs:
        raise ValueError(
            f"Unrestricted design matrix has {len(y_unrestricted):,} rows "
            f"but the restricted fit used {n_obs:,} -- the two models were "
            "built from different data, which breaks the likelihood-ratio "
            "test's nesting assumption. Rerun both stages with the same "
            "--sample-frac (or omit it for the full data both times)."
        )
    unrestricted_ncols = X_unrestricted.shape[1]
    unrestricted_colnames = set(X_unrestricted.columns)
    del unrestricted_df
    gc.collect()

    unrestricted_result = _fit_logit(y_unrestricted, X_unrestricted, "unrestricted (fully-interacted)", args.method)
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

    method_desc = (
        "Firth's penalized MLE (firthmodels, fit_intercept=False, max_iter=100)"
        if args.method == "firth"
        else "ordinary MLE (statsmodels, method='lbfgs')"
    )
    lines = [
        "Chow test -- claim-type homogeneity of shared covariates",
        "=" * 60,
        f"Method:                              {method_desc}",
        f"Separating-code exclusion:           {'ON -- dropped ' + str(sorted(_SEPARATING_HCPCS_CODES)) if args.exclude_separating_codes else 'OFF'}",
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
        "",
        f"NOTE: {len(_PENDING_ENCODING_COLS)} raw column(s) (secondary "
        "diagnosis/procedure codes, legacy provider IDs, unaddressed "
        "CMS category codes, and 3 fixed-constant fields) were excluded "
        "from this baseline's feature set pending a real Phase 2 encoding "
        "decision -- see _PENDING_* groups in this file's source.",
    ]
    if args.method == "firth":
        lines += [
            "",
            "CAVEAT (Firth method): the chi-square approximation used for the",
            "p-value above is well-established away from separation, but under",
            "TRUE separation (as confirmed for 6 HCPCS codes -- see module",
            "docstring) it is a widely-used, empirically-validated-via-simulation",
            "practical approach (Heinze & Schemper 2002), not a rigorously proven",
            "exact asymptotic result -- LR tests near a parameter-space boundary",
            "can follow a mixture of chi-squares rather than a single one (Self &",
            "Liang 1987). Compare against the --exclude-separating-codes run,",
            "which has no separation left and needs no such caveat.",
        ]
    report = "\n".join(lines)
    print("\n" + report)
    results_path.write_text(report + "\n")
    print(f"\nWritten to {results_path}")


if __name__ == "__main__":
    main()
