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
    was hit doing exactly that at --sample-frac 0.2 (FreePhysicalMemory
    checked at ~2.77 GB at the time; the failed allocation itself was
    only ~2 MB, consistent with the earlier full-file load having already
    pushed pyarrow's own allocator close to its ceiling, not a literal
    out-of-memory condition). _load_train()/_stratified_sample_streaming()
    below now pick the sampled row indices from a single-column
    (label-only) pass, then stream the rest of the file in
    --stream-batch-size-row batches via pyarrow's iter_batches(),
    filtering each batch down to its kept rows before converting more
    than one batch's worth to pandas at a time -- so the full file is
    never materialized as one in-memory frame at all. NOTE: pandas
    writes a parquet file as very few (sometimes one) row groups by
    default, and pyarrow's Parquet reader decompresses at row-group
    granularity internally regardless of iter_batches' batch_size --
    so this reduces PEAK memory during the pandas-conversion step even
    on a coarse-row-group file, but doesn't fully avoid a single-row-
    group file needing to be decompressed as one Arrow-level chunk. The
    function prints the file's actual row-group count so this can be
    checked directly rather than assumed; if it still crashes on a file
    with very few row groups, the real fix is rewriting
    train_model.parquet with a smaller row_group_size (a
    build_features.py-side change, out of this script's scope).

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
fit output next, not treated as resolved. A real occurrence of this was
observed at --sample-frac 0.02 (23,039 rows): both fits converged to
IDENTICAL log-likelihoods with exp-overflow / log-divide-by-zero /
HessianInversionWarning on both -- ~9 events per parameter at that sample
size, well under the standard 10-20-events-per-variable rule of thumb.
Not yet confirmed whether this clears at a larger sample.

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
RESULTS_PATH = REPORTS_DIR / "chow_test_results.txt"
RESTRICTED_SUMMARY_PATH = REPORTS_DIR / "chow_restricted_fit_summary.json"
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
# through both design matrices completely untouched. That file's own
# comment anticipated "82 non-numeric columns... never in scope for this
# treatment", but that estimate (from a 2026-09-17 check of only the
# columns carrying real NaN) undercounts the true list once every
# non-numeric column is checked -- the actual count is 130. Same
# "the real number was bigger than the estimate" pattern that's recurred
# throughout this project's audits (the 2-of-3 count, the 0-of-3 count,
# the NPI-flag table).
#
# Excluded from the Phase 2 BASELINE/Chow-test feature set HERE, not from
# train_model.parquet itself -- XGBoost (Phase 2's other model) can use or
# encode several of these natively and shouldn't inherit a baseline-
# specific exclusion decision. This is a scope decision for getting the
# Chow test running now, not a permanent verdict on each column -- grouped
# below so the ones genuinely worth an encoding pass later (Category D
# especially) aren't confused with the ones that are dead weight.

# A. Already documented elsewhere as fixed/near-constant, carrying zero
#    signal -- TARGET_DEFINITION.md's "why not a native denial field"
#    table uses these three to argue no native denial field exists, but
#    nothing ever added them to build_features.py's DROP_COLUMNS as
#    FEATURES. Worth moving there permanently (it's the shared file
#    XGBoost also reads) rather than excluding only here -- flagged, not
#    done unilaterally.
_PENDING_CONSTANT = {
    "CARR_CLM_PMT_DNL_CD", "CLM_DISP_CD", "CLM_MDCR_NON_PMT_RSN_CD",
}

# B. Raw dates with no Phase 2 numeric transform yet -- same reasoning as
#    _DATE_COLS_NOT_YET_FEATURIZED above, a longer list than originally
#    known.
_PENDING_DATES = {
    "LINE_1ST_EXPNS_DT", "LINE_LAST_EXPNS_DT", "REV_CNTR_DT",
    *(f"PRCDR_DT{i}" for i in range(1, 25)),
}

# C. Legacy/secondary provider-identifier strings beyond the 6 NPI fields
#    FEATURE_ENGINEERING.md Section 1 already resolved (drop raw value,
#    keep a presence flag) -- UPIN (the pre-NPI legacy identifier), PIN,
#    and a second tier of NPI/tax-ID fields that discussion never covered.
_PENDING_LEGACY_IDS = {
    "RFR_PHYSN_UPIN", "PRF_PHYSN_UPIN", "AT_PHYSN_UPIN", "OP_PHYSN_UPIN",
    "RNDRNG_PHYSN_UPIN", "CARR_CLM_RFRNG_PIN_NUM", "CARR_PRFRNG_PIN_NUM",
    "CARR_CLM_BLG_NPI_NUM", "ORG_NPI_NUM", "TAX_NUM",
}

# D. Secondary/tertiary diagnosis & procedure codes -- only the PRINCIPAL
#    diagnosis (PRNCPAL_DGNS_CD) got cardinality encoding in
#    build_chow_design_matrix.py; these (up to 25 additional diagnoses, 12
#    external-cause-of-injury codes, 24 additional procedures per claim)
#    never did. Real clinical signal plausibly lives here -- the one
#    category most worth a genuine encoding pass later, not a permanent
#    drop. XGBoost can likely use a cheaper representation (e.g.
#    presence-of-any-code, or clinically-grouped flags) than the
#    one-hot-per-code treatment a linear baseline would need.
_PENDING_SECONDARY_CODES = {
    *(f"ICD_DGNS_CD{i}" for i in range(1, 26)),
    *(f"ICD_DGNS_E_CD{i}" for i in range(1, 13)),
    "FST_DGNS_E_CD",
    *(f"ICD_PRCDR_CD{i}" for i in range(1, 25)),
    "LINE_ICD_DGNS_CD", "LINE_ICD_DGNS_VRSN_CD",
}

# E. Other CMS categorical/indicator code fields with no cardinality check
#    or encoding decision made yet -- each would need its own review (some
#    are likely low-cardinality and cheap to one-hot, e.g.
#    LINE_PLACE_OF_SRVC_CD; others may turn out constant, like Category A).
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

# G. Numeric-sounding but object dtype -- a real data-quality question
#    (probably a non-numeric sentinel value in a lab-result field), worth
#    checking on its own; excluded here rather than guessed at.
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

# Labeled groups, for the printed breakdown in _prepare_xy -- so a run's
# console output states clearly WHICH kind of column is being dropped and
# why, rather than one undifferentiated list.
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
        "memory (see module docstring's MEMORY section) -- for a fast, "
        "low-memory correctness check of this whole script before "
        "committing to the full ~1.15M-row fit. Omit for the full run. "
        "Must match between a --restricted-only run and the "
        "--skip-restricted run that follows it, or the two fits won't be "
        "nested over the same rows.",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=50_000,
        help="Rows per pyarrow batch when streaming a --sample-frac subsample "
        "(default 50,000). Lower this if the file's row-group layout is "
        "coarse enough that streaming still uses too much memory at the "
        "default (this script prints the file's row-group count so you "
        "can tell). No effect when --sample-frac is omitted.",
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


def _stratified_sample_streaming(
    parquet_path: Path, frac: float, label_col: str, batch_size: int, seed: int = 42
) -> pd.DataFrame:
    """Pick a stratified sample of row positions from a SINGLE-COLUMN
    (label-only) read of the parquet file -- a few MB even at ~1.15M rows
    -- then stream the rest of the file in `batch_size`-row pyarrow
    batches via iter_batches(), filtering each batch down to just its
    kept rows before converting more than one batch's worth to pandas at
    a time. The full file is never materialized as one in-memory
    DataFrame. See the module docstring's MEMORY section for why this
    replaced a plain pd.read_parquet() + df.sample() -- that combination
    hit a real pyarrow.lib.ArrowMemoryError at --sample-frac 0.2 on this
    machine.

    Deterministic given the same file, frac, and seed -- required so a
    --restricted-only run and the --skip-restricted run that follows it
    see the identical row set (both stages call this function fresh, each
    reading the file from scratch).
    """
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"    {parquet_path.name}: {total_rows:,} rows across {pf.num_row_groups} row group(s)")
    if pf.num_row_groups <= 2:
        print(
            "    [note] very few row groups -- pyarrow decompresses at "
            "row-group granularity internally regardless of batch_size, so "
            "streaming still helps at the pandas-conversion step but may not "
            "fully avoid a crash if a single row group alone is too large. "
            "If this still fails, rewrite train_model.parquet with a "
            "smaller row_group_size (a build_features.py-side change)."
        )

    # Step 1: the one unavoidable full-file pass -- label column only.
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

    # Step 2: stream in batches, filtering each down to its kept rows
    # before it ever becomes a full-size pandas DataFrame.
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
    """Load train_model.parquet, or a streamed stratified subsample of it.
    When sample_frac is given, the full file is NEVER loaded into memory
    first -- see _stratified_sample_streaming's docstring."""
    if sample_frac is None:
        print("Loading train_model.parquet (full)...")
        return pd.read_parquet(TRAIN_PARQUET_PATH)
    print(f"Streaming a --sample-frac {sample_frac} stratified subsample of train_model.parquet...")
    return _stratified_sample_streaming(TRAIN_PARQUET_PATH, sample_frac, LABEL_COL, stream_batch_size)


def _prepare_xy(design_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Split a design matrix into (y, X): drop ID/label/not-yet-encoded
    columns (see _NON_FEATURE_COLS and _DROP_GROUPS above), printing which
    group each dropped column belongs to, then fail loudly on anything
    LEFT OVER that statsmodels would otherwise choke on deep inside its
    own code with a much less specific error -- a genuinely new,
    unaccounted-for non-numeric column, or a NaN that shouldn't exist
    given every claim-type-specific field is supposed to already be
    zero-filled by build_chow_design_matrix.py. The strict fail-loud check
    stays in place for anything NOT in the enumerated groups above,
    specifically so a future new leak is still caught loudly rather than
    silently swallowed by a blanket "drop any non-numeric column" rule."""
    present_non_feature = _NON_FEATURE_COLS & set(design_df.columns)
    print(f"    dropping {len(present_non_feature)} non-feature column(s) from X:")
    for group_label, group_cols in _DROP_GROUPS:
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
        "",
        f"NOTE: {len(_PENDING_ENCODING_COLS)} raw column(s) (secondary "
        "diagnosis/procedure codes, legacy provider IDs, unaddressed "
        "CMS category codes, and 3 fixed-constant fields) were excluded "
        "from this baseline's feature set pending a real Phase 2 encoding "
        "decision -- see _PENDING_* groups in this file's source. The "
        "omnibus test above covers every covariate that WAS encoded, not "
        "literally every column in train_model.parquet.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_PATH.write_text(report + "\n")
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
