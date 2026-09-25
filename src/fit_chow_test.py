"""
Phase 2 -- fit the Chow test.

The two design matrices (restricted/pooled and unrestricted/fully-
interacted) are built by src/build_chow_design_matrix.py (see
data/FEATURE_ENGINEERING.md Sections 3, 5, 6 for the full design/derivation
history). This script fits both and computes

    LR = -2 * (ll_restricted - ll_unrestricted)   ~   chi2(df)
    H0: every genuinely-shared covariate has one coefficient, pooled
        across claim types (the restricted model is adequate)
    H1: at least one genuinely-shared covariate's effect differs by
        claim type somewhere

DEGREES OF FREEDOM. df is the number of independent restrictions, which
is rank(unrestricted) - rank(restricted) -- NOT the raw column-count
difference unless both matrices are full rank. CORRECTED 2026-09-24:
check_rank.py found both matrices rank-deficient (restricted 144 cols /
rank 119; unrestricted 310 / 275), so every earlier run's reported df
(162/166, from column counts) was wrong -- the true value on those
matrices was 275 - 119 = 156. build_chow_design_matrix.py now removes
every dependency explain_dependencies.py identified, and this script
ENFORCES the fix: it computes each matrix's numerical rank (chunked, see
_matrix_rank_chunked) and refuses to fit unless both are full rank, then
requires the name-based df (_compute_df_and_table, Sigma (n_i - 1) over
genuinely-shared covariates) to equal both the column-count difference
and rank(U) - rank(R). Three independent computations of the same number.

MEMORY. The design matrices are ~3 GB each at full size on an 8 GB
machine. This script never holds train, intermediate, and both design
matrices at once (restricted is built, checked, fit, and reduced to
scalars before the unrestricted matrix is built); never pre-casts X to
float64 itself; explicitly del's each matrix AND fit result (which keeps
its own exog reference) plus gc.collect(); supports --sample-frac
(streamed, never loads the full file first -- 2026-09-23 fix) and
--restricted-only / --skip-restricted. The rank check is chunked (TSQR:
QR of row blocks, combined) so it never materializes a full float64
copy or SVD workspace.

FITTING, REWRITTEN 2026-09-24 (after the first smoke tests on the
full-rank matrices). Three problems, each verified on the --sample-frac
0.02 data before changing anything:
  1. statsmodels lbfgs did not fit these matrices: dollar columns up to
     ~$410,000 beside 0/1 dummies overflowed exp(), and it then reported
     converged=True at a log-likelihood WORSE than the intercept-only
     model (-15,969 vs about -8,516 at 0.02). Every earlier standard-
     method result, including the identical restricted/unrestricted
     log-likelihoods and LR = 0, was this failure, not a finding.
  2. Two separately-penalized Firth fits are not Firth's penalized LR
     test: each model's own penalty 0.5*log|X'WX| enters the difference
     (339.1 that way vs. 213.2 for the proper test, same data).
  3. firthmodels needs ~4x the design matrix in workspace (~10 GB at full
     size), which an 8 GB machine cannot hold.
Now: the unrestricted design is reparameterized as U' = [restricted
columns, Z] (_build_reparameterized_design, span equality verified), the
restricted model is U' with Z's coefficients held at 0 -- penalized, for
Firth, by the FULL model's information, which is the Heinze & Schemper
penalized LR test -- and both fits run in src/chunked_logit.py (Newton-
Raphson over row chunks on column-scaled data; validated against
firthmodels and statsmodels' Newton fits to ~1e-9 in log-likelihood).
df gains a fourth cross-check: the number of tested coefficients (|Z|).
MEMORY now: both design matrices are held in their compact int8/float64
DataFrame form for the rank/df checks, and ONE float64 array (U') is
materialized; each stage (--restricted-only / --skip-restricted) rebuilds
it. No fit ever reports an LR from a non-converged fit.

KNOWN, CONFIRMED SEPARATION (2026-09-23): 6 HCPCS codes (94010, 96156,
99401, 99408, 99495, M1069) have an EXACT 0.000 denial rate within
claim_type=carrier across the full population (up to 45,788 claims for
one code). Mechanism (denial_rules.py): 4 of 6 risk factors structurally
excluded for these codes, PRVDR_NUM NaN for them (silently dropped by
rule_provider_outlier's groupby), duplicate_claim independently rare.
CORRECTED 2026-09-24: PRVDR_NUM is NaN for ALL carrier claims (100%, per
the presence audit), not just for these codes -- see the open
provider_outlier question in reports/SESSION_LOG.md.

TWO FIXES FOR SEPARATION, MEANT TO BE COMPARED (2026-09-23):
  --method firth: Firth's penalized logistic regression (see
      _fit_logit_firth, including its asymptotic caveat).
  --exclude-separating-codes: remove the 6 codes' dummies. CORRECTED
      2026-09-24: now applied to the shared INTERMEDIATE, before either
      matrix is built (_exclude_separating_from_intermediate), instead of
      post hoc per matrix. Same end result for the codes themselves (they
      merge into the reference category in both matrices), but it has to
      happen upstream now: build_chow_design_matrix.py's within-claim-
      type dummy-trap fix drops a reference term per claim-type block
      when a group's dummies sum to the claim_type dummy, and excluding
      codes AFTER that choice could remove a second term from the same
      block, leaving the restricted model's pooled reference column
      outside the unrestricted model's span -- a silent nesting
      violation. Excluding first lets the trap check see the final
      groups.
  Output filenames are suffixed by method/exclusion so runs don't
  overwrite each other.

Run with (from the repo root, inside the venv):
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

from chunked_logit import ChunkedLogit
from build_chow_design_matrix import (
    _GENUINELY_SHARED_PREFIXES,
    PROCESSED_DIR,
    add_claim_type_interactions,
    build_chow_design_matrix,
    build_restricted_design_matrix,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
TRAIN_PARQUET_PATH = PROCESSED_DIR / "train_model.parquet"

LABEL_COL = "is_denied"

_ID_LABEL_COLS = {"BENE_ID", "CLM_ID", "_row_id", LABEL_COL}

_DATE_COLS_NOT_YET_FEATURIZED = {"CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT"}

# --- Columns not yet given ANY Phase 2 encoding decision -----------------
# Confirmed present in real train_model.parquet by running this script
# (2026-09-23). Excluded from the Phase 2 BASELINE/Chow-test feature set
# HERE only, not from train_model.parquet (XGBoost shouldn't inherit a
# baseline-specific exclusion).

# A. Already documented elsewhere as fixed/near-constant.
_PENDING_CONSTANT = {
    "CARR_CLM_PMT_DNL_CD", "CLM_DISP_CD", "CLM_MDCR_NON_PMT_RSN_CD",
}

# B. Raw dates with no Phase 2 numeric transform yet.
_PENDING_DATES = {
    "LINE_1ST_EXPNS_DT", "LINE_LAST_EXPNS_DT", "REV_CNTR_DT",
    *(f"PRCDR_DT{i}" for i in range(1, 25)),
}

# C. Legacy/secondary provider-identifier strings.
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

# E. Other CMS categorical/indicator code fields.
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

# F. Sequence/line-position numbers.
_PENDING_SEQUENCE_NUMS = {"LINE_NUM", "CLM_LINE_NUM"}

# G. Numeric-sounding but object dtype.
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
_SEPARATING_HCPCS_CODES = {"94010", "96156", "99401", "99408", "99495", "M1069"}


def _separating_hcpcs_columns(colnames: set[str]) -> set[str]:
    """Every column derived from a separating HCPCS code present in
    `colnames` (pooled 'hcpcs_<code>' and/or 'hcpcs_<code>__x__<ct>').
    After the 2026-09-24 change fit_chow_test.py removes these upstream
    (_exclude_separating_from_intermediate), so in this script's own flow
    _prepare_xy's call to this is a no-op double-check; check_rank.py and
    explain_dependencies.py still use it post hoc, which is fine for their
    per-matrix rank diagnostics."""
    return {
        c for c in colnames
        if any(c == f"hcpcs_{code}" or c.startswith(f"hcpcs_{code}__x__") for code in _SEPARATING_HCPCS_CODES)
    }


def _exclude_separating_from_intermediate(intermediate: pd.DataFrame) -> pd.DataFrame:
    """Drop the separating codes' dummies from the SHARED intermediate,
    before either matrix is built -- see module docstring for why this has
    to happen upstream now (nesting with the within-claim-type dummy-trap
    fix)."""
    cols = sorted(_separating_hcpcs_columns(set(intermediate.columns)))
    print(f"  --exclude-separating-codes: dropping {len(cols)} dummy column(s) from the shared intermediate: {cols}")
    return intermediate.drop(columns=cols)


def _matrix_rank_chunked(X: pd.DataFrame, chunk_rows: int = 100_000) -> int:
    """Numerical rank of X without materializing a full float64 copy or an
    SVD workspace of X's size: tall-skinny QR (TSQR) -- R factors of row
    blocks, combined by QR of their stack -- has the same singular values
    as X. Tolerance matches numpy.linalg.matrix_rank's default
    (S.max() * max(M, N) * eps), so results agree with check_rank.py."""
    n_rows, n_cols = X.shape
    return _rank_from_chunks(
        (X.iloc[s : s + chunk_rows].to_numpy(dtype=np.float64) for s in range(0, n_rows, chunk_rows)),
        n_rows, n_cols,
    )


def _rank_from_chunks(chunks, n_rows: int, n_cols: int) -> int:
    """TSQR rank over an iterable of float64 row blocks (see
    _matrix_rank_chunked). Split out 2026-09-24 so the reparameterized
    design's span check can stream blocks assembled on the fly."""
    R = None
    for block in chunks:
        stacked = block if R is None else np.vstack([R, block])
        R = np.linalg.qr(stacked, mode="r")
        del block, stacked
    s = np.linalg.svd(R, compute_uv=False)
    tol = s.max() * max(n_rows, n_cols) * np.finfo(np.float64).eps
    return int((s > tol).sum())


def _require_full_rank(X: pd.DataFrame, label: str) -> int:
    rank = _matrix_rank_chunked(X)
    print(f"    rank check ({label}): rank {rank} of {X.shape[1]} columns")
    if rank != X.shape[1]:
        raise ValueError(
            f"{label} design matrix is rank-deficient ({rank} of {X.shape[1]}) -- "
            "a coefficient is not identifiable and the LR df would be wrong. "
            "Run `python src/explain_dependencies.py` to name the dependency, "
            "then fix it in build_chow_design_matrix.py. Not fitting."
        )
    return rank


class _SimpleFitResult:
    """Minimal stand-in for a statsmodels fit result (.llf, .nobs,
    .mle_retvals), so a Firth fit and a statsmodels fit are interchangeable."""

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
        "of train_model.parquet, without ever loading the full file. Must "
        "match between a --restricted-only run and the --skip-restricted "
        "run that follows it.",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=50_000,
        help="Rows per pyarrow batch when streaming a --sample-frac subsample.",
    )
    parser.add_argument(
        "--restricted-only",
        action="store_true",
        help="Fit only the restricted (pooled) model, write its summary, and exit.",
    )
    parser.add_argument(
        "--skip-restricted",
        action="store_true",
        help="Load a previously-saved restricted fit summary instead of "
        "refitting it. Use after --restricted-only with the SAME "
        "--method/--exclude-separating-codes/--sample-frac.",
    )
    parser.add_argument(
        "--method",
        choices=["standard", "firth"],
        default="standard",
        help="'standard': ordinary MLE. 'firth': Firth's penalized MLE, "
        "tested with the penalized likelihood-ratio test (restricted fit "
        "penalized by the full model's information). Both fitted by "
        "chunked Newton-Raphson (src/chunked_logit.py) since 2026-09-24.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Newton-Raphson iteration cap for each fit (chunked_logit.py).",
    )
    parser.add_argument(
        "--exclude-separating-codes",
        action="store_true",
        help="Remove the 6 confirmed structurally-separating HCPCS codes' "
        "dummies from the shared intermediate before either matrix is built.",
    )
    return parser.parse_args()


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = args.method + ("_excl" if args.exclude_separating_codes else "")
    return (
        REPORTS_DIR / f"chow_test_results__{suffix}.txt",
        REPORTS_DIR / f"chow_restricted_fit_summary__{suffix}.json",
    )


def _stratified_sample_streaming(
    parquet_path: Path, frac: float, label_col: str, batch_size: int, seed: int = 42,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Pick a stratified sample of row positions from a label-only read,
    then stream the file in batches, keeping only sampled rows. Never
    materializes the full file. Deterministic given file, frac, seed."""
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
            f"file's own metadata reports {total_rows:,} rows."
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
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        batch_mask = keep_mask[row_offset : row_offset + n]
        row_offset += n
        if batch_mask.any():
            parts.append(batch.to_pandas().loc[batch_mask].copy())
        del batch

    if row_offset != total_rows:
        raise ValueError(
            f"Streamed {row_offset:,} rows via iter_batches but the file's "
            f"metadata reports {total_rows:,}."
        )

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"    kept {len(result):,} of {total_rows:,} rows (shape: {result.shape})")
    return result


def _columns_to_load(parquet_path: Path) -> list[str]:
    """Every column of train_model.parquet EXCEPT the ones _prepare_xy drops
    anyway (_NON_FEATURE_COLS), keeping the label. Added 2026-09-24: the
    first full-size run was OOM-killed (peak RSS 5.5 GB in a ~7 GB box)
    while materializing the reparameterized design, because ~130 raw
    object-dtype columns (secondary dx/procedure codes, dates, legacy IDs)
    were loaded, carried through build_chow_design_matrix, and only then
    dropped -- and freed string memory isn't reliably returned to the OS.
    Not loading them is equivalent (verified at --sample-frac 0.2: both
    design matrices identical, column for column and value for value);
    the only visible difference is that rank_fix_report no longer lists
    never-loaded columns."""
    names = pq.ParquetFile(parquet_path).schema_arrow.names
    return [c for c in names if c not in _NON_FEATURE_COLS or c == LABEL_COL]


def _load_train(sample_frac: float | None, stream_batch_size: int) -> pd.DataFrame:
    columns = _columns_to_load(TRAIN_PARQUET_PATH)
    if sample_frac is None:
        print(f"Loading train_model.parquet (full; {len(columns)} columns needed for the design matrices)...")
        return pd.read_parquet(TRAIN_PARQUET_PATH, columns=columns)
    print(f"Streaming a --sample-frac {sample_frac} stratified subsample of train_model.parquet...")
    return _stratified_sample_streaming(TRAIN_PARQUET_PATH, sample_frac, LABEL_COL, stream_batch_size, columns=columns)


def _prepare_xy(design_df: pd.DataFrame, exclude_separating: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Split a design matrix into (y, X): drop ID/label/not-yet-encoded
    columns, printing which group each dropped column belongs to, then
    fail loudly on anything left over (non-numeric, or unexpected NaN)."""
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
            "accounted for by any of the _PENDING_* groups above: "
            f"{non_numeric}"
        )

    na_counts = X.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        raise ValueError(
            "Unexpected NaN(s) remain in the design matrix. Offending "
            f"column(s):\n{na_counts}"
        )

    return y, X


def _fit_logit_standard(y: pd.Series, X: pd.DataFrame, label: str):
    """SUPERSEDED 2026-09-24 -- no longer called; kept for the record. lbfgs
    on these unscaled dollar columns overflowed and reported converged=True
    at a log-likelihood worse than the intercept-only model (module
    docstring). main() now fits with chunked_logit.ChunkedLogit.

    Ordinary MLE via statsmodels, method='lbfgs'. X passed as a
    DataFrame so statsmodels does the float64 upcast exactly once."""
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
    """SUPERSEDED 2026-09-24 -- no longer called; kept for the record. Two
    separately-penalized fits are not Firth's penalized LR test, and
    firthmodels' workspace cannot hold the full-size unrestricted matrix in
    8 GB (module docstring). main() now fits with chunked_logit.ChunkedLogit.

    Firth's penalized (bias-reduced) logistic regression via firthmodels
    (successor to the archived firthlogist; written against firthmodels
    0.8.2's verified API). fit_intercept=False -- the claim_type dummies
    are the intercepts (cell-means coding). .fit() computes only cheap
    Wald statistics; the expensive per-coefficient .lrt() is never called.
    max_iter=500 (package default 25; was 100 until 2026-09-24).

    WHY 500 (2026-09-24): the first 0.02 smoke test's unrestricted Firth fit
    stopped at the 100-iteration cap with converged=False. Tracing the
    solver showed it was not diverging, just slow: prvdr_num_freq__x__*
    are tiny-scale columns (a share of claims, ~1e-3) whose coefficients
    have to reach ~400, and firthmodels clips every Newton step to
    max_step=5 per coefficient, so it takes ~85 clipped steps to get there.
    With max_iter=400 the same fit converged at iteration 105 (max |score|
    8e-5), to the SAME penalized log-likelihood (-4847.0450) the capped
    run had already reached. 500 leaves headroom; main() now refuses to
    compute an LR from any fit that reports converged=False.

    NOTE (2026-09-24): the first Firth attempt failed with "Weighted design
    matrix is rank deficient". That was the RAW collinearity now fixed in
    build_chow_design_matrix.py (and enforced by _require_full_rank), not
    separation -- the deficiency count was identical with and without the
    separating codes.

    CAVEAT: the chi-square justification for a PENALIZED LR test is
    well-established away from separation, but under TRUE separation it
    is empirically validated (Heinze & Schemper 2002), not a proven exact
    asymptotic result -- LR tests near a parameter-space boundary can
    follow a mixture of chi-squares (Self & Liang 1987). Compare against
    the --exclude-separating-codes run, which needs no such caveat.
    """
    from firthmodels import FirthLogisticRegression

    print(f"\n  fitting {label} (Firth's penalized MLE via firthmodels): {X.shape[0]:,} rows x {X.shape[1]} columns")
    print("    this can be slower than the standard lbfgs fit -- watch for a long run.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FirthLogisticRegression(fit_intercept=False, max_iter=500)
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
    # SUPERSEDED 2026-09-24 with the two fitters above; kept for the record.
    result = _fit_logit_firth(y, X, label) if method == "firth" else _fit_logit_standard(y, X, label)
    # 2026-09-24: an LR statistic from a fit that stopped short of its
    # optimum is not a likelihood-ratio test. Refuse rather than report one
    # (the first Firth smoke test printed converged=False and still went
    # on to compute LR/p).
    if result.mle_retvals.get("converged") is not True:
        raise RuntimeError(
            f"{label} fit did not converge (converged="
            f"{result.mle_retvals.get('converged')!r}) -- not computing an LR "
            "statistic from it. Investigate before rerunning."
        )
    return result


def _compute_df_and_table(
    unrestricted_colnames: set[str], restricted_colnames: set[str]
) -> tuple[int, list[tuple[str, int]]]:
    """df = Sigma_i (n_i - 1) from the two matrices' column names: a
    genuinely-shared covariate is a base name appearing as 1+
    '<base>__x__<claim_type>' columns ONLY in the unrestricted set and as
    a single plain '<base>' column ONLY in the restricted set. Raises on
    anything that doesn't fit that pattern. Valid only when both matrices
    are full rank -- main() enforces that before calling this."""
    only_unrestricted = unrestricted_colnames - restricted_colnames
    only_restricted = restricted_colnames - unrestricted_colnames

    n_i_by_base: dict[str, int] = defaultdict(int)
    for col in only_unrestricted:
        if "__x__" not in col:
            raise ValueError(
                f"{col!r} exists only in the unrestricted matrix but isn't "
                "an interaction term -- the two matrices have diverged."
            )
        base = col.rsplit("__x__", 1)[0]
        n_i_by_base[base] += 1

    unaccounted_restricted = only_restricted - set(n_i_by_base.keys())
    # n_i = 0: a dummy category whose ONLY unrestricted term was removed as
    # a within-claim-type dummy-trap reference (build_chow_design_matrix.py
    # fix C prefers to avoid this but can't always). Its pooled column is
    # still in the unrestricted span (= claim_type dummy minus the block's
    # other terms), so nesting holds, and it contributes n_i - 1 = -1 --
    # exactly consistent with the column counts. Anything else unmatched is
    # still an error.
    zero_term_dummies = {c for c in unaccounted_restricted if c.startswith(_GENUINELY_SHARED_PREFIXES)}
    unaccounted_restricted -= zero_term_dummies
    if unaccounted_restricted:
        raise ValueError(
            "Column(s) present only in the restricted matrix with no "
            f"matching interaction terms in the unrestricted matrix: "
            f"{unaccounted_restricted}"
        )
    for c in zero_term_dummies:
        n_i_by_base[c] = 0
    missing_pooled = {b for b, k in n_i_by_base.items() if k > 0} - only_restricted
    if missing_pooled:
        raise ValueError(
            "Covariate(s) interacted in the unrestricted matrix with no "
            f"pooled column in the restricted matrix: {missing_pooled}"
        )

    df_table = sorted(n_i_by_base.items(), key=lambda kv: (-kv[1], kv[0]))
    df_total = sum(n_i - 1 for _, n_i in df_table)
    return df_total, df_table


TEST_FORM = "joint-penalized-LR-on-reparameterized-design (2026-09-24)"


def _save_restricted_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n  restricted fit summary written to {path}")


def _load_restricted_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"--skip-restricted was passed but {path} doesn't exist -- run "
            "with --restricted-only (with the SAME --method/"
            "--exclude-separating-codes/--sample-frac) first."
        )
    summary = json.loads(path.read_text())
    if "rank" not in summary:
        raise ValueError(
            f"{path} predates the 2026-09-24 full-rank enforcement (no 'rank' "
            "field) -- it was fit on a rank-deficient matrix. Rerun "
            "--restricted-only."
        )
    if summary.get("test_form") != TEST_FORM:
        raise ValueError(
            f"{path} was produced by the superseded separate-fits form of the "
            "test (lbfgs / separately-penalized Firth -- see module docstring, "
            "2026-09-24). Rerun --restricted-only."
        )
    return summary


def _build_reparameterized_design(
    X_R: pd.DataFrame, X_U: pd.DataFrame, chunk_rows: int = 100_000
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Put the Chow test in "full model, test some coefficients = 0" form.

    U' = [every restricted column, Z], where Z = the unrestricted-only
    interaction terms MINUS one term per covariate that has a pooled column
    (for a covariate x: {x_pooled, x*ct2, x*ct3} spans the same space as
    {x*ct1, x*ct2, x*ct3}, and "coefficients on x*ct2, x*ct3 = 0" is exactly
    "one pooled slope"). The restricted model is then U' with Z's
    coefficients held at 0 -- the form Firth's penalized LR test needs (the
    restricted fit penalized by the FULL model's information), and, for
    ordinary MLE, the same log-likelihood as fitting X_R alone.

    Verified, not assumed: U' is full rank, has exactly as many columns as
    X_U, and every X_U term left out of Z lies in span(U') (rank of
    [U', left-out terms] == rank(U')). Together with X_U's own full rank
    that proves span(U') == span(X_U), so the full U' fit IS the
    unrestricted model. Returns (A, colnames, z_mask, left_out) with A a
    C-contiguous float64 array (the one full float64 materialization).
    """
    r_cols = list(X_R.columns)
    r_set = set(r_cols)
    by_base: dict[str, list[str]] = defaultdict(list)
    for c in X_U.columns:
        if c not in r_set:
            by_base[c.rsplit("__x__", 1)[0]].append(c)
    z_cols: list[str] = []
    left_out: list[str] = []
    for base, cols in by_base.items():
        cols = sorted(cols)
        if base in r_set:
            left_out.append(cols[0])
            z_cols += cols[1:]
        else:
            z_cols += cols
    # (A pooled column whose only unrestricted term was removed as a
    # within-claim-type dummy-trap reference -- the n_i = 0 case -- gets no Z
    # term; the span check below covers it like everything else.)

    n = len(X_R)
    k = len(r_cols) + len(z_cols)
    if k != X_U.shape[1]:
        raise ValueError(
            f"Reparameterized design has {k} columns but the unrestricted matrix "
            f"has {X_U.shape[1]} -- the correspondence between the two matrices is off."
        )
    A = np.empty((n, k), dtype=np.float64)
    for j, c in enumerate(r_cols):
        A[:, j] = X_R[c].to_numpy(dtype=np.float64)
    for j, c in enumerate(z_cols, start=len(r_cols)):
        A[:, j] = X_U[c].to_numpy(dtype=np.float64)

    rank_A = _rank_from_chunks((A[s : s + chunk_rows] for s in range(0, n, chunk_rows)), n, k)
    if rank_A != k:
        raise ValueError(f"Reparameterized design is rank-deficient ({rank_A} of {k}). Not fitting.")
    if left_out:
        L = X_U[left_out]
        rank_aug = _rank_from_chunks(
            (np.hstack([A[s : s + chunk_rows], L.iloc[s : s + chunk_rows].to_numpy(dtype=np.float64)])
             for s in range(0, n, chunk_rows)),
            n, k + len(left_out),
        )
        if rank_aug != k:
            raise ValueError(
                f"{rank_aug - k} unrestricted term(s) are NOT in the span of the "
                "reparameterized design -- it would not be the unrestricted model. Not fitting."
            )
    print(
        f"    reparameterized design: {len(r_cols)} restricted + {len(z_cols)} tested (Z) = {k} columns, "
        f"full rank; {len(left_out)} left-out unrestricted term(s) verified to lie in its span"
    )
    z_mask = np.r_[np.zeros(len(r_cols), dtype=bool), np.ones(len(z_cols), dtype=bool)]
    return A, r_cols + z_cols, z_mask, left_out


def _build_intermediate(args: argparse.Namespace) -> pd.DataFrame:
    train = _load_train(args.sample_frac, args.stream_batch_size)
    intermediate = build_chow_design_matrix(train)
    del train
    gc.collect()
    report = intermediate.attrs.get("rank_fix_report", {})
    print(
        f"  rank fixes: {len(report.get('identity_dropped', []))} identity-redundant, "
        f"{len(report.get('claim_type_determined_dropped', []))} claim-type-determined column(s) removed"
    )
    if report.get("zero_with_gaps_warning"):
        print(f"  [warning] partly-NaN all-zero column(s) dropped: {report['zero_with_gaps_warning']}")
    if args.exclude_separating_codes:
        intermediate = _exclude_separating_from_intermediate(intermediate)
    return intermediate


def main() -> None:
    args = parse_args()
    if args.restricted_only and args.skip_restricted:
        raise SystemExit("--restricted-only and --skip-restricted are mutually exclusive.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path, restricted_summary_path = _output_paths(args)
    penalty_weight = 0.5 if args.method == "firth" else 0.0

    cached = None
    if args.skip_restricted:
        print(f"Loading cached restricted fit summary from {restricted_summary_path}...")
        cached = _load_restricted_summary(restricted_summary_path)
        for field, current in (("sample_frac", args.sample_frac), ("method", args.method), ("exclude_separating_codes", args.exclude_separating_codes)):
            if cached.get(field) != current:
                raise ValueError(
                    f"Cached restricted fit was built with {field}={cached.get(field)!r}, "
                    f"this run passed {current!r} -- not comparable. Rerun --restricted-only."
                )

    intermediate = _build_intermediate(args)

    # Both matrices are still built exactly as before (same builders, same
    # checks), kept in their compact int8/float64 DataFrame form; only the
    # reparameterized design is materialized as float64, once.
    print("\nBuilding restricted (pooled) design matrix...")
    restricted_df = build_restricted_design_matrix(intermediate)
    y, X_R = _prepare_xy(restricted_df, exclude_separating=args.exclude_separating_codes)
    del restricted_df
    gc.collect()
    restricted_rank = _require_full_rank(X_R, "restricted")

    print("\nBuilding unrestricted (fully-interacted) design matrix...")
    unrestricted_df, interaction_cols, skipped_terms = add_claim_type_interactions(intermediate)
    del intermediate
    gc.collect()
    y_u, X_U = _prepare_xy(unrestricted_df, exclude_separating=args.exclude_separating_codes)
    del unrestricted_df
    gc.collect()
    if not y_u.equals(y):
        raise ValueError("Restricted and unrestricted matrices do not cover identical rows.")
    del y_u
    unrestricted_rank = _require_full_rank(X_U, "unrestricted")

    restricted_colnames = set(X_R.columns)
    unrestricted_colnames = set(X_U.columns)
    df_chow, df_table = _compute_df_and_table(unrestricted_colnames, restricted_colnames)
    shape_diff = X_U.shape[1] - X_R.shape[1]
    rank_diff = unrestricted_rank - restricted_rank

    print("\nBuilding the reparameterized full design (see _build_reparameterized_design)...")
    A, colnames, z_mask, _ = _build_reparameterized_design(X_R, X_U)
    restricted_ncols, unrestricted_ncols = X_R.shape[1], X_U.shape[1]
    del X_R, X_U
    gc.collect()
    if not (df_chow == shape_diff == rank_diff == int(z_mask.sum())):
        raise ValueError(
            f"df disagreement: name-based {df_chow}, column-count {shape_diff}, "
            f"rank-based {rank_diff}, tested-coefficient count {int(z_mask.sum())}. "
            "Do not trust the LR test until resolved."
        )

    y_arr = y.to_numpy(dtype=np.float64)
    n_obs = len(y_arr)
    del y
    model = ChunkedLogit(A, y_arr, penalty_weight=penalty_weight)  # scales A in place
    method_desc = (
        "Firth's penalized MLE (chunked Newton-Raphson, src/chunked_logit.py; "
        "restricted fit penalized by the full model's information)"
        if args.method == "firth"
        else "ordinary MLE (chunked Newton-Raphson, src/chunked_logit.py)"
    )

    if cached is None:
        print(f"\n  fitting RESTRICTED model ({method_desc}): Z's {int(z_mask.sum())} coefficients held at 0")
        res_r = model.fit(fixed_zero=z_mask, max_iter=args.max_iter)
        print(
            f"    converged: {res_r.converged} after {res_r.n_iter} iteration(s); "
            f"log-likelihood {res_r.llf:.4f} (unpenalized {res_r.llf_unpenalized:.4f}); "
            f"max|score| {res_r.max_abs_score:.3g}; Newton decrement {res_r.newton_decrement:.3g}"
        )
        summary = {
            "test_form": TEST_FORM,
            "llf": res_r.llf,
            "llf_unpenalized": res_r.llf_unpenalized,
            "converged": res_r.converged,
            "n_iter": res_r.n_iter,
            "max_abs_score": res_r.max_abs_score,
            "newton_decrement": res_r.newton_decrement,
            "nobs": n_obs,
            "ncols": restricted_ncols,
            "rank": restricted_rank,
            "colnames": sorted(restricted_colnames),
            "sample_frac": args.sample_frac,
            "method": args.method,
            "exclude_separating_codes": args.exclude_separating_codes,
        }
        _save_restricted_summary(restricted_summary_path, summary)
        if not res_r.converged:
            raise RuntimeError(
                "Restricted fit did not converge -- no LR statistic will be computed from it. "
                "See the iteration trace above (separation shows up as coefficients running "
                "off while the log-likelihood stalls)."
            )
        if args.restricted_only:
            print("\n--restricted-only: stopping here. Rerun with --skip-restricted to finish the test.")
            return
    else:
        summary = cached
        if summary["nobs"] != n_obs or set(summary["colnames"]) != restricted_colnames:
            raise ValueError("Cached restricted fit does not match this run's rows/columns -- rerun --restricted-only.")
        if not summary["converged"]:
            raise RuntimeError("Cached restricted fit did not converge -- nothing to test against.")

    print(f"\n  fitting UNRESTRICTED model ({method_desc}): all {len(colnames)} coefficients free")
    res_u = model.fit(max_iter=args.max_iter)
    print(
        f"    converged: {res_u.converged} after {res_u.n_iter} iteration(s); "
        f"log-likelihood {res_u.llf:.4f} (unpenalized {res_u.llf_unpenalized:.4f}); "
        f"max|score| {res_u.max_abs_score:.3g}; Newton decrement {res_u.newton_decrement:.3g}"
    )
    if not res_u.converged:
        raise RuntimeError("Unrestricted fit did not converge -- not computing an LR statistic.")
    del model, A
    gc.collect()

    ll_restricted = summary["llf"]
    ll_unrestricted = res_u.llf
    LR = -2 * (ll_restricted - ll_unrestricted)
    if LR < -1e-6:
        raise RuntimeError(
            f"LR = {LR:.6f} < 0: the restricted fit beat the unrestricted one, which is impossible "
            "for nested models at their optima -- one fit is not at its optimum."
        )
    LR = max(LR, 0.0)
    p_value = stats.chi2.sf(LR, df_chow)

    n_i_counts = Counter(n_i for _, n_i in df_table)
    breakdown = ", ".join(
        f"{n_covariates} covariate(s) with n_i={n_i} (contributing {n_i - 1} df each)"
        for n_i, n_covariates in sorted(n_i_counts.items())
    )
    lines = [
        "Chow test -- claim-type homogeneity of shared covariates",
        "=" * 60,
        f"Method:                              {method_desc}",
        f"Test form:                           {TEST_FORM}",
        f"Separating-code exclusion:           {'ON -- dropped ' + str(sorted(_SEPARATING_HCPCS_CODES)) if args.exclude_separating_codes else 'OFF'}",
        f"--sample-frac:                       {args.sample_frac if args.sample_frac is not None else 'full train set'}",
        f"n obs:                               {n_obs:,}",
        f"restricted (pooled) columns / rank:  {restricted_ncols} / {restricted_rank}",
        f"unrestricted columns / rank:         {unrestricted_ncols} / {unrestricted_rank}",
        f"degrees of freedom:                  {df_chow}  (name-based = column-count = rank-based = tested coefficients)  [{breakdown}]",
        "",
        f"log-likelihood, restricted:    {ll_restricted:.4f}   (converged, {summary['n_iter']} iterations, max|score| {summary['max_abs_score']:.3g})",
        f"log-likelihood, unrestricted:  {ll_unrestricted:.4f}   (converged, {res_u.n_iter} iterations, max|score| {res_u.max_abs_score:.3g})",
        f"unpenalized log-likelihoods:   {summary['llf_unpenalized']:.4f} / {res_u.llf_unpenalized:.4f}",
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
        f"Unrestricted interaction terms skipped (constant within claim type, "
        f"within-DME duplicate, or within-claim-type dummy-trap reference -- "
        f"none carry information beyond columns kept): {len(skipped_terms)}",
        "",
        f"NOTE: {len(_PENDING_ENCODING_COLS)} raw column(s) were excluded "
        "from this baseline's feature set pending a Phase 2 encoding "
        "decision -- see _PENDING_* groups in this file's source.",
    ]
    if args.method == "firth":
        lines += [
            "",
            "CAVEAT (Firth method): the chi-square approximation used for the",
            "p-value above is well-established away from separation, but under",
            "TRUE separation it is empirically validated (Heinze & Schemper",
            "2002), not a proven exact asymptotic result -- LR tests near a",
            "parameter-space boundary can follow a mixture of chi-squares (Self",
            "& Liang 1987). Compare against the --exclude-separating-codes run.",
        ]
    report = "\n".join(lines)
    print("\n" + report)
    results_path.write_text(report + "\n")
    print(f"\nWritten to {results_path}")


if __name__ == "__main__":
    main()
