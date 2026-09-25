"""
Phase 2 -- build the design matrices for the Chow test (and, eventually, the
Phase 2 baseline logistic regression). Operates on a COPY of
train_model.parquet only -- never mutates the shared file XGBoost also
reads from. See data/FEATURE_ENGINEERING.md Sections 5-6 for the full
reasoning behind each encoding choice below.

Run after build_features.py.

Three stages:
  1. build_chow_design_matrix() -- cardinality encoding for the shared/
     2-of-3 covariates that needed it (HCPCS_CD, PRNCPAL_DGNS_CD, PRVDR_NUM,
     provider_state), PLUS (2026-09-24) removal of every column the rank-
     deficiency investigation proved carries no information beyond claim
     type or another column -- see RANK-DEFICIENCY FIXES below.
  2. add_claim_type_interactions() -- builds the UNRESTRICTED (fully-
     interacted) model's design matrix. Two passes:
       (a) numeric claim-type-exclusive/2-of-3 covariates: zero-fill-on-a-
           copy, build value x claim_type_dummy terms only for the claim
           type(s) each covariate is actually populated in
           (FEATURE_ENGINEERING.md Section 3's degrees-of-freedom
           correction) -- and (2026-09-24) SKIPPING any claim type where
           the covariate is constant, since value*dummy is then just
           c*dummy (see RANK-DEFICIENCY FIXES).
       (b) genuinely-3-of-3 categorical dummy groups (provider_state,
           HCPCS_CD, PRNCPAL_DGNS_CD): interact against ALL three
           claim_type dummies, but SKIP any resulting interaction term
           that is constant within that claim type (originally only the
           all-zero case, 2026-09-22; generalized to all-one too,
           2026-09-24), and drop one reference term per claim-type block
           when a group's dummies sum to exactly that claim_type dummy
           (the within-claim-type dummy trap, 2026-09-24).
     Run via build_full_design_matrix().
  3. build_restricted_design_matrix() -- builds the RESTRICTED (pooled)
     model's design matrix: every genuinely shared covariate (2-of-3 or
     3-of-3) as ONE plain column instead of separate per-claim-type
     interaction terms -- this is the actual restriction the Chow test's
     likelihood-ratio comparison is testing. Claim-type-EXCLUSIVE fields
     (n_i=1) get identical treatment in both models.

Both design matrices are built from the SAME intermediate
(build_chow_design_matrix() output), so they cover identical rows and an
identical universe of covariates -- required for the likelihood-ratio
test's nesting assumption to actually hold (pre-flight checklist item 1).

RANK-DEFICIENCY FIXES (2026-09-24). check_rank.py found both matrices
rank-deficient (restricted 144 cols / rank 119; unrestricted 310 / 275),
sample-size-invariant and unrelated to the HCPCS separation; firthmodels'
Newton-Raphson refused to fit because of it, and every earlier LBFGS fit
had silently carried it (its HessianInversionWarning). explain_dependencies.py
then wrote out every dependency as an exact equation -- all 25 restricted
and all 35 unrestricted accounted for. Fixes, in order of where they run:

  A. CLAIM-TYPE-DETERMINED COLUMNS (build_chow_design_matrix, both
     matrices). A column whose value is the SAME on every row of each
     claim type is an exact linear combination of the three claim_type
     dummies and carries no other information. "Present" means NON-NULL:
     a 0 is a present value like any other ("0 is a value, NaN is the
     absence of one" -- the project's standing rule), so a field that is
     non-null and always exactly 0 within a claim type counts as constant
     there, while a claim type where the field is 100% NaN is simply
     absent and is not judged at all. Dropped entirely when constant in
     EVERY claim type it's present in. This single rule removes:
       - the 5 NPI presence flags -- confirmed pure claim-type indicators
         (has_performing = carrier; has_referring = carrier + dme;
         has_attending = has_operating = has_rendering = outpatient),
         upgrading the 2026-09-22 "4 of 5 are claim-type-exclusive"
         finding;
       - the always-$0 fields (MSP 1st/2nd paid, blood deductible, MCO-paid
         switch, beneficiary payment amounts, DME screening savings,
         reduced-payment-physician-assistant);
       - the fixed-nonzero ones (CLAIM_QUERY_CODE = 3 and
         NCH_PROFNL_CMPNT_CHRG_AMT = $4 and REV_CNTR_UNIT_CNT = 1 on every
         outpatient claim; LINE_BENE_PRMRY_PYR_PD_AMT = $0 on every
         carrier claim and $1 on every DME claim).
     Partial presence within a claim type (some rows NaN, some not) is
     NOT treated as constant when the present values are nonzero -- the
     zero-filled term is then c * (presence indicator), which varies. The
     one ambiguous case -- partial presence where every present value is
     0 -- becomes an all-zero column after the existing zero-fill, so it
     is dropped for rank's sake but printed as a warning: the zero-fill
     itself is conflating a present 0 with absence there, a pre-existing
     issue outside this fix's scope.
     When a column is constant in SOME but not all of its present claim
     types, it is kept, and only the constant claim type's interaction
     term is skipped in the unrestricted matrix (fix C).

  B. NAMED EXACT IDENTITIES (build_chow_design_matrix, both matrices).
     Verified numerically on the actual data every run (raises if any no
     longer hold), then the redundant side dropped:
       outpatient-exclusive fields --
         CLM_OP_PRVDR_PMT_AMT   == CLM_TOT_CHRG_AMT (Synthea pays the
                                   full charge, no contractual adjustment)
         REV_CNTR_PRVDR_PMT_AMT == CLM_TOT_CHRG_AMT
         REV_CNTR_CASH_DDCTBL_AMT == NCH_BENE_PTB_DDCTBL_AMT
         REV_CNTR_COINSRNC_WGE_ADJSTD_C == REV_CNTR_RDCD_COINSRNC_AMT
         REV_CNTR_PTNT_RSPNSBLTY_PMT == NCH_BENE_PTB_DDCTBL_AMT +
                                        REV_CNTR_RDCD_COINSRNC_AMT
           (a genuine Medicare accounting identity: patient
            responsibility = deductible + coinsurance)
       within DME only (these differ in carrier, so only the DME
       interaction term is redundant) --
         LINE_PRMRY_ALOWD_CHRG_AMT == LINE_ALOWD_CHRG_AMT
         CARR_CLM_PRMRY_PYR_PD_AMT == NCH_CARR_CLM_ALOWD_AMT
       Kept representatives: CLM_TOT_CHRG_AMT, NCH_BENE_PTB_DDCTBL_AMT,
       REV_CNTR_RDCD_COINSRNC_AMT, LINE_ALOWD_CHRG_AMT,
       NCH_CARR_CLM_ALOWD_AMT.

  C. PER-CLAIM-TYPE CONSTANCY + WITHIN-CLAIM-TYPE DUMMY TRAP
     (add_claim_type_interactions, unrestricted only). Pass 1 skips
     value x claim_type terms where the value is constant in that claim
     type; Pass 2's guard now skips terms constant within the claim type
     (all-0 OR all-1, not just all-0). And: drop_first=True picks the
     reference category once, globally -- if that category never occurs
     in carrier (or DME), the remaining dummies x carrier sum to exactly
     claim_type_carrier, a dummy trap inside the claim-type block that
     drop_first cannot see. Confirmed for HCPCS in both carrier and DME.
     Fixed by dropping one term in any such block as the within-claim-
     type reference -- preferring a category that also has a term in
     another claim type (so it keeps n_i >= 1), most frequent first;
     applied uniformly to state_/hcpcs_/dgns_. Nesting is preserved:
     the restricted model's pooled column for the dropped category is
     still reproducible as claim_type_dummy minus the block's other terms.

  D. carr_num_freq REMOVED. Each state maps to one Medicare carrier
     number, so a frequency encoding of CARR_NUM is a function of state --
     exactly in the span of the state dummies within each claim type
     (the solved coefficients were each state's own share: CA 0.099, FL
     0.094, NY 0.061...). It was redundant with provider_state and only
     escaped the restricted matrix's rank check because the outpatient
     zero-fill broke the identity there.

Nesting still holds throughout because every removal either happens in
the shared intermediate (both matrices) or removes a column that is an
exact combination of columns the unrestricted matrix keeps.

MEMORY NOTE (2026-09-22): every function below builds new columns into a
plain dict first and does ONE pd.concat at the end, rather than assigning
columns one at a time via df[new_col] = ... inside a loop. This isn't a
style preference -- pandas stores a DataFrame as contiguous per-dtype
blocks, and repeated one-at-a-time column assignment fragments those
blocks, forcing pandas to periodically re-consolidate them into one
contiguous array. That consolidation needs a temporary array roughly the
size of the whole block being merged -- confirmed to actually happen here:
adding this file's Section-6 interaction terms one column at a time hit a
literal numpy.core._exceptions.ArrayMemoryError trying to allocate an 888
MiB temporary array during exactly this kind of consolidation.

SECOND MEMORY NOTE (2026-09-22): a leading `df = df.copy()` at the top of
a function is ALSO a consolidation trigger -- only copy when a function
genuinely mutates its input in place. add_claim_type_interactions() never
does (no copy); build_restricted_design_matrix() does (keeps its copy).

THIRD MEMORY NOTE (2026-09-22): every genuinely binary column (one-hot
dummies and every interaction term built from them) is int8, not int64 --
an 8x memory reduction, no information loss. NUMERIC claim-type-exclusive/
2-of-3 covariates keep their natural float64.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TOP_N_HCPCS = 20
TOP_N_DGNS = 20

_CLAIM_TYPES = ("carrier", "outpatient", "dme")
_CLAIM_TYPE_COLS = ("claim_type_carrier", "claim_type_outpatient", "claim_type_dme")

# Columns never touched by the interaction-construction logic -- identifiers,
# dates needing their own transformation, the label, and the claim_type
# dummies themselves (the interaction PARTNER, and protected -- never
# eligible for any of the rank-deficiency removals below).
_NEVER_INTERACT = {
    "BENE_ID", "CLM_ID", "_row_id", "is_denied",
    "claim_type_carrier", "claim_type_outpatient", "claim_type_dme",
    "CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT",
}

# The 5 NPI presence flags. CORRECTED 2026-09-24: these were believed to be
# genuinely-shared 3-of-3 covariates; explain_dependencies.py proved each is
# an exact function of claim type (see module docstring, fix A). They are
# now removed by the general claim-type-determined rule in
# build_chow_design_matrix(), not by name -- listed here only so the
# downstream code that still references the group keeps working, and so
# __main__ can confirm they're gone.
_NPI_FLAGS = (
    "has_referring_physician", "has_performing_physician",
    "has_attending_physician", "has_operating_physician",
    "has_rendering_physician",
)

# Prefixes for the one-hot dummies build_chow_design_matrix() produces --
# genuinely-3-of-3 shared covariates, interacted in Pass 2.
_GENUINELY_SHARED_PREFIXES = ("state_", "hcpcs_", "dgns_")

_BINARY_DTYPE = "int8"

# --- Fix B: named exact identities (2026-09-24) ---------------------------
# (redundant column, ((kept column, coefficient), ...)). Verified on the
# actual data every run by _verify_identity() before anything is dropped.
_REDUNDANT_RAW_IDENTITIES: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    ("CLM_OP_PRVDR_PMT_AMT", (("CLM_TOT_CHRG_AMT", 1.0),)),
    ("REV_CNTR_PRVDR_PMT_AMT", (("CLM_TOT_CHRG_AMT", 1.0),)),
    ("REV_CNTR_CASH_DDCTBL_AMT", (("NCH_BENE_PTB_DDCTBL_AMT", 1.0),)),
    ("REV_CNTR_COINSRNC_WGE_ADJSTD_C", (("REV_CNTR_RDCD_COINSRNC_AMT", 1.0),)),
    (
        "REV_CNTR_PTNT_RSPNSBLTY_PMT",
        (("NCH_BENE_PTB_DDCTBL_AMT", 1.0), ("REV_CNTR_RDCD_COINSRNC_AMT", 1.0)),
    ),
)

# (redundant column, claim type, kept column): identical ONLY within that
# claim type, so only the one interaction term is redundant.
_REDUNDANT_WITHIN_TYPE: tuple[tuple[str, str, str], ...] = (
    ("LINE_PRMRY_ALOWD_CHRG_AMT", "dme", "LINE_ALOWD_CHRG_AMT"),
    ("CARR_CLM_PRMRY_PYR_PD_AMT", "dme", "NCH_CARR_CLM_ALOWD_AMT"),
)
_REDUNDANT_INTERACTION_TERMS = {f"{col}__x__{ct}" for col, ct, _ in _REDUNDANT_WITHIN_TYPE}

# Statuses (see _within_type_status) under which value*claim_type_dummy is
# an exact multiple of the dummy (or zero) and so must not enter the matrix.
_DETERMINED = ("determined", "zero_with_gaps")


def top_n_encode(
    series: pd.Series, n: int, prefix: str, categories: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode the top-n most frequent categories. Everything else
    buckets into '__OTHER__'; real NaN buckets into its own '__MISSING__'
    category rather than being silently dropped or folded into '__OTHER__'.

    Reference-cell one-hot (drop_first=True). NOTE (2026-09-24): drop_first
    picks ONE global reference category -- it cannot prevent a dummy trap
    inside a claim-type interaction block when that reference never occurs
    in the claim type. _interact_with_zero_variance_guard() handles that
    case (module docstring, fix C). dtype=int8 -- see THIRD MEMORY NOTE.

    `categories`, added 2026-09-25 for the Phase 2 baseline model: pass the
    list returned from a TRAIN call here to encode val/test against the same
    fixed category set (own-split top-n would silently redefine what
    "__OTHER__" means between splits -- a real code moving in or out of the
    top-n between train and val is a wrong-vocabulary bug, not a shift the
    model should absorb). Returns (dummies, categories_used) always, so the
    caller can pass categories_used straight into the next split's call.
    """
    if categories is None:
        categories = series.value_counts(dropna=True).head(n).index.tolist()
    bucketed = series.where(series.isin(categories), other="__OTHER__")
    bucketed = bucketed.where(series.notna(), other="__MISSING__")
    dummies = pd.get_dummies(bucketed, prefix=prefix, drop_first=True, dtype=_BINARY_DTYPE)
    return dummies, categories


def frequency_encode(series: pd.Series, freq_map: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Map each category to its own frequency (share of non-null rows) --
    for PRVDR_NUM, whose high cardinality makes one-hot infeasible. Real NaN
    is preserved, NOT filled (a 2-of-3 covariate whose absent claim type
    must stay NaN for the interaction logic below).

    `freq_map`, added 2026-09-25: pass the map returned from a TRAIN call to
    score val/test against train's own frequencies, not the val/test split's
    own -- a provider unseen in train maps to NaN (handled downstream the
    same as any other absent-in-this-claim-type value), never a val-specific
    frequency the model was never fit against. Returns (encoded, freq_map)
    always, mirroring top_n_encode."""
    if freq_map is None:
        freq_map = series.value_counts(normalize=True, dropna=True)
    return series.map(freq_map), freq_map


def _claim_type_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {ct: (df[f"claim_type_{ct}"] == 1).to_numpy() for ct in _CLAIM_TYPES}


def _within_type_status(values: pd.Series, ct_mask: np.ndarray) -> str:
    """Classify one column within one claim type. "Present" = non-null; a 0
    is a present value (module docstring, fix A).
      'absent'         -- 100% NaN in this claim type (not judged)
      'determined'     -- non-null on every row, one value (0 included)
      'zero_with_gaps' -- partly NaN, every present value is 0: the zero-
                          filled interaction term becomes all-zero
      'varies'         -- anything else (incl. partly NaN with a nonzero
                          constant, whose term is c * presence indicator)"""
    v = values[ct_mask]
    present = v.notna()
    if not present.any():
        return "absent"
    vals = v[present]
    lo, hi = vals.min(), vals.max()
    if present.all():
        return "determined" if lo == hi else "varies"
    if lo == hi == 0:
        return "zero_with_gaps"
    return "varies"


def _verify_identity(
    df: pd.DataFrame, target: str, rhs: tuple[tuple[str, float], ...], mask: np.ndarray | None = None
) -> None:
    """Raise unless `target == sum(coef * col)` holds exactly (to float
    tolerance) on every row where target is present, AND every rhs column
    has the identical presence pattern -- an identity that only holds on
    the rows both happen to share isn't safe to drop on."""
    sub = df if mask is None else df.loc[mask]
    t = sub[target]
    present = t.notna()
    for col, _ in rhs:
        if not (sub[col].notna() == present).all():
            raise ValueError(
                f"Identity check failed: {target} and {col} have different "
                "NaN patterns -- the 2026-09-24 identity no longer holds on "
                "this data. Do not drop it; re-run explain_dependencies.py."
            )
    lhs = t[present].to_numpy(dtype=np.float64)
    rhs_vals = np.zeros_like(lhs)
    for col, coef in rhs:
        rhs_vals += coef * sub.loc[present, col].to_numpy(dtype=np.float64)
    if not np.allclose(lhs, rhs_vals, rtol=1e-9, atol=1e-6):
        worst = float(np.max(np.abs(lhs - rhs_vals)))
        scope = "" if mask is None else " (within the stated claim type)"
        raise ValueError(
            f"Identity check failed{scope}: {target} != "
            + " + ".join(f"{c}*{k}" for k, c in rhs)
            + f" (max abs difference {worst:.6g}). Do not drop it; re-run "
            "explain_dependencies.py."
        )


def build_chow_design_matrix(
    df: pd.DataFrame, encoders: dict | None = None, return_encoders: bool = False
):
    """Encode the shared/2-of-3 covariates that needed cardinality
    treatment, then remove everything the 2026-09-24 rank-deficiency
    investigation proved redundant (module docstring, fixes A, B, D).

    Returns the shared intermediate both design matrices are built from.
    df.attrs["rank_fix_report"] records what was removed and why, for
    __main__ / callers to print (attrs is informational only).

    `encoders`/`return_encoders`, added 2026-09-25 for the Phase 2 baseline
    model (src/fit_baseline_model.py), which needs val/test encoded against
    TRAIN's categories/frequencies, not their own -- see top_n_encode's and
    frequency_encode's docstrings. Every existing caller (the Chow-test
    scripts) passes neither, so gets exactly the old behavior: encoders
    fit fresh from `df` and a bare DataFrame returned, unchanged.

    Leading .copy() kept -- protects the caller's train_model.parquet frame.
    """
    df = df.copy()
    encoders = dict(encoders) if encoders else {}

    binary_downcast = {col: _BINARY_DTYPE for col in (*_CLAIM_TYPE_COLS, *_NPI_FLAGS) if col in df.columns}
    if binary_downcast:
        df = df.astype(binary_downcast)

    masks = _claim_type_masks(df)

    # --- Fix B: verify, then drop, the named exact identities ---
    identity_dropped: list[str] = []
    for target, rhs in _REDUNDANT_RAW_IDENTITIES:
        if not all(c in df.columns for c in (target, *(k for k, _ in rhs))):
            continue
        _verify_identity(df, target, rhs)
        identity_dropped.append(target)
    for col, ct, keep in _REDUNDANT_WITHIN_TYPE:
        if col in df.columns and keep in df.columns:
            _verify_identity(df, col, ((keep, 1.0),), mask=masks[ct])
    df = df.drop(columns=identity_dropped)

    # --- Encoding (fix D: CARR_NUM no longer frequency-encoded) ---
    # `categories=`/`freq_map=` fix WHICH values map to which bucket; that
    # alone isn't enough to guarantee identical output COLUMNS across
    # splits, since pd.get_dummies only emits a column for a category that
    # actually appears in the data handed to it -- a category absent from
    # val's rows (e.g. an __MISSING__ bucket with no NaN in val, or a state
    # with zero val claims) would otherwise silently vanish from val's
    # matrix instead of appearing as an all-zero column. So every dummy
    # block is additionally reindexed to the exact column list recorded
    # from the FIRST (train) call -- fill_value=0 for anything missing,
    # and reindex also drops anything unexpected (there shouldn't be any,
    # since categories/freq_map already fix the value-to-bucket mapping).
    state_dummies = pd.get_dummies(
        df["provider_state"], prefix="state", drop_first=True, dtype=_BINARY_DTYPE
    )
    if "state_columns" in encoders:
        state_dummies = state_dummies.reindex(columns=encoders["state_columns"], fill_value=0).astype(_BINARY_DTYPE)
    else:
        encoders["state_columns"] = list(state_dummies.columns)

    hcpcs_dummies, hcpcs_cats = top_n_encode(
        df["HCPCS_CD"], TOP_N_HCPCS, prefix="hcpcs", categories=encoders.get("hcpcs_categories")
    )
    encoders["hcpcs_categories"] = hcpcs_cats
    if "hcpcs_columns" in encoders:
        hcpcs_dummies = hcpcs_dummies.reindex(columns=encoders["hcpcs_columns"], fill_value=0).astype(_BINARY_DTYPE)
    else:
        encoders["hcpcs_columns"] = list(hcpcs_dummies.columns)

    dgns_dummies, dgns_cats = top_n_encode(
        df["PRNCPAL_DGNS_CD"], TOP_N_DGNS, prefix="dgns", categories=encoders.get("dgns_categories")
    )
    encoders["dgns_categories"] = dgns_cats
    if "dgns_columns" in encoders:
        dgns_dummies = dgns_dummies.reindex(columns=encoders["dgns_columns"], fill_value=0).astype(_BINARY_DTYPE)
    else:
        encoders["dgns_columns"] = list(dgns_dummies.columns)

    prvdr_freq, prvdr_freq_map = frequency_encode(df["PRVDR_NUM"], freq_map=encoders.get("prvdr_num_freq_map"))
    encoders["prvdr_num_freq_map"] = prvdr_freq_map
    freq_encoded = pd.DataFrame({"prvdr_num_freq": prvdr_freq}, index=df.index)
    df = pd.concat([df, state_dummies, hcpcs_dummies, dgns_dummies, freq_encoded], axis=1)
    df = df.drop(columns=["provider_state", "HCPCS_CD", "PRNCPAL_DGNS_CD", "PRVDR_NUM", "CARR_NUM"])

    # --- Fix A: drop claim-type-determined columns ---
    # Reused verbatim from `encoders` when given (val/test), rather than
    # recomputed from this split's own data: a column that happens to be
    # constant within one claim type in train but NOT in val (or the
    # reverse) would otherwise make train's and val's intermediates
    # disagree on which columns even exist -- silently breaking the fixed
    # design-matrix column set the baseline model needs. train's decision
    # governs; a warning is printed (not raised) if val's own constancy
    # pattern would have disagreed, since that's a real, if second-order,
    # distribution-shift signal worth seeing, not a build-breaking one.
    zero_with_gaps_warn: list[str] = []
    if "determined_dropped" in encoders:
        determined_dropped = [c for c in encoders["determined_dropped"] if c in df.columns]
        disagreements = []
        for col in determined_dropped:
            statuses = [_within_type_status(df[col], masks[ct]) for ct in _CLAIM_TYPES]
            present = [s for s in statuses if s != "absent"]
            if present and not all(s in _DETERMINED for s in present):
                disagreements.append(col)
        if disagreements:
            print(
                f"    [note] {len(disagreements)} column(s) train found claim-type-determined "
                f"are NOT determined on this split's own data (kept dropped, per train's "
                f"decision, for a fixed column set): {disagreements}"
            )
    else:
        determined_dropped = []
        for col in df.columns:
            if col in _NEVER_INTERACT or col.startswith("risk_"):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            statuses = [_within_type_status(df[col], masks[ct]) for ct in _CLAIM_TYPES]
            present = [s for s in statuses if s != "absent"]
            if present and all(s in _DETERMINED for s in present):
                determined_dropped.append(col)
                if "zero_with_gaps" in present:
                    zero_with_gaps_warn.append(col)
        encoders["determined_dropped"] = determined_dropped
    df = df.drop(columns=determined_dropped)

    df.attrs["rank_fix_report"] = {
        "identity_dropped": identity_dropped,
        "within_type_redundant_terms": sorted(_REDUNDANT_INTERACTION_TERMS),
        "claim_type_determined_dropped": determined_dropped,
        "zero_with_gaps_warning": zero_with_gaps_warn,
        "carr_num_freq_removed": True,
    }
    if return_encoders:
        return df, encoders
    return df


def _null_pattern(df: pd.DataFrame, col: str) -> tuple[list[str], list[str]]:
    """(absent_types, present_types) -- which claim types a numeric column
    is 100% null in versus actually populated in. Shared by both
    construction functions so they can never disagree."""
    null_by_type = {
        ct: df.loc[df[f"claim_type_{ct}"] == 1, col].isna().mean() for ct in _CLAIM_TYPES
    }
    absent_types = [ct for ct, rate in null_by_type.items() if rate == 1.0]
    present_types = [ct for ct in _CLAIM_TYPES if ct not in absent_types]
    return absent_types, present_types


def _interact_with_zero_variance_guard(
    df: pd.DataFrame, cols: list[str], masks: dict[str, np.ndarray] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """For each binary dummy in `cols`, build dummy*claim_type_dummy for
    each claim type, SKIPPING any term that is constant within that claim
    type (all 0: the category never occurs there -- the original
    2026-09-22 guard, e.g. 13 HCPCS categories x DME; or all 1:
    dummy*claim_type == claim_type exactly -- added 2026-09-24).

    Then (2026-09-24) the within-claim-type dummy trap: for each prefix
    group and claim type, if the group's kept terms sum to exactly the
    claim_type dummy (the global drop_first reference never occurs in this
    claim type), drop one term in that block as the within-claim-type
    reference -- preferring a category that also has a term in another
    claim type, most frequent first. See module docstring, fix C.

    Returns (df, new_interaction_cols, dropped_cols) -- dropped_cols covers
    both skip reasons, for reporting.
    """
    if masks is None:
        masks = _claim_type_masks(df)
    new_cols: dict[str, pd.Series] = {}
    dropped: list[str] = []

    for col in cols:
        for ct in _CLAIM_TYPES:
            new_col = f"{col}__x__{ct}"
            if _within_type_status(df[col], masks[ct]) in _DETERMINED:
                dropped.append(new_col)
                continue
            new_cols[new_col] = df[col] * df[f"claim_type_{ct}"]

    # Within-claim-type dummy trap. Reference choice: prefer a category that
    # still has a term in ANOTHER claim type, so dropping this one leaves it
    # with n_i >= 1 -- dropping a category's ONLY term (e.g. a DME-only HCPCS
    # code in the DME block) keeps nesting valid but leaves a pooled column
    # with no unrestricted counterpart (n_i = 0, contributing -1 df), which
    # fit_chow_test.py's name-based df then has to special-case. Caught on
    # synthetic data before the first real run (2026-09-24). Among eligible
    # terms, the most frequent is the reference (most stable baseline).
    n_terms = Counter(c.rsplit("__x__", 1)[0] for c in new_cols)
    for prefix in _GENUINELY_SHARED_PREFIXES:
        for ct in _CLAIM_TYPES:
            suffix = f"__x__{ct}"
            block = [c for c in new_cols if c.startswith(prefix) and c.endswith(suffix)]
            if not block:
                continue
            block_sum = np.zeros(len(df), dtype=np.int32)
            for c in block:
                block_sum += new_cols[c].to_numpy(dtype=np.int32)
            if np.array_equal(block_sum, df[f"claim_type_{ct}"].to_numpy(dtype=np.int32)):
                shared = [c for c in block if n_terms[c.rsplit("__x__", 1)[0]] >= 2]
                ref = max(shared or block, key=lambda c: int(new_cols[c].sum()))
                del new_cols[ref]
                n_terms[ref.rsplit("__x__", 1)[0]] -= 1
                dropped.append(ref)

    df = df.drop(columns=cols)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df, list(new_cols.keys()), dropped


def add_claim_type_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Builds the UNRESTRICTED (fully-interacted) model's design matrix.

    Pass 1 -- numeric claim-type-exclusive/2-of-3 covariates: one
    value*claim_type term per claim type the covariate is present in,
    zero-filled on this copy only -- SKIPPING (2026-09-24) terms where the
    value is constant in that claim type (fix C) and the named within-DME
    duplicates (fix B).

    Pass 2 -- genuinely-3-of-3 categorical dummy groups, via
    _interact_with_zero_variance_guard() (constancy guard + within-claim-
    type dummy trap).

    No leading .copy() -- never mutates its input (SECOND MEMORY NOTE).

    Returns (df_with_interactions, new_interaction_cols, skipped_cols).
    """
    masks = _claim_type_masks(df)
    interaction_cols: list[str] = []
    skipped: list[str] = []
    cols_to_drop: list[str] = []
    new_cols: dict[str, pd.Series] = {}

    # --- Pass 1: numeric claim-type-exclusive / 2-of-3 covariates ---
    for col in list(df.columns):
        if col in _NEVER_INTERACT or col in _NPI_FLAGS:
            continue
        if col.startswith(_GENUINELY_SHARED_PREFIXES):
            continue
        if col.startswith("risk_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        absent_types, present_types = _null_pattern(df, col)
        if not absent_types:
            continue

        filled = df[col].fillna(0)
        for ct in present_types:
            new_col = f"{col}__x__{ct}"
            if new_col in _REDUNDANT_INTERACTION_TERMS:
                skipped.append(new_col)
                continue
            if _within_type_status(df[col], masks[ct]) in _DETERMINED:
                skipped.append(new_col)
                continue
            new_cols[new_col] = filled * df[f"claim_type_{ct}"]
            interaction_cols.append(new_col)
        cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # --- Pass 2: genuinely-3-of-3 categorical dummy groups ---
    shared_dummy_cols = [c for c in df.columns if c.startswith(_GENUINELY_SHARED_PREFIXES)]
    shared_dummy_cols += [c for c in df.columns if c in _NPI_FLAGS]
    df, new_pass2_cols, pass2_dropped = _interact_with_zero_variance_guard(df, shared_dummy_cols, masks)
    interaction_cols += new_pass2_cols

    return df, interaction_cols, skipped + pass2_dropped


def build_restricted_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the RESTRICTED (pooled) model's design matrix.

    - Claim-type-EXCLUSIVE fields (n_i=1): identical to the unrestricted
      model -- one interaction term against their one claim type. The
      named within-DME duplicates (fix B) are skipped here too when they
      take this form, so both matrices stay name-consistent.
    - Genuinely SHARED covariates: ONE plain column each (the actual
      restriction under test); 2-of-3 covariates zero-filled for their
      absent claim type.

    Leading .copy() kept -- this function mutates columns in place.
    """
    df = df.copy()

    cols_to_drop: list[str] = []
    new_cols: dict[str, pd.Series] = {}

    for col in list(df.columns):
        if col in _NEVER_INTERACT or col in _NPI_FLAGS:
            continue
        if col.startswith(_GENUINELY_SHARED_PREFIXES):
            continue
        if col.startswith("risk_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        absent_types, present_types = _null_pattern(df, col)
        if not absent_types:
            continue

        if len(present_types) == 1:
            ct = present_types[0]
            new_col = f"{col}__x__{ct}"
            cols_to_drop.append(col)
            if new_col in _REDUNDANT_INTERACTION_TERMS:
                continue
            new_cols[new_col] = df[col].fillna(0) * df[f"claim_type_{ct}"]
        else:
            df[col] = df[col].fillna(0)

    df = df.drop(columns=cols_to_drop)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def build_full_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run stages 1+2 -- the UNRESTRICTED model's design matrix."""
    design = build_chow_design_matrix(df)
    design, interaction_cols, skipped = add_claim_type_interactions(design)
    return design, interaction_cols, skipped


# ---------------------------------------------------------------------------
# Phase 2 baseline mixed model (added 2026-09-25)
# ---------------------------------------------------------------------------
# The model Chow Stage 2 actually selected (reports/chow_stage2_results__firth.txt,
# 2026-09-24 corrected-label run): 5 of the 11 genuinely-shared covariates
# need claim-type-specific effects (kept as interaction terms, same as the
# unrestricted matrix); the other 6 can share one pooled coefficient (kept
# as a single column, same as the restricted matrix). Every claim-type-
# EXCLUSIVE covariate (n_i=1) is unaffected either way -- both matrices
# already treat those identically.
MIXED_INTERACT_NUMERIC = {"prvdr_num_freq", "CARR_CLM_CASH_DDCTBL_APLD_AMT"}
MIXED_INTERACT_PREFIXES = ("state_", "hcpcs_", "dgns_")  # provider_state, HCPCS_CD, PRNCPAL_DGNS_CD
MIXED_POOL_NUMERIC = {
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "LINE_BENE_PTB_DDCTBL_AMT",
    "LINE_SRVC_CNT",
    "LINE_ALOWD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_NCH_PMT_AMT",
}


def build_mixed_design_matrix(intermediate: pd.DataFrame) -> pd.DataFrame:
    """Builds the Phase 2 baseline model's design matrix from an already-
    built shared intermediate (build_chow_design_matrix() output, called by
    the caller so it can pass `encoders=`/`return_encoders=` for val/test --
    see that function's docstring).

    Implementation: build the UNRESTRICTED matrix (every shared covariate
    interacted, exactly like the Chow test), then for MIXED_POOL_NUMERIC
    only, replace its per-claim-type interaction term(s) with ONE pooled
    column taken from the RESTRICTED matrix. Reusing both existing,
    already-verified builders (rather than writing a third variant of the
    same claim-type logic) means every rank-deficiency fix (A-D) and every
    presence/constancy edge case they handle is inherited automatically,
    not re-solved here.

    Does NOT itself guarantee train/val column alignment (add_claim_type_
    interactions' per-split skip decisions -- fix C -- can differ trivially
    between splits, e.g. a category constant in val's claim type by chance
    but not train's). The caller (fit_baseline_model.py) builds train's
    matrix first, then reindexes val/test's matrix to train's exact column
    list (fill_value=0) -- the same fixed-vocabulary discipline already
    applied to the categorical/frequency encoders themselves.
    """
    unrestricted, _, _ = add_claim_type_interactions(intermediate)
    restricted = build_restricted_design_matrix(intermediate)

    pool_interaction_cols = [
        c for c in unrestricted.columns
        if "__x__" in c and c.rsplit("__x__", 1)[0] in MIXED_POOL_NUMERIC
    ]
    mixed = unrestricted.drop(columns=pool_interaction_cols)
    pooled_present = [v for v in MIXED_POOL_NUMERIC if v in restricted.columns]
    missing = MIXED_POOL_NUMERIC - set(pooled_present)
    if missing:
        raise ValueError(
            f"build_mixed_design_matrix: expected pooled column(s) not found in the "
            f"restricted design matrix -- {sorted(missing)}. Did Stage 2's variable "
            "list change? Update MIXED_POOL_NUMERIC/MIXED_INTERACT_NUMERIC to match "
            "the current reports/chow_stage2_results__firth.txt."
        )
    mixed = pd.concat([mixed, restricted[pooled_present]], axis=1)
    return mixed


if __name__ == "__main__":
    train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
    intermediate = build_chow_design_matrix(train)
    report = intermediate.attrs.get("rank_fix_report", {})

    unrestricted, interaction_cols, skipped = add_claim_type_interactions(intermediate)
    restricted = build_restricted_design_matrix(intermediate)

    print(f"train_model.parquet:        {train.shape[1]} columns")
    print(f"unrestricted design matrix: {unrestricted.shape[1]} columns ({len(interaction_cols)} interaction terms)")
    print(f"restricted design matrix:   {restricted.shape[1]} columns")

    print("\n2026-09-24 rank-deficiency fixes:")
    print(f"  [B] identity-redundant columns dropped ({len(report.get('identity_dropped', []))}): {report.get('identity_dropped')}")
    print(f"  [B] within-DME redundant terms skipped: {report.get('within_type_redundant_terms')}")
    det = report.get("claim_type_determined_dropped", [])
    print(f"  [A] claim-type-determined columns dropped ({len(det)}): {det}")
    if report.get("zero_with_gaps_warning"):
        print(
            f"  [A] WARNING -- partly-NaN, all-present-values-0 (dropped; zero-fill conflates "
            f"a present 0 with absence here): {report['zero_with_gaps_warning']}"
        )
    npi_left = [c for c in _NPI_FLAGS if c in intermediate.columns]
    print(f"  [A] NPI flags remaining (expect none): {npi_left or 'none'}")
    print(f"  [D] carr_num_freq present anywhere (expect False): {any(c.startswith('carr_num_freq') for c in unrestricted.columns) or 'carr_num_freq' in restricted.columns}")
    trap_refs = [c for c in skipped if c.startswith(_GENUINELY_SHARED_PREFIXES) and "__x__" in c]
    print(f"  [C] unrestricted terms skipped ({len(skipped)} total; {len(trap_refs)} from dummy groups)")

    unrestricted_mb = unrestricted.memory_usage(deep=True).sum() / 1_048_576
    restricted_mb = restricted.memory_usage(deep=True).sum() / 1_048_576
    print(f"\nunrestricted design matrix memory usage: {unrestricted_mb:.1f} MiB")
    print(f"restricted design matrix memory usage:   {restricted_mb:.1f} MiB")

    for col in ("prvdr_num_freq",):
        present = col in restricted.columns
        no_nan = restricted[col].isna().sum() == 0 if present else False
        print(f"[restricted] {col}: single plain column={present}, no NaN={no_nan}")

    col_diff = unrestricted.shape[1] - restricted.shape[1]
    print(f"\nColumn count difference (unrestricted - restricted): {col_diff}")
    print(f"Row count match: {len(unrestricted) == len(restricted) == len(train)}")
    print("Full-rank verification: run src/check_rank.py (or fit_chow_test.py, which now enforces it).")
