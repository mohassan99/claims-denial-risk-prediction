"""
Phase 2 -- build the design matrices for the Chow test (and, eventually, the
Phase 2 baseline logistic regression). Operates on a COPY of
train_model.parquet only -- never mutates the shared file XGBoost also
reads from. See data/FEATURE_ENGINEERING.md Section 5 for the full
reasoning behind each encoding choice below.

Run after build_features.py.

Three stages:
  1. build_chow_design_matrix() -- cardinality encoding for the 5 shared/
     2-of-3 covariates that needed it (HCPCS_CD, PRNCPAL_DGNS_CD, PRVDR_NUM,
     CARR_NUM, provider_state).
  2. add_claim_type_interactions() -- builds the UNRESTRICTED (fully-
     interacted) model's design matrix: zero-fill-on-a-copy for every
     remaining numeric claim-type-exclusive/2-of-3 covariate, building
     value x claim_type_dummy terms only for the claim type(s) each
     covariate is actually populated in (FEATURE_ENGINEERING.md Section 3's
     degrees-of-freedom correction). Run via build_full_design_matrix().
  3. build_restricted_design_matrix() -- builds the RESTRICTED (pooled)
     model's design matrix: every genuinely shared covariate (2-of-3 or
     3-of-3) as ONE plain column instead of separate per-claim-type
     interaction terms -- this is the actual restriction the Chow test's
     likelihood-ratio comparison is testing. Claim-type-EXCLUSIVE fields
     (n_i=1) get identical treatment in both models (one interaction term,
     no restriction possible with only one claim type to begin with --
     FEATURE_ENGINEERING.md Section 3 checklist item 2), so this function
     shares that logic with add_claim_type_interactions() rather than
     re-deriving it.

Both design matrices are built from the SAME intermediate
(build_chow_design_matrix() output), so they cover identical rows and an
identical universe of covariates -- required for the likelihood-ratio
test's nesting assumption to actually hold (pre-flight checklist item 1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TOP_N_HCPCS = 20
TOP_N_DGNS = 20

_CLAIM_TYPES = ("carrier", "outpatient", "dme")

# Columns never touched by the interaction-construction logic -- identifiers,
# dates needing their own transformation (FEATURE_ENGINEERING.md Section 3
# checklist item 2), the label, and the claim_type dummies themselves (which
# are the interaction PARTNER, not something to interact against itself).
_NEVER_INTERACT = {
    "BENE_ID", "CLM_ID", "_row_id", "is_denied",
    "claim_type_carrier", "claim_type_outpatient", "claim_type_dme",
    "CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT",
}

# The 5 NPI presence flags -- confirmed 3-of-3, 0% null everywhere
# (FEATURE_ENGINEERING.md Section 3 checklist item 4). Not prefix-matchable
# like the one-hot dummies below, so listed explicitly.
_NPI_FLAGS = {
    "has_referring_physician", "has_performing_physician",
    "has_attending_physician", "has_operating_physician",
    "has_rendering_physician",
}

# Prefixes for the one-hot dummies build_chow_design_matrix() already
# produced -- 3-of-3 shared covariates encoded as plain columns by design,
# with no NaN left to zero-fill. Identical in both the restricted and
# unrestricted design matrices as currently scoped (see both functions'
# docstrings for the Stage-2/per-variable-testing caveat on this).
_ALREADY_ENCODED_PREFIXES = ("state_", "hcpcs_", "dgns_")


def top_n_encode(series: pd.Series, n: int, prefix: str) -> pd.DataFrame:
    """One-hot encode the top-n most frequent categories. Everything else
    buckets into '__OTHER__'; real NaN buckets into its own '__MISSING__'
    category rather than being silently dropped or folded into '__OTHER__'.

    This is also the resolution to HCPCS_CD's within-carrier-missingness
    question (FEATURE_ENGINEERING.md Section 3 checklist item 4, left open
    since 2026-09-15): the '__MISSING__' dummy IS the missing-indicator that
    was anticipated there, arrived at via the same general-purpose encoding
    every other high-cardinality shared covariate needed anyway, rather than
    a bespoke flag built just for HCPCS_CD.

    Reference-cell one-hot (drop_first=True), not cell-means -- the
    cell-means requirement documented for claim_type only applies to fields
    used in the claim-type-interaction construction; this isn't that.
    """
    top_categories = series.value_counts(dropna=True).head(n).index.tolist()
    bucketed = series.where(series.isin(top_categories), other="__OTHER__")
    bucketed = bucketed.where(series.notna(), other="__MISSING__")
    return pd.get_dummies(bucketed, prefix=prefix, drop_first=True, dtype=int)


def frequency_encode(series: pd.Series) -> pd.Series:
    """Map each category to its own frequency (share of non-null rows) --
    for PRVDR_NUM/CARR_NUM, whose high cardinality makes one-hot infeasible.

    Real NaN is preserved, NOT filled: both are confirmed 2-of-3 shared
    covariates, and per the Chow-test design their NaN for the absent claim
    type must stay NaN so the interaction-construction functions below can
    correctly omit (unrestricted) or zero-fill-as-one-column (restricted)
    that claim type. Fabricating a frequency value here would repeat the
    exact 0-vs-missing mistake the 2026-09-16 zero-fill correction was
    written to prevent, just for a different encoding scheme.
    """
    freq = series.value_counts(normalize=True, dropna=True)
    return series.map(freq)


def build_chow_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the 5 shared/2-of-3 covariates that needed cardinality
    treatment before they could enter any regression at all, then drop
    their raw (string/high-cardinality) columns -- the encoded versions
    replace them entirely, nothing left over half-encoded.

    ADDED 2026-09-17: CARR_NUM was missed in the original 2026-09-17 pass --
    the pre-flight checklist named it alongside PRVDR_NUM as needing
    cardinality treatment (both are confirmed 2-of-3 covariates, both
    high-cardinality provider identifiers), but only PRVDR_NUM was actually
    implemented. Caught while building the interaction-term construction
    below, which would otherwise have tried to multiply a raw provider-
    number STRING against a claim_type dummy -- a clear failure that
    surfaced the gap rather than a silent one.

    This is the shared starting point for BOTH the restricted and
    unrestricted design matrices below -- neither function re-derives this
    encoding.
    """
    df = df.copy()

    state_dummies = pd.get_dummies(
        df["provider_state"], prefix="state", drop_first=True, dtype=int
    )
    hcpcs_dummies = top_n_encode(df["HCPCS_CD"], TOP_N_HCPCS, prefix="hcpcs")
    dgns_dummies = top_n_encode(df["PRNCPAL_DGNS_CD"], TOP_N_DGNS, prefix="dgns")
    df = pd.concat([df, state_dummies, hcpcs_dummies, dgns_dummies], axis=1)

    df["prvdr_num_freq"] = frequency_encode(df["PRVDR_NUM"])
    df["carr_num_freq"] = frequency_encode(df["CARR_NUM"])

    # Drop the raw columns now that their encoded replacements exist --
    # leaving them in would just be dead, unusable string columns sitting
    # in what's supposed to be a numeric design matrix.
    df = df.drop(columns=["provider_state", "HCPCS_CD", "PRNCPAL_DGNS_CD", "PRVDR_NUM", "CARR_NUM"])

    return df


def _null_pattern(df: pd.DataFrame, col: str) -> tuple[list[str], list[str]]:
    """Shared helper: for a numeric column, return (absent_types,
    present_types) -- which claim types it's 100% null in versus actually
    populated in. Used identically by both the restricted and unrestricted
    construction functions so they can never silently disagree about which
    claim types a given covariate applies to."""
    null_by_type = {
        ct: df.loc[df[f"claim_type_{ct}"] == 1, col].isna().mean() for ct in _CLAIM_TYPES
    }
    absent_types = [ct for ct, rate in null_by_type.items() if rate == 1.0]
    present_types = [ct for ct in _CLAIM_TYPES if ct not in absent_types]
    return absent_types, present_types


def add_claim_type_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Builds the UNRESTRICTED (fully-interacted) model's design matrix: for
    every remaining NUMERIC column whose nullness is structurally
    determined by claim_type -- whether that's a claim-type-EXCLUSIVE field
    (~166 REV_CNTR_*/DMERC_LINE_*/CARR_CLM_*-family columns, populated in
    exactly 1 claim type) or a confirmed 2-of-3 SHARED covariate
    (carr_num_freq, prvdr_num_freq) -- build interaction terms against only
    the claim type(s) it's actually populated in, per
    FEATURE_ENGINEERING.md Section 3's degrees-of-freedom correction.

    Both cases get IDENTICAL treatment here: a claim-type-exclusive field
    (n_i=1) and a 2-of-3 shared covariate (n_i=2) differ only in how many
    interaction terms they get (1 vs 2) -- the mechanism (zero-fill on this
    copy, build value*dummy for each present claim type, omit the term for
    each absent one) is the same either way.

    Runs on the COPY returned by build_chow_design_matrix() -- the raw
    zero-fill happens HERE, on this copy, never in train_model.parquet
    itself (FEATURE_ENGINEERING.md Section 3 checklist item 3's correction).

    Returns (df_with_interactions, list_of_new_interaction_column_names) --
    the second value is for the not-yet-written Chow-test fitting step,
    which needs to know exactly which columns are "unrestricted-model-only"
    versus shared with the restricted model.
    """
    df = df.copy()

    interaction_cols: list[str] = []
    cols_to_drop: list[str] = []

    for col in list(df.columns):
        if col in _NEVER_INTERACT or col in _NPI_FLAGS:
            continue
        if col.startswith(_ALREADY_ENCODED_PREFIXES):
            continue
        if col.startswith("risk_"):  # defensive -- should already be absent from train_model.parquet
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Non-numeric claim-type-exclusive/shared columns are out of
            # scope here -- this treatment only ever applies to NUMERIC
            # dollar/count fields (FEATURE_ENGINEERING.md Section 3's
            # original framing). Checked directly against real data
            # (2026-09-17): of 84 non-numeric columns still carrying real
            # NaN in train_model.parquet, only CARR_NUM/PRVDR_NUM were ever
            # claim-type-exclusive, and both are already consumed into
            # numeric carr_num_freq/prvdr_num_freq above before this loop
            # runs. Every other flagged column (ICD_DGNS_CD*/
            # ICD_PRCDR_CD* family, HCPCS_CD, BETOS_CD, etc.) is ordinary,
            # legitimate partial real-world missingness in a non-claim-
            # type-exclusive field -- never a candidate for this treatment.
            continue

        absent_types, present_types = _null_pattern(df, col)

        if not absent_types:
            # Genuinely shared 3-of-3 (or no missingness at all) -- not a
            # claim-type-exclusive/2-of-3 field. This function only builds
            # interaction terms for fields that NEED them (some claim type
            # structurally lacks the field); a true 3-of-3 shared covariate
            # is Stage 2 (per-variable homogeneity testing) territory, not
            # something to interact by default here. Left untouched --
            # still present in the final design matrix as a plain column.
            continue

        filled = df[col].fillna(0)  # zero-fill ONLY here, on this copy
        for ct in present_types:
            new_col = f"{col}__x__{ct}"
            df[new_col] = filled * df[f"claim_type_{ct}"]
            interaction_cols.append(new_col)

        # Drop the raw column: keeping it alongside its own interaction
        # terms would recreate the same collinearity problem already
        # documented for claim_type's own encoding (Section 3) -- the raw
        # column still has real NaN (unfittable) and is redundant with the
        # interaction terms that now carry its information.
        cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)
    return df, interaction_cols


def build_restricted_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the RESTRICTED (pooled) model's design matrix -- the actual
    counterpart being compared against add_claim_type_interactions()'s
    unrestricted output in the Chow test's likelihood-ratio comparison.

    Two categories, given DIFFERENT treatment on purpose -- this is where
    the restricted and unrestricted matrices genuinely diverge, which is
    exactly what the Chow test needs to have something to test:

    - Claim-type-EXCLUSIVE fields (n_i=1): IDENTICAL to the unrestricted
      model -- one interaction term against their one applicable
      claim_type dummy, zero-filled on this copy. There's no "pooled vs.
      interacted" distinction possible when a covariate exists in only one
      claim type to begin with (FEATURE_ENGINEERING.md Section 3 checklist
      item 2 -- these sit outside the hypothesis being tested, in both
      models, for the same reason). Confirmed algebraically too: n_i=1
      contributes n_i-1=0 degrees of freedom to the LR test either way.
    - Genuinely SHARED covariates (n_i in {2,3} -- carr_num_freq/
      prvdr_num_freq, and the already-encoded 3-of-3 dummies/NPI flags):
      appear as ONE plain column each here, forced to a single shared
      coefficient -- versus n_i separate interaction terms in the
      unrestricted model. THIS is the actual restriction under test: does
      forcing one shared coefficient across claim types cost a
      significant amount of fit, versus letting each claim type have its
      own? 2-of-3 covariates get zero-filled on this copy for their one
      absent claim type, so the single shared column has no raw NaN left
      to break the fit.

    Runs on the COPY returned by build_chow_design_matrix() -- same
    starting point as add_claim_type_interactions(), so both design
    matrices cover identical rows and an identical covariate universe
    (required for the likelihood-ratio test's nesting assumption to hold --
    pre-flight checklist item 1).
    """
    df = df.copy()

    for col in list(df.columns):
        if col in _NEVER_INTERACT or col in _NPI_FLAGS:
            continue
        if col.startswith(_ALREADY_ENCODED_PREFIXES):
            continue
        if col.startswith("risk_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        absent_types, present_types = _null_pattern(df, col)

        if not absent_types:
            # Genuinely shared 3-of-3, no NaN -- already a valid single
            # plain column, nothing to do.
            continue

        if len(present_types) == 1:
            # Claim-type-exclusive (n_i=1): identical treatment to the
            # unrestricted model -- see docstring above for why.
            ct = present_types[0]
            filled = df[col].fillna(0)
            df[f"{col}__x__{ct}"] = filled * df[f"claim_type_{ct}"]
            df = df.drop(columns=[col])
        else:
            # 2-of-3 shared covariate: the actual restriction being tested.
            # ONE plain column, zero-filled for the one absent claim type --
            # versus 2 separate interaction terms in the unrestricted model.
            df[col] = df[col].fillna(0)

    return df


def build_full_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run stages 1+2. This is the UNRESTRICTED (fully-interacted) model's
    design matrix. See build_restricted_design_matrix() for its pooled
    counterpart -- built separately, from the same starting point."""
    design = build_chow_design_matrix(df)
    design, interaction_cols = add_claim_type_interactions(design)
    return design, interaction_cols


if __name__ == "__main__":
    train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
    intermediate = build_chow_design_matrix(train)

    unrestricted, interaction_cols = add_claim_type_interactions(intermediate)
    restricted = build_restricted_design_matrix(intermediate)

    print(f"train_model.parquet:      {train.shape[1]} columns")
    print(f"unrestricted design matrix: {unrestricted.shape[1]} columns ({len(interaction_cols)} interaction terms)")
    print(f"restricted design matrix:   {restricted.shape[1]} columns")

    # Sanity check on PRVDR_NUM's NaN pattern -- CORRECTED 2026-09-17.
    # The original version of this check asserted prvdr_num_freq's NaN
    # count must equal carrier's row count EXACTLY. That was too strong a
    # claim: PRVDR_NUM is 100% null for carrier (structural, confirmed
    # 2026-09-15) PLUS a tiny separate residual gap within outpatient --
    # confirmed directly: outpatient shows 136 null PRVDR_NUM rows out of
    # 367,542 (~0.037%), not 0. Expected total is therefore carrier's full
    # count plus that small residual, not carrier's count alone.
    n_carrier = (train["claim_type_carrier"] == 1).sum()
    n_outpatient_residual = (
        train.loc[train["claim_type_outpatient"] == 1, "PRVDR_NUM"].isna().sum()
    )
    expected_nan = n_carrier + n_outpatient_residual
    n_nan = intermediate["prvdr_num_freq"].isna().sum()
    status = "OK" if n_nan == expected_nan else "MISMATCH -- investigate before using this design matrix"
    print(
        f"\nprvdr_num_freq NaN count (pre-interaction): {n_nan:,} "
        f"(expected: carrier {n_carrier:,} + outpatient residual {n_outpatient_residual:,} "
        f"= {expected_nan:,}) [{status}]"
    )

    # Sanity check on the interaction terms themselves: for a 2-of-3
    # covariate, exactly 2 interaction columns should exist in the
    # UNRESTRICTED matrix, never 3 -- a 3rd would mean the omit-third-
    # interaction logic failed and an always-zero column slipped through.
    prvdr_terms = [c for c in interaction_cols if c.startswith("prvdr_num_freq__x__")]
    carr_terms = [c for c in interaction_cols if c.startswith("carr_num_freq__x__")]
    print(f"\n[unrestricted] prvdr_num_freq interaction terms: {prvdr_terms} (expect exactly 2, never 3)")
    print(f"[unrestricted] carr_num_freq interaction terms:  {carr_terms} (expect exactly 2, never 3)")

    # Sanity check on the RESTRICTED matrix: the 2-of-3 covariates should
    # appear as a SINGLE plain column each (no __x__ suffix at all), with
    # zero NaN remaining -- the opposite pattern from the unrestricted
    # matrix's 2 separate interaction terms. This is the actual restriction
    # the Chow test is testing, so getting this backwards would silently
    # invalidate the whole comparison.
    for col in ("prvdr_num_freq", "carr_num_freq"):
        present = col in restricted.columns
        no_interaction_variant = not any(c.startswith(f"{col}__x__") for c in restricted.columns)
        no_nan = restricted[col].isna().sum() == 0 if present else False
        status = "OK" if (present and no_interaction_variant and no_nan) else "MISMATCH -- investigate"
        print(
            f"[restricted]   {col}: present_as_single_column={present}, "
            f"no_interaction_variant={no_interaction_variant}, no_nan={no_nan} [{status}]"
        )

    # Row-count/column-universe cross-check: the two matrices should differ
    # ONLY in how the 2-of-3 covariates are represented (1 column vs 2 each
    # = a net +2 column difference), not in row count or in any other
    # covariate's presence.
    col_diff = unrestricted.shape[1] - restricted.shape[1]
    print(f"\nColumn count difference (unrestricted - restricted): {col_diff} (expect exactly 2 -- one extra column per 2-of-3 covariate)")
    print(f"Row count match: {len(unrestricted) == len(restricted) == len(train)}")

    print(f"\nhcpcs_dummies columns: {[c for c in unrestricted.columns if c.startswith('hcpcs_')][:5]} ... ({sum(c.startswith('hcpcs_') for c in unrestricted.columns)} total)")
    print(f"dgns_dummies columns:  {[c for c in unrestricted.columns if c.startswith('dgns_')][:5]} ... ({sum(c.startswith('dgns_') for c in unrestricted.columns)} total)")
