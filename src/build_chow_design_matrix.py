"""
Phase 2 -- build the design matrix for the Chow test (and, eventually, the
Phase 2 baseline logistic regression). Operates on a COPY of
train_model.parquet only -- never mutates the shared file XGBoost also
reads from. See data/FEATURE_ENGINEERING.md Section 5 for the full
reasoning behind each encoding choice below.

Run after build_features.py.

Two stages, run in order by build_full_design_matrix():
  1. build_chow_design_matrix() -- cardinality encoding for the 5 shared/
     2-of-3 covariates that needed it (HCPCS_CD, PRNCPAL_DGNS_CD, PRVDR_NUM,
     CARR_NUM, provider_state).
  2. add_claim_type_interactions() -- the interaction-term construction
     itself: zero-fill-on-a-copy for every remaining numeric
     claim-type-exclusive/2-of-3 covariate, building value x claim_type_dummy
     terms only for the claim type(s) each covariate is actually populated
     in (FEATURE_ENGINEERING.md Section 3's degrees-of-freedom correction).

This produces the UNRESTRICTED (fully-interacted) model's design matrix.
The RESTRICTED (pooled) model's design matrix is simpler (each shared
covariate as one plain column, no interaction) and is built separately at
Chow-test-fitting time, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TOP_N_HCPCS = 20
TOP_N_DGNS = 20

_CLAIM_TYPES = ("carrier", "outpatient", "dme")

# Columns never touched by add_claim_type_interactions() -- identifiers,
# dates needing their own transformation (FEATURE_ENGINEERING.md Section 3
# checklist item 2), the label, and the claim_type dummies themselves (which
# are the interaction PARTNER, not something to interact against itself).
_NEVER_INTERACT = {
    "BENE_ID", "CLM_ID", "_row_id", "is_denied",
    "claim_type_carrier", "claim_type_outpatient", "claim_type_dme",
    "CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT",
}


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
    type must stay NaN so add_claim_type_interactions() below can correctly
    omit that claim type's term. Fabricating a frequency value there would
    repeat the exact 0-vs-missing mistake the 2026-09-16 zero-fill
    correction was written to prevent, just for a different encoding
    scheme.
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


def add_claim_type_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The interaction-term construction itself: for every remaining NUMERIC
    column whose nullness is structurally determined by claim_type --
    whether that's a claim-type-EXCLUSIVE field (~166 REV_CNTR_*/
    DMERC_LINE_*/CARR_CLM_*-family columns, populated in exactly 1 claim
    type) or a confirmed 2-of-3 SHARED covariate (carr_num_freq,
    prvdr_num_freq, now that build_chow_design_matrix() has encoded them
    into numeric form) -- build interaction terms against only the claim
    type(s) it's actually populated in, per
    FEATURE_ENGINEERING.md Section 3's degrees-of-freedom correction.

    Both cases get IDENTICAL treatment here, which is the point: a
    claim-type-exclusive field (n_i=1) and a 2-of-3 shared covariate
    (n_i=2) differ only in how many interaction terms they get (1 vs 2) --
    the mechanism (zero-fill on this copy, build value*dummy for each
    present claim type, omit the term for each absent one) is the same
    either way, and this function doesn't need to distinguish them by name.

    Runs on the COPY returned by build_chow_design_matrix() -- the raw
    zero-fill happens HERE, on this copy, never in train_model.parquet
    itself (FEATURE_ENGINEERING.md Section 3 checklist item 3's correction).

    Returns (df_with_interactions, list_of_new_interaction_column_names) --
    the second value is for the not-yet-written Chow-test fitting step,
    which needs to know exactly which columns are "unrestricted-model-only"
    versus shared with the restricted model.
    """
    df = df.copy()

    already_encoded_prefixes = ("state_", "hcpcs_", "dgns_")
    already_encoded_exact = {"prvdr_num_freq", "carr_num_freq"}

    interaction_cols: list[str] = []
    cols_to_drop: list[str] = []

    for col in list(df.columns):
        if col in _NEVER_INTERACT or col in already_encoded_exact:
            continue
        if col.startswith(already_encoded_prefixes):
            continue
        if col.startswith("risk_"):  # defensive -- should already be absent from train_model.parquet
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Any remaining non-numeric column at this point is a bug
            # elsewhere (build_chow_design_matrix should have encoded or
            # dropped every string column) -- surface it loudly rather than
            # silently skipping, since silently skipping a column that
            # SHOULD have been a covariate is exactly the kind of thing
            # this project has caught going wrong before.
            raise TypeError(
                f"Column '{col}' is non-numeric and wasn't encoded by "
                f"build_chow_design_matrix() -- add it there before calling "
                f"add_claim_type_interactions()."
            )

        null_by_type = {
            ct: df.loc[df[f"claim_type_{ct}"] == 1, col].isna().mean() for ct in _CLAIM_TYPES
        }
        absent_types = [ct for ct, rate in null_by_type.items() if rate == 1.0]
        present_types = [ct for ct in _CLAIM_TYPES if ct not in absent_types]

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


def build_full_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run both stages in order. This is the UNRESTRICTED (fully-interacted)
    model's design matrix -- see this module's docstring for what still
    needs to be built separately (the restricted/pooled design matrix, at
    Chow-test-fitting time)."""
    design = build_chow_design_matrix(df)
    design, interaction_cols = add_claim_type_interactions(design)
    return design, interaction_cols


if __name__ == "__main__":
    train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
    design, interaction_cols = build_full_design_matrix(train)

    print(f"train_model.parquet: {train.shape[1]} columns")
    print(f"design matrix:       {design.shape[1]} columns")
    print(f"interaction terms built: {len(interaction_cols)}")

    # Sanity check on PRVDR_NUM's NaN pattern -- CORRECTED 2026-09-17.
    # The original version of this check asserted prvdr_num_freq's NaN
    # count must equal carrier's row count EXACTLY. That was too strong a
    # claim: PRVDR_NUM is 100% null for carrier (structural, confirmed
    # 2026-09-15) PLUS a tiny separate residual gap within outpatient --
    # confirmed directly: outpatient shows 136 null PRVDR_NUM rows out of
    # 367,542 (~0.037%), not 0. That's the SAME kind of thing as HCPCS_CD's
    # within-carrier missingness (Section 5) -- ordinary, small,
    # within-claim-type gap -- just three orders of magnitude smaller, so
    # it never got its own separate note until this check caught it.
    # Expected total is therefore carrier's full count plus that small
    # residual, not carrier's count alone.
    #
    # NOTE: this check now runs against the INTERMEDIATE build_chow_design_
    # matrix() output, not the final design matrix -- prvdr_num_freq no
    # longer exists as a standalone column in the final output, since
    # add_claim_type_interactions() consumes it into
    # prvdr_num_freq__x__outpatient / prvdr_num_freq__x__dme and drops the
    # raw frequency column.
    intermediate = build_chow_design_matrix(train)
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
    # covariate, exactly 2 interaction columns should exist, never 3 --
    # a 3rd would mean the omit-third-interaction logic failed and an
    # always-zero column slipped through.
    prvdr_terms = [c for c in interaction_cols if c.startswith("prvdr_num_freq__x__")]
    carr_terms = [c for c in interaction_cols if c.startswith("carr_num_freq__x__")]
    print(f"\nprvdr_num_freq interaction terms: {prvdr_terms} (expect exactly 2, never 3)")
    print(f"carr_num_freq interaction terms:  {carr_terms} (expect exactly 2, never 3)")

    print(f"\nhcpcs_dummies columns: {[c for c in design.columns if c.startswith('hcpcs_')][:5]} ... ({sum(c.startswith('hcpcs_') for c in design.columns)} total)")
    print(f"dgns_dummies columns:  {[c for c in design.columns if c.startswith('dgns_')][:5]} ... ({sum(c.startswith('dgns_') for c in design.columns)} total)")
