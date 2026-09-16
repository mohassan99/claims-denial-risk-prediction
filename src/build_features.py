"""
Phase 1 completion -- apply the feature-engineering decisions documented in
data/FEATURE_ENGINEERING.md to train/val/test.parquet, consistently, in one
place, rather than reconstructing each fix by hand per session.

Run after build_target_and_split.py. Produces train_model.parquet,
val_model.parquet, test_model.parquet -- the Phase-2-ready feature sets.
train/val/test.parquet themselves are left untouched, in case anything here
needs revisiting later.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ---------------------------------------------------------------------------
# Decision: SSA-numeric -> USPS-alpha state crosswalk
# (FEATURE_ENGINEERING.md Section 1) -- PRVDR_STATE_CD mixes both conventions
# for the same states (e.g. "CA" and "05" both meaning California); this
# normalizes to one consistent alpha form. Codes not in the table (e.g. "00"
# Unknown/other, or international/territory codes 54+) pass through
# unchanged via fillna rather than being silently dropped.
# ---------------------------------------------------------------------------
SSA_STATE_CROSSWALK = {
    "01": "AL", "02": "AK", "03": "AZ", "04": "AR", "05": "CA", "06": "CO", "07": "CT",
    "08": "DE", "09": "DC", "10": "FL", "11": "GA", "12": "HI", "13": "ID", "14": "IL",
    "15": "IN", "16": "IA", "17": "KS", "18": "KY", "19": "LA", "20": "ME", "21": "MD",
    "22": "MA", "23": "MI", "24": "MN", "25": "MS", "26": "MO", "27": "MT", "28": "NE",
    "29": "NV", "30": "NH", "31": "NJ", "32": "NM", "33": "NY", "34": "NC", "35": "ND",
    "36": "OH", "37": "OK", "38": "OR", "39": "PA", "40": "PR", "41": "RI", "42": "SC",
    "43": "SD", "44": "TN", "45": "TX", "46": "UT", "47": "VT", "48": "VI", "49": "VA",
    "50": "WA", "51": "WV", "52": "WI", "53": "WY",
}

# ---------------------------------------------------------------------------
# Decision: 5 NPI role-fields -> binary presence flags, derived BEFORE the
# raw NPI values are dropped (FEATURE_ENGINEERING.md Section 1). PRVDR_NPI
# has no distinct "role" meaning versus PRVDR_NUM, so it gets no flag -- it's
# just dropped below.
# ---------------------------------------------------------------------------
NPI_ROLE_FIELDS = {
    "RFR_PHYSN_NPI": "has_referring_physician",
    "PRF_PHYSN_NPI": "has_performing_physician",
    "AT_PHYSN_NPI": "has_attending_physician",
    "OP_PHYSN_NPI": "has_operating_physician",
    "RNDRNG_PHYSN_NPI": "has_rendering_physician",
}

# ---------------------------------------------------------------------------
# Decision: columns dropped outright -- raw NPI values, PRVDR_ZIP, always-
# 100%-null columns, and zero-variance/collinear-with-claim_type columns
# (FEATURE_ENGINEERING.md Sections 1-3).
# ---------------------------------------------------------------------------
DROP_COLUMNS = [
    # Raw NPI values -- presence already captured above.
    "RFR_PHYSN_NPI", "PRF_PHYSN_NPI", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
    "RNDRNG_PHYSN_NPI", "PRVDR_NPI",
    "PRVDR_ZIP",
    # Always-100%-null across the whole dataset.
    "LINE_SERVICE_DEDUCTIBLE", "FI_CLM_PROC_DT", "OT_PHYSN_UPIN", "OT_PHYSN_NPI",
    "ICD_PRCDR_CD25", "PRCDR_DT25", "RSN_VISIT_CD1", "RSN_VISIT_CD2", "RSN_VISIT_CD3",
    "REV_CNTR_NDC_QTY",
    # Confirmed zero-variance 2026-09-13: single repeated value wherever
    # populated -- carries no information regardless of claim type.
    "PRVDR_SPCLTY",
    "ICD_DGNS_VRSN_CD1", "ICD_DGNS_VRSN_CD2", "ICD_DGNS_VRSN_CD3", "ICD_DGNS_VRSN_CD4",
    "ICD_DGNS_VRSN_CD5", "ICD_DGNS_VRSN_CD6", "ICD_DGNS_VRSN_CD7", "ICD_DGNS_VRSN_CD8",
    "ICD_DGNS_VRSN_CD9", "ICD_DGNS_VRSN_CD10", "ICD_DGNS_VRSN_CD11", "ICD_DGNS_VRSN_CD12",
    # Added 2026-09-16: PRNCPAL_DGNS_VRSN_CD is a 13th zero-variance field in
    # the same family as the ICD_DGNS_VRSN_CD group above -- populated (2 of
    # 3 claim types: carrier + DME) with a constant ICD-10 flag value, no
    # variance to model. See FEATURE_ENGINEERING.md Section 2.
    "PRNCPAL_DGNS_VRSN_CD",
    # Added 2026-09-16: confirmed via crosstab (FEATURE_ENGINEERING.md
    # Section 3, pre-flight checklist item 4) to be a perfect 1-to-1
    # re-encoding of claim_type under CMS's own coding scheme
    # (carrier->71/O, outpatient->40/W, dme->82/M). Including either would
    # recreate the claim_type dummy set under a different label and
    # reproduce the exact rank-deficiency bug already documented for the
    # applicability-indicator case.
    "NCH_CLM_TYPE_CD", "NCH_NEAR_LINE_REC_IDENT_CD",
]

# ---------------------------------------------------------------------------
# Decision: claim_type from _source_file, cell-means encoding (ALL 3 dummies,
# no dropped reference, no shared intercept assumed) -- see
# FEATURE_ENGINEERING.md Section 3's encoding-scheme discussion for why this
# specific variant is required (needed as interaction terms in Phase 2, not
# just as plain categorical features).
# ---------------------------------------------------------------------------
SOURCE_FILE_TO_CLAIM_TYPE = {
    "carrier.csv": "carrier",
    "outpatient.csv": "outpatient",
    "dme.csv": "dme",
}

_CLAIM_TYPE_COLS = ["claim_type_carrier", "claim_type_outpatient", "claim_type_dme"]

# ---------------------------------------------------------------------------
# Fields explicitly EXCLUDED from the generic claim-type-exclusive fill below
# -- these are either genuinely shared/multi-claim-type fields already given
# their own dedicated encoding (see FEATURE_ENGINEERING.md Section 3's
# finalized shared-feature audit), or non-feature/date columns that need
# their own transformation, not a fill.
#
# CARR_NUM and PRVDR_NUM specifically: confirmed 2-of-3 shared covariates.
# Per FEATURE_ENGINEERING.md's degrees-of-freedom correction, these should be
# LEFT with their real NaN for the one claim type where they're structurally
# absent -- the Chow-test design omits that claim type's interaction term
# entirely rather than zero-filling a column that would be constant at zero
# for the whole dataset. Zero-filling them here would be the exact mistake
# that correction was written to prevent.
#
# HCPCS_CD specifically: has its own, still-undecided within-carrier
# missingness question (62.5% null WITHIN carrier, not structural by claim
# type -- FEATURE_ENGINEERING.md Section 3 checklist item 4). Excluded here
# so that decision stays its own explicit commit once made, rather than
# silently absorbed into this generic pass.
# ---------------------------------------------------------------------------
_ALREADY_HANDLED = {
    "BENE_ID", "CLM_ID", "_row_id", "is_denied",
    "HCPCS_CD", "PRNCPAL_DGNS_CD", "PRVDR_NUM", "CARR_NUM",
    "provider_state",
    "has_referring_physician", "has_performing_physician", "has_attending_physician",
    "has_operating_physician", "has_rendering_physician",
}
_DATE_LIKE_COLS = {"CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT"}


def fill_claim_type_exclusive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill every column whose nullness is structurally determined by
    claim_type, so no raw NaN reaches the Phase 2 design matrix.
    (FEATURE_ENGINEERING.md Section 3, pre-flight checklist item 3.)

    Applicability is derived EMPIRICALLY at runtime -- the same
    null-rate-by-claim_type technique used for the shared-feature audit --
    rather than hardcoded against a fixed list of ~166 column names. This
    project has already found the source data dictionary wrong about which
    claim types a field applies to twice (PRNCPAL_DGNS_CD, PRVDR_NUM -- see
    FEATURE_ENGINEERING.md Section 3), so deriving this from the data itself
    is more reliable than trusting documentation or a hardcoded list.

    Numeric columns get zero-filled for the claim type(s) where they're
    structurally 100% null. String/object columns get an explicit
    "NOT_APPLICABLE" sentinel instead of 0 -- a blanket zero-as-missing rule
    already produced real false positives this session (the NPI presence
    flags and PRNCPAL_DGNS_VRSN_CD, where 0/0.0 is a legitimate value, not a
    missingness marker), so this function never reuses that pattern for
    anything but genuinely numeric columns.

    Only fills a column for the claim type(s) where it's structurally 100%
    null. Genuine WITHIN-claim-type missingness (e.g. HCPCS_CD's ~62.5% null
    rate within carrier specifically) is a different problem this function
    does not address -- such columns belong in _ALREADY_HANDLED so this
    function never touches them until that separate decision is made.
    """
    df = df.copy()
    if not all(c in df.columns for c in _CLAIM_TYPE_COLS):
        raise ValueError("claim_type_* dummies must exist before calling this function")

    skip = _ALREADY_HANDLED | _DATE_LIKE_COLS | set(_CLAIM_TYPE_COLS)

    for col in df.columns:
        if col in skip or col.startswith("claim_type_") or col.startswith("risk_"):
            continue

        null_by_type = {}
        for ct_col in _CLAIM_TYPE_COLS:
            mask = df[ct_col] == 1
            null_by_type[ct_col] = df.loc[mask, col].isna().mean() if mask.any() else 0.0

        # Structurally absent for a claim type = 100% null for that type.
        absent_types = [ct for ct, rate in null_by_type.items() if rate == 1.0]

        # Nothing structurally missing (genuinely shared 3-of-3, or the
        # column has no missingness at all) -- leave untouched.
        if not absent_types:
            continue

        absent_mask = df[absent_types].eq(1).any(axis=1)

        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[absent_mask, col] = 0
        else:
            df.loc[absent_mask, col] = "NOT_APPLICABLE"

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # NPI presence flags -- must run before the raw NPI columns are dropped.
    for npi_col, flag_col in NPI_ROLE_FIELDS.items():
        if npi_col in df.columns:
            df[flag_col] = df[npi_col].notna().astype(int)

    # Provider state: normalize, then drop the raw column -- keeping both
    # would just be two redundant encodings of the same information.
    if "PRVDR_STATE_CD" in df.columns:
        df["provider_state"] = df["PRVDR_STATE_CD"].map(SSA_STATE_CROSSWALK).fillna(
            df["PRVDR_STATE_CD"]
        )
        df = df.drop(columns=["PRVDR_STATE_CD"])

    # claim_type: cell-means one-hot, then drop _source_file for the same
    # redundancy reason.
    if "_source_file" in df.columns:
        claim_type = df["_source_file"].map(SOURCE_FILE_TO_CLAIM_TYPE)
        for ct in ("carrier", "outpatient", "dme"):
            df[f"claim_type_{ct}"] = (claim_type == ct).astype(int)
        df = df.drop(columns=["_source_file"])

    existing_drops = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Zero-fill (numeric) / sentinel-fill (categorical) every remaining
    # claim-type-exclusive field -- see fill_claim_type_exclusive_fields()
    # docstring above. Must run AFTER claim_type_* dummies exist and AFTER
    # DROP_COLUMNS has removed anything that shouldn't reach this pass.
    df = fill_claim_type_exclusive_fields(df)

    return df


def main() -> None:
    for name in ("train", "val", "test"):
        in_path = PROCESSED_DIR / f"{name}.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} not found -- run build_target_and_split.py first.")

        df = pd.read_parquet(in_path)
        n_cols_before = df.shape[1]
        df = build_features(df)
        n_cols_after = df.shape[1]

        out_path = PROCESSED_DIR / f"{name}_model.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{name}: {n_cols_before} -> {n_cols_after} columns -> {out_path}")

    # Sanity check on the state fix + claim_type balance, using train only.
    train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
    if "provider_state" in train.columns:
        print("\nprovider_state value_counts (top 15) after normalization:")
        print(train["provider_state"].value_counts().head(15))

    ct_cols = [c for c in train.columns if c.startswith("claim_type_")]
    if ct_cols:
        print("\nclaim_type distribution (should sum to len(train) across the 3 columns):")
        print(train[ct_cols].sum())

    # Sanity check on the new zero-fill/sentinel-fill step: confirm no raw
    # NaN survives in any column outside the still-open exclusions
    # (HCPCS_CD's within-carrier gap, CARR_NUM/PRVDR_NUM's deliberate
    # 2-of-3 NaN, and date columns).
    still_open = {"HCPCS_CD", "CARR_NUM", "PRVDR_NUM"} | {
        "CLM_FROM_DT", "CLM_THRU_DT", "NCH_WKLY_PROC_DT"
    }
    remaining_nulls = train.drop(columns=[c for c in still_open if c in train.columns]).isna().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls):
        print("\nWARNING: unexpected remaining NaNs outside the known-open exclusions:")
        print(remaining_nulls)
    else:
        print("\nZero-fill/sentinel-fill check: no unexpected NaNs remain.")


if __name__ == "__main__":
    main()
