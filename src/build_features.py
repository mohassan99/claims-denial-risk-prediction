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
# (FEATURE_ENGINEERING.md Sections 1-3, 6).
# ---------------------------------------------------------------------------
DROP_COLUMNS = [
    # Raw NPI values -- presence already captured above.
    "RFR_PHYSN_NPI", "PRF_PHYSN_NPI", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
    "RNDRNG_PHYSN_NPI", "PRVDR_NPI",
    "PRVDR_ZIP",
    # Always-100%-null across the whole dataset (original 10, found 2026-09-13).
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
    # Added 2026-09-16: LINE_CMS_TYPE_SRVC_CD. Confirmed constant ("1") on
    # BOTH carrier LINE_NUM=1 and carrier's other lines, and confirmed as an
    # officially-documented fixed value for Carrier (CMS User Guide, May
    # 2023, Table 6-4: "Line HCFA Type Service Code" = 1). NOT independently
    # verified for DME, where this field is also populated (2-of-3 pattern,
    # per the shared-feature audit) but has no corresponding entry in the
    # User Guide's DME table (6-6) -- if DME later shows real variance here,
    # this drop decision needs revisiting for that claim type specifically.
    "LINE_CMS_TYPE_SRVC_CD",
    # Added 2026-09-16: three fields confirmed EXACT duplicates of another
    # kept field, verified via direct equality checks against train.parquet
    # (100% exact match, diff == 0 for every row, no tolerance needed -- not
    # just highly correlated). All three trace to the same root cause: this
    # dataset's Synthea generation never models a provider write-off/
    # discount (LINE_SBMTD_CHRG_AMT == LINE_ALOWD_CHRG_AMT, confirmed on
    # 602,755 interior carrier lines, std == 0.0) or non-assignment billing
    # (LINE_NCH_PMT_AMT == LINE_PRVDR_PMT_AMT and the claim-level rollup
    # NCH_CLM_PRVDR_PMT_AMT == NCH_CARR_CLM_ALOWD_AMT, both 100% exact across
    # all 718,074 carrier rows) -- i.e. every carrier claim behaves as if
    # fully assigned and billed at exactly the allowed rate, with no
    # exceptions in this data. This is a disclosed synthetic-data limitation
    # (see FEATURE_ENGINEERING.md Section 2's new subsection), not a
    # cleaning artifact -- kept field chosen as whichever one participates
    # in the real payer payment-reconciliation identity (allowed over
    # billed; the primary NCH payment field over its provider-payment
    # mirror). Dropping these is not optional the way a merely-correlated
    # pair might be: exact equality makes the design matrix singular for
    # any linear model.
    "LINE_SBMTD_CHRG_AMT", "LINE_PRVDR_PMT_AMT", "NCH_CLM_PRVDR_PMT_AMT",
    # Added 2026-09-21: 20 more always-100%-null columns, found via the
    # pre-sentinel-fill full presence audit (data/full_claim_type_presence_
    # audit_PREFILL.csv) -- the original 2026-09-13 always-null sweep (the
    # 10 columns above) was simply incomplete, not a data change. Confirmed
    # directly against train_model.parquet before adding here: without this
    # fix, each of these 20 sits as a CONSTANT string ("NOT_APPLICABLE")
    # across every one of ~1.15M rows, since fill_claim_type_exclusive_
    # fields() correctly (but silently) sentinel-fills a column that's
    # 100%-null in ALL THREE claim types just as it would one that's null in
    # only one or two -- zero variance, zero information, dead weight in the
    # design matrix either way. Dropping these does NOT change the Chow-test
    # 2-of-3 (50) or 3-of-3 (21) counts from the corrected pre-fill audit --
    # this is strictly a separate 0-of-3 bucket, confirmed by construction
    # (n_present == 0 for all 20, checked before this list was written).
    "CARR_LINE_MTUS_CD", "CARR_LINE_RX_NUM", "CLM_CLNCL_TRIL_NUM", "FI_NUM",
    "HCPCS_1ST_MDFR_CD", "HCPCS_2ND_MDFR_CD", "HCPCS_3RD_MDFR_CD", "HCPCS_4TH_MDFR_CD",
    "LINE_NDC_CD", "LINE_PMT_80_100_CD",
    "REV_CNTR_1ST_ANSI_CD", "REV_CNTR_2ND_ANSI_CD", "REV_CNTR_3RD_ANSI_CD",
    "REV_CNTR_4TH_ANSI_CD", "REV_CNTR_APC_HIPPS_CD", "REV_CNTR_DSCNT_IND_CD",
    "REV_CNTR_IDE_NDC_UPC_NUM", "REV_CNTR_NDC_QTY_QLFR_CD", "REV_CNTR_OTAF_PMT_CD",
    "REV_CNTR_PACKG_IND_CD",
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
    """Sentinel-fill CATEGORICAL (string/object) columns whose nullness is
    structurally determined by claim_type. Does NOT touch numeric columns --
    see the correction below.

    CORRECTED 2026-09-16 (originally also zero-filled numeric columns; that
    was wrong and has been removed). 0 is a legitimate real value for
    numeric CMS dollar/count fields (a genuine $0 charge, a genuine 0
    count) -- zero-filling the raw column conflates "genuinely $0" with
    "field doesn't apply to this claim type," the same distinct-value-vs-
    missingness conflation already caught for the NPI presence flags and
    PRNCPAL_DGNS_VRSN_CD (Section 2). It also actively discards information:
    XGBoost handles real NaN natively and can treat "this field is absent"
    as its own signal, which a zero-fill silently removes. General
    principle: 0 is a value, NaN is the absence of one -- never use the
    former to represent the latter, for any field type.

    Numeric claim-type-exclusive columns are therefore left with their real
    NaN in train_model.parquet. The zero-fill-for-interaction trick this was
    originally trying to implement (value * claim_type_dummy, so the
    interaction term cleanly zeroes out where a field doesn't apply) is
    still correct -- but it belongs ONLY inside the script that actually
    builds the Chow-test/logistic-regression design matrix
    (src/build_chow_design_matrix.py), applied there on a COPY, scoped to
    that one multiplication -- never baked upstream into this shared file,
    which XGBoost also reads from natively.

    String/object columns don't have this problem -- "NOT_APPLICABLE" is an
    explicit new category, not an overloaded existing value -- so those are
    still sentinel-filled here, derived empirically at runtime (same
    null-rate-by-claim_type technique as the shared-feature audit) rather
    than a hardcoded column list.

    NOTE (2026-09-21): this function makes no distinction between a column
    that's 100%-null in ONE or TWO claim types (a genuine claim-type-
    exclusive/2-of-3 field, correctly sentinel-filled for the claim types it
    doesn't apply to) and one that's 100%-null in ALL THREE (i.e. globally
    always-null) -- the latter also passes through this loop and gets
    sentinel-filled into a dataset-wide CONSTANT, which is harmless but
    wasteful (zero variance either way). 20 such columns were found this way
    via the pre-sentinel-fill presence audit and moved to DROP_COLUMNS above
    instead, alongside the original 10 always-null columns from 2026-09-13 --
    this function's fill logic itself did not need to change, since
    filtering globally-always-null columns out via DROP_COLUMNS before this
    function ever sees them is the correct fix, not adding a special case
    here.

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

        # Numeric columns: leave real NaN in place -- see docstring
        # correction above. The zero-fill trick belongs in Phase 2's
        # design-matrix code, not here.
        if pd.api.types.is_numeric_dtype(df[col]):
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

    # Sentinel-fill (categorical only, see docstring) every remaining
    # claim-type-exclusive field. Must run AFTER claim_type_* dummies exist
    # and AFTER DROP_COLUMNS has removed anything that shouldn't reach this
    # pass. Numeric claim-type-exclusive columns intentionally keep real NaN
    # -- see fill_claim_type_exclusive_fields()'s docstring.
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

    # Sanity check on the sentinel-fill step (2026-09-16, corrected):
    # categorical columns should have NO remaining NaN after the fill -- any
    # that do indicate a bug in fill_claim_type_exclusive_fields, since that
    # function is now the only thing responsible for clearing NaN from
    # string/object columns. Numeric columns are EXPECTED to retain real NaN
    # now (see that function's docstring) -- reported separately below as
    # informational, not a warning.
    categorical_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    remaining_cat_nulls = train[categorical_cols].isna().sum()
    remaining_cat_nulls = remaining_cat_nulls[remaining_cat_nulls > 0]
    if len(remaining_cat_nulls):
        print("\nWARNING: categorical columns still have NaN after sentinel-fill (unexpected):")
        print(remaining_cat_nulls)
    else:
        print("\nSentinel-fill check: no unexpected NaNs remain in categorical columns.")

    # Sanity check on the 20 newly-dropped always-null columns (2026-09-21):
    # confirm none of them survived into the final output. Any that did
    # would mean DROP_COLUMNS's names don't match train_model.parquet's
    # actual column names -- a typo, not a logic error.
    newly_dropped = [
        "CARR_LINE_MTUS_CD", "CARR_LINE_RX_NUM", "CLM_CLNCL_TRIL_NUM", "FI_NUM",
        "HCPCS_1ST_MDFR_CD", "HCPCS_2ND_MDFR_CD", "HCPCS_3RD_MDFR_CD", "HCPCS_4TH_MDFR_CD",
        "LINE_NDC_CD", "LINE_PMT_80_100_CD",
        "REV_CNTR_1ST_ANSI_CD", "REV_CNTR_2ND_ANSI_CD", "REV_CNTR_3RD_ANSI_CD",
        "REV_CNTR_4TH_ANSI_CD", "REV_CNTR_APC_HIPPS_CD", "REV_CNTR_DSCNT_IND_CD",
        "REV_CNTR_IDE_NDC_UPC_NUM", "REV_CNTR_NDC_QTY_QLFR_CD", "REV_CNTR_OTAF_PMT_CD",
        "REV_CNTR_PACKG_IND_CD",
    ]
    survived = [c for c in newly_dropped if c in train.columns]
    if survived:
        print(f"\nWARNING: expected these 20 newly-dropped columns to be gone, but {len(survived)} survived: {survived}")
    else:
        print(f"\nConfirmed: all 20 newly-added always-null columns (2026-09-21) successfully dropped.")

    numeric_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    numeric_nulls = train[numeric_cols].isna().sum()
    numeric_nulls = numeric_nulls[numeric_nulls > 0]
    print(
        f"\n{len(numeric_nulls)} numeric columns retain real (intentional) NaN -- "
        "expected for claim-type-exclusive fields; the zero-fill needed for "
        "Chow-test/logistic-regression interaction terms happens later, in "
        "Phase 2's design-matrix code, scoped to a copy -- not here."
    )


if __name__ == "__main__":
    main()
