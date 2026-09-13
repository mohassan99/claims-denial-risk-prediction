"""
Phase 1 completion — apply the feature-engineering decisions documented in
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
# 100%-null columns, and zero-variance columns confirmed 2026-09-13 via
# .value_counts() (FEATURE_ENGINEERING.md Sections 1-2).
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


if __name__ == "__main__":
    main()
