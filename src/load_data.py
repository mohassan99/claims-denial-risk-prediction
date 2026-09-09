"""
Phase 1 Step 1/3 — load and wrangle the CMS Synthetic Claims PUF.

Download first (manual, one-time): the collection page is a JS app, so grab it
by hand rather than scripting the fetch --
https://data.cms.gov/collection/synthetic-medicare-enrollment-fee-for-service-claims-and-prescription-drug-event
Unzip into data/raw/ -- you should end up with beneficiary_2015.csv ...
beneficiary_2023.csv, carrier.csv, dme.csv, hha.csv, hospice.csv, inpatient.csv,
outpatient.csv, pde.csv, snf.csv (all pipe-delimited, per the CMS user guide).

This script concatenates the claim-type files you're modeling on into one
frame for target construction + feature engineering. Start with carrier +
outpatient (highest claim volume per the CMS user guide's Table 3-1: carrier
is 59% of claims, outpatient 30% -- covers ~89% of claim volume with 2 files)
and add inpatient/DME/SNF/hospice/hha once the pipeline works end to end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Code columns that must stay strings -- leading zeros matter (Phase 1 Common
# Pitfalls: silent dtype coercion on procedure codes with leading zeros).
CODE_DTYPE_COLS = [
    "BENE_ID",
    "HCPCS_CD",
    "PRNCPAL_DGNS_CD",
    "PRVDR_NUM",
    "ICD_DGNS_CD1",
    "ICD_DGNS_CD2",
]


def load_claim_file(filename: str) -> pd.DataFrame:
    """Load one pipe-delimited RIF claim file with code columns forced to str."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download + unzip the CMS Synthetic Claims PUF "
            f"into {RAW_DIR} first (see this file's module docstring)."
        )

    # Only pass dtype overrides for columns that actually exist in this file --
    # not every claim type has every code column.
    header = pd.read_csv(path, sep="|", nrows=0).columns
    dtype_map = {c: str for c in CODE_DTYPE_COLS if c in header}

    df = pd.read_csv(path, sep="|", dtype=dtype_map, low_memory=False)
    df["_source_file"] = filename
    return df


def load_and_concat(claim_files: tuple[str, ...] = ("carrier.csv", "outpatient.csv")) -> pd.DataFrame:
    """Load and concatenate the given claim files. Row counts at each step are
    printed -- keep this output, it's your data-lineage narrative for the
    Phase 5 report (Phase 1 Step 3 requirement)."""
    frames = []
    for fname in claim_files:
        df = load_claim_file(fname)
        print(f"{fname}: {len(df):,} rows, {df['BENE_ID'].nunique():,} unique beneficiaries")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nCombined: {len(combined):,} rows across {len(claim_files)} claim types")
    return combined


def check_missingness(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Missingness by column, sorted descending -- feeds the Phase 1 Step 4
    missingness heatmap and the data dictionary's missing-value decisions."""
    return (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    combined = load_and_concat()
    print("\nTop missingness (%):")
    print(check_missingness(combined))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "combined_claims_raw.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}")
