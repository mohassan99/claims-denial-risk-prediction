"""
Phase 1 Step 1/3 — load and wrangle the CMS Synthetic Claims PUF.

Download first (manual, one-time): the collection page is a JS app, so grab it
by hand rather than scripting the fetch --
https://data.cms.gov/collection/synthetic-medicare-enrollment-fee-for-service-claims-and-prescription-drug-event
Unzip into data/raw/ -- you should end up with beneficiary_2015.csv ...
beneficiary_2023.csv, carrier.csv, dme.csv, hha.csv, hospice.csv, inpatient.csv,
outpatient.csv, pde.csv, snf.csv (all pipe-delimited, per the CMS user guide).

Default claim set: carrier + outpatient + dme. Carrier is 59% of claim volume
and outpatient 30% per the CMS user guide's Table 3-1 -- covers ~89% of claim
volume with those two alone. DME added 2026-09-12 specifically so
rule_missing_prior_auth (denial_rules.py) has claims to check: it flags HCPCS
codes starting with "E"/"K" (DME Level II codes), which carrier/outpatient
claims essentially never use -- without DME claims in the mix that rule fires
at 0% by construction, not because of a bug. Add inpatient/SNF/hospice/hha
once the pipeline works end to end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Code columns that must stay strings -- leading zeros matter (Phase 1 Common
# Pitfalls: silent dtype coercion on procedure codes with leading zeros).
# Kept explicit for columns that don't cleanly match the _CD/_NUM suffix
# pattern below (e.g. ICD_DGNS_CD1/2 end in a digit, not "_CD").
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

    header = pd.read_csv(path, sep="|", nrows=0).columns

    # Force every CMS "code"/identifier-number column to string, not just the
    # explicit CODE_DTYPE_COLS list above. CMS RIF _CD/_NUM fields are always
    # categorical identifiers, never real numeric quantities, and several
    # (e.g. PRVDR_STATE_CD, an SSA state code) can carry leading zeros that
    # silently vanish under pandas' default int inference.
    #
    # Why this matters beyond a single file: pandas infers dtype per-file
    # independently. If one claim file's column happens to parse cleanly as
    # int64 and another's doesn't (nulls, a stray non-numeric value, etc.),
    # concatenating the two produces a mixed int/str "object" column that
    # pyarrow can't write to parquet at all -- this is exactly what happened
    # 2026-09-11 concatenating carrier.csv + outpatient.csv on
    # PRVDR_STATE_CD (ArrowTypeError: "Expected bytes, got a 'int' object").
    # Forcing str at read time, per-file, before concat, prevents the two
    # files from ever disagreeing on this column's dtype in the first place.
    force_str_cols = {
        c for c in header if c in CODE_DTYPE_COLS or c.endswith("_CD") or c.endswith("_NUM")
    }
    dtype_map = {c: str for c in force_str_cols}

    df = pd.read_csv(path, sep="|", dtype=dtype_map, low_memory=False)
    df["_source_file"] = filename
    return df


def load_and_concat(
    claim_files: tuple[str, ...] = ("carrier.csv", "outpatient.csv", "dme.csv")
) -> pd.DataFrame:
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
