"""
Phase 2 -- build the design matrix for the Chow test (and, eventually, the
Phase 2 baseline logistic regression). Operates on a COPY of
train_model.parquet only -- never mutates the shared file XGBoost also
reads from. See data/FEATURE_ENGINEERING.md Section 5 for the full
reasoning behind each encoding choice below.

Run after build_features.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TOP_N_HCPCS = 20
TOP_N_DGNS = 20


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
    for PRVDR_NUM, whose 8,460 unique values make one-hot infeasible.

    Real NaN is preserved, NOT filled: PRVDR_NUM is a confirmed 2-of-3
    shared covariate (absent for carrier), and per the Chow-test design its
    NaN for the absent claim type must stay NaN so the interaction-term
    construction (still to be written -- see build_chow_design_matrix's
    TODO) can correctly omit that claim type's term. Fabricating a
    frequency value there would repeat the exact 0-vs-missing mistake the
    2026-09-16 zero-fill correction was written to prevent, just for a
    different encoding scheme.
    """
    freq = series.value_counts(normalize=True, dropna=True)
    return series.map(freq)


def build_chow_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the 4 shared covariates that needed cardinality treatment
    before they could enter any regression at all. Does NOT yet build the
    claim-type interaction terms (the 2-of-3-covariate omit-third-
    interaction logic, or the numeric-claim-type-exclusive zero-fill) --
    those are the next piece of this script, written once this encoding
    step is confirmed correct.
    """
    df = df.copy()

    state_dummies = pd.get_dummies(
        df["provider_state"], prefix="state", drop_first=True, dtype=int
    )
    hcpcs_dummies = top_n_encode(df["HCPCS_CD"], TOP_N_HCPCS, prefix="hcpcs")
    dgns_dummies = top_n_encode(df["PRNCPAL_DGNS_CD"], TOP_N_DGNS, prefix="dgns")
    df = pd.concat([df, state_dummies, hcpcs_dummies, dgns_dummies], axis=1)

    df["prvdr_num_freq"] = frequency_encode(df["PRVDR_NUM"])

    return df


if __name__ == "__main__":
    train = pd.read_parquet(PROCESSED_DIR / "train_model.parquet")
    design = build_chow_design_matrix(train)

    print(f"train_model.parquet: {train.shape[1]} columns")
    print(f"design matrix:       {design.shape[1]} columns")

    # Sanity check: PRVDR_NUM is absent for carrier (confirmed 2026-09-15),
    # so prvdr_num_freq's NaN count should equal carrier's row count exactly
    # -- not 0 (would mean NaN got silently filled) and not some other
    # number (would mean the encoding touched rows it shouldn't have).
    n_carrier = (train["claim_type_carrier"] == 1).sum()
    n_nan = design["prvdr_num_freq"].isna().sum()
    status = "OK" if n_nan == n_carrier else "MISMATCH -- investigate before using this design matrix"
    print(f"\nprvdr_num_freq NaN count: {n_nan:,} (expected: carrier row count = {n_carrier:,}) [{status}]")

    print(f"\nhcpcs_dummies columns: {[c for c in design.columns if c.startswith('hcpcs_')][:5]} ... ({sum(c.startswith('hcpcs_') for c in design.columns)} total)")
    print(f"dgns_dummies columns:  {[c for c in design.columns if c.startswith('dgns_')][:5]} ... ({sum(c.startswith('dgns_') for c in design.columns)} total)")
