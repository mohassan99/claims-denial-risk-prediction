"""
Verify the provider_outlier mechanism suspected to explain
diagnose_separation.py's Step 3/4 finding: 6 HCPCS codes (94010, 96156,
99401, 99408, 99495, M1069) show an exact 0.000 denial rate within
claim_type=carrier across the FULL population (up to 45,788 claims per
code, confirmed 2026-09-23).

Four of denial_reasons.py's six risk factors are DETERMINISTICALLY
excluded for these codes by design, confirmed directly from
denial_rules.py:
  - dx_procedure_mismatch: "Returns False for any procedure code not in
    the mapping" -- these codes are explicitly documented as
    "deliberately left unmapped".
  - missing_prior_auth: only checks HCPCS codes starting with "E"/"K".
    None of these six do.
  - missing_hcpcs: only fires when HCPCS is ABSENT. These claims have
    HCPCS present (that's how they got selected here at all).
  - deprecated_code: only the 10 specific 2010-deprecated consult codes.
    Different codes entirely.

That leaves only duplicate_claim (independently documented as extremely
rare dataset-wide, plausibly zero by chance in any 45K-row slice) and
provider_outlier -- the one factor that SHOULD apply regardless of HCPCS
code, since it's a property of the billing provider, not the procedure.

ORIGINAL HYPOTHESIS (revised below after a real, surprising finding):
rule_provider_outlier computes its 95th-percentile claim-volume/payment
cutoff GLOBALLY across every provider in the dataset -- specialty_col
defaults to None everywhere in the pipeline. The original hypothesis was
that providers billing these routine/low-intensity codes are
systematically LOWER-volume than whatever sets the dataset-wide top-5%
bar.

ACTUAL FINDING (2026-09-23, first run of this script): each of the 6
codes has EXACTLY 1 distinct PRVDR_NUM value across tens of thousands of
claims (96156: 45,788 claims, 1 provider) -- and that one "provider" still
shows 0% volume-outlier status. A single real provider billing 45,788
claims of one code alone would trivially clear a 292-claim cutoff (the
actual 95th-percentile value found on this data) -- so "low-volume real
provider" cannot be the explanation here. The far more likely mechanism,
matching a pattern already caught twice elsewhere in this project (the
NPI presence flags, the missing_hcpcs discovery): PRVDR_NUM is plausibly
NaN (missing) for these specific claims, and pandas' groupby(...).size()
DROPS NaN keys by default (dropna=True) -- meaning these claims were never
even a CANDIDATE for the volume-outlier computation at all, not evaluated
and found low-volume. This script now prints the raw provider value(s)
directly (type, null-ness, and that value's own entry in claim_counts) to
confirm or refute this directly rather than continuing to infer it.

WHAT THIS SCRIPT CAN AND CAN'T CONFIRM EXACTLY:
  - CAN replicate the CLAIM-VOLUME half of the outlier rule exactly:
    claim_count per provider needs no payment column and is computed
    identically here as it would be anywhere else.
  - CANNOT replicate the PAYMENT half exactly: rule_provider_outlier's
    actual payment_col (CLM_PMT_AMT) isn't in train_model.parquet at all
    -- it was deliberately dropped from the modeling feature set (see
    decisions-and-learnings.md's Leakage fix section), and even if a
    stand-in payment column were used, apply_payment_consequence() means
    any SURVIVING payment-like field reflects POST-denial-zeroing values,
    not the native pre-label values the rule actually ran against during
    target construction.
  - Also computed here on train_model.parquet (the ~1.15M-row TRAIN
    split only), not the full pre-split dataset denial_reasons.py likely
    ran against originally -- percentile cutoffs and provider volume
    counts will differ somewhat from the original computation, though a
    low-volume provider in the full population should also read as
    low-volume within a ~68%-of-full train subset. (This caveat is now
    secondary to the NaN-grouping question above, which doesn't depend on
    split membership at all.)

Run with (from the repo root, inside the venv):
    python src/verify_provider_outlier_mechanism.py
"""

from __future__ import annotations

import pandas as pd

from fit_chow_test import LABEL_COL, TRAIN_PARQUET_PATH

PERCENTILE = 0.95  # matches rule_provider_outlier's default in denial_rules.py

# The 6 confirmed structural-zero codes (claim_type=carrier), from
# diagnose_separation.py's Step 4, run against the full ~1.15M-row
# population on 2026-09-23.
SUSPECT_CODES = ["94010", "96156", "99401", "99408", "99495", "M1069"]


def main() -> None:
    print("Reading PRVDR_NUM, HCPCS_CD, claim_type dummies, is_denied (full population)...")
    df = pd.read_parquet(
        TRAIN_PARQUET_PATH,
        columns=[
            "PRVDR_NUM", "HCPCS_CD", LABEL_COL,
            "claim_type_carrier", "claim_type_outpatient", "claim_type_dme",
        ],
    )
    print(f"  {len(df):,} rows\n")
    n_null_prvdr = int(df["PRVDR_NUM"].isna().sum())
    print(f"  PRVDR_NUM null count, whole population: {n_null_prvdr:,} ({n_null_prvdr / len(df):.4%})\n")

    # --- Replicate rule_provider_outlier's CLAIM-VOLUME half exactly ---
    # (payment half not replicated -- see module docstring.) NOTE:
    # groupby()'s default dropna=True means any NaN PRVDR_NUM rows are
    # silently excluded from claim_counts entirely -- deliberately NOT
    # overridden here, since that's the exact behavior rule_provider_outlier
    # itself has (it doesn't pass dropna=False either), and reproducing
    # that faithfully is the point of this check.
    claim_counts = df.groupby("PRVDR_NUM").size()
    vol_cut = claim_counts.quantile(PERCENTILE)
    outlier_providers_by_volume = set(claim_counts[claim_counts >= vol_cut].index)
    print(f"Volume cutoff (95th percentile of claim_count per provider): {vol_cut:.1f}")
    print(
        f"Providers flagged as volume outliers: {len(outlier_providers_by_volume):,} "
        f"of {claim_counts.size:,} total providers (NaN excluded from this count by groupby's default)\n"
    )

    overall_outlier_share = df["PRVDR_NUM"].isin(outlier_providers_by_volume).mean()
    print(f"Overall population's claims-from-a-volume-outlier-provider share: {overall_outlier_share:.4%}\n")

    print("=" * 78)
    print("Per-suspect-code check: what share of THIS code's claims come from a volume-outlier provider?")
    print("=" * 78)
    for code in SUSPECT_CODES:
        mask = (df["HCPCS_CD"] == code) & (df["claim_type_carrier"] == 1)
        n = int(mask.sum())
        if n == 0:
            print(f"  {code}: 0 claims found under claim_type=carrier (check code/claim_type)")
            continue
        code_providers_raw = df.loc[mask, "PRVDR_NUM"].unique()
        code_providers = set(code_providers_raw)
        overlap = code_providers & outlier_providers_by_volume
        claims_from_outlier = df.loc[mask, "PRVDR_NUM"].isin(outlier_providers_by_volume).mean()
        print(
            f"  {code}: {n:,} claims, {len(code_providers):,} distinct providers, "
            f"{len(overlap)} of them ({len(overlap) / len(code_providers):.2%}) are volume outliers -- "
            f"{claims_from_outlier:.4%} of this code's claims come from a volume-outlier provider "
            f"(vs {overall_outlier_share:.4%} population-wide)"
        )
        # A single real provider billing 10,000s of claims of one code alone
        # should trivially clear a 292-claim cutoff -- if that's not
        # happening, print exactly what the raw value is (NaN would be
        # dropped by groupby's default dropna=True and so would never even
        # be a CANDIDATE for the outlier flag, a materially different and
        # more serious explanation than "evaluated and found low-volume").
        if len(code_providers) <= 3:
            for val in code_providers_raw:
                is_null = bool(pd.isna(val))
                own_count = claim_counts.get(val)
                print(
                    f"      raw PRVDR_NUM value: {val!r} (type={type(val).__name__}, is_null={is_null}) -- "
                    f"its own entry in claim_counts: {own_count!r} "
                    f"({'excluded entirely by groupby dropna=True' if own_count is None else 'present normally'})"
                )

    print(
        "\nIf the per-code detail above shows is_null=True and 'excluded entirely', that confirms "
        "these claims' PRVDR_NUM is missing and was never a candidate for the volume-outlier flag "
        "at all -- a structural gap in rule_provider_outlier (same 'NaN silently dropped, not "
        "evaluated' class of issue already caught in the NPI flags and the missing_hcpcs "
        "discovery), not a specialty/volume-mix effect. If instead a real non-null provider ID "
        "shows a genuinely low claim_count, the original low-volume-provider hypothesis holds."
    )


if __name__ == "__main__":
    main()
