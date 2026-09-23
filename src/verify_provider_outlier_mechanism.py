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

HYPOTHESIS: rule_provider_outlier computes its 95th-percentile claim-
volume/payment cutoff GLOBALLY across every provider in the dataset --
specialty_col defaults to None in both the rule's own signature and
denial_reasons.py's compute_risk_factors(), and nothing in the pipeline
overrides it. If providers billing these specific routine/low-intensity
codes (health behavior intervention, SBIRT screening, transitional care
management, preventive counseling, spirometry, a quality-measure code)
are systematically lower-volume than whatever sets a dataset-wide top-5%
bar (plausibly high-throughput labs/DME suppliers/imaging), none of them
would ever cross it -- explaining the exact zero.

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
    ran against originally (is_denied needs to exist before a stratified
    train/val/test split can be done on it) -- percentile cutoffs and
    provider volume counts will differ somewhat from the original
    computation, though a low-volume provider in the full population
    should also read as low-volume within a ~68%-of-full train subset,
    so this should still be a reliable DIRECTIONAL test even if not a
    byte-for-byte replication.

If the claim-volume-only check alone already shows ~0% overlap for all 6
codes, that's sufficient to confirm the mechanism (duplicate_claim's
independent rarity covers the rest). If it doesn't, the payment half
would need checking against an earlier pre-feature-engineering file this
script doesn't have access to.

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

    # --- Replicate rule_provider_outlier's CLAIM-VOLUME half exactly ---
    # (payment half not replicated -- see module docstring.)
    claim_counts = df.groupby("PRVDR_NUM").size()
    vol_cut = claim_counts.quantile(PERCENTILE)
    outlier_providers_by_volume = set(claim_counts[claim_counts >= vol_cut].index)
    print(f"Volume cutoff (95th percentile of claim_count per provider): {vol_cut:.1f}")
    print(
        f"Providers flagged as volume outliers: {len(outlier_providers_by_volume):,} "
        f"of {claim_counts.size:,} total providers\n"
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
        code_providers = set(df.loc[mask, "PRVDR_NUM"].unique())
        overlap = code_providers & outlier_providers_by_volume
        claims_from_outlier = df.loc[mask, "PRVDR_NUM"].isin(outlier_providers_by_volume).mean()
        print(
            f"  {code}: {n:,} claims, {len(code_providers):,} distinct providers, "
            f"{len(overlap)} of them ({len(overlap) / len(code_providers):.2%}) are volume outliers -- "
            f"{claims_from_outlier:.4%} of this code's claims come from a volume-outlier provider "
            f"(vs {overall_outlier_share:.4%} population-wide)"
        )

    print(
        "\nIf every suspect code's 'claims from a volume-outlier provider' share is at or near "
        "0% (well below the population-wide share above), that confirms the claim-volume half "
        "of provider_outlier as (at least a major part of) the mechanism -- combined with "
        "duplicate_claim's independent dataset-wide rarity, this would fully account for the "
        "exact-zero finding without needing the payment half checked. If instead these codes' "
        "shares are comparable to or higher than the population rate, the volume half doesn't "
        "explain it and the payment half (or something else) would need investigating instead."
    )


if __name__ == "__main__":
    main()
