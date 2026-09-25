"""
Phase 1 Step 2/3 -- build is_denied, split before any target-leaking transform.

Run after load_data.py has produced data/processed/combined_claims_raw.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from denial_reasons import (
    apply_payment_consequence,
    audit_risk_factors_by_claim_type,
    calibration_report,
    sample_denials,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# GUARDRAIL (added 2026-09-24): a large claim-type x HCPCS cell with EXACTLY
# zero denials is reported every build. This alone would have exposed the
# PRVDR_NUM bug in Phase 1: 34 of 37 carrier codes sat at 0.000.
ZERO_CELL_MIN_CLAIMS = 1_000

# If the real-data denial rate lands outside 10-15% on first run, adjust this
# and rerun -- don't hand-edit REASON_CATALOG base_prob values individually
# unless the calibration_report shows one specific factor is over/under-firing.
CALIBRATION_SCALE = 1.4


def write_label_audit(df: pd.DataFrame, result: pd.DataFrame) -> None:
    """GUARDRAILS (added 2026-09-24) -- the label is checked per claim type,
    not just overall. (1) Every risk factor's firing rate in every claim type
    against RISK_FACTOR_EXPECTATION -- raises on any violation. (2) Denial
    rate per claim type. (3) Large claim-type x HCPCS cells with zero
    denials, which must each be explainable. Written to
    reports/label_audit.txt so the numbers are reviewable, not just printed."""
    source = df["_source_file"]
    rates = audit_risk_factors_by_claim_type(result, source)
    by_type = result["is_denied"].groupby(source.to_numpy()).agg(["size", "mean"])
    code = df["HCPCS_CD"].astype("object").where(df["HCPCS_CD"].notna(), "<missing>")
    cells = result["is_denied"].groupby([source.to_numpy(), code.to_numpy()]).agg(["size", "sum"])
    zero = cells[(cells["sum"] == 0) & (cells["size"] >= ZERO_CELL_MIN_CLAIMS)].sort_values("size", ascending=False)
    lines = [
        "Label audit -- written by build_target_and_split.py on every build",
        "=" * 66,
        f"Overall is_denied rate: {result['is_denied'].mean():.2%}",
        "",
        "Denial rate by claim type:",
        by_type.to_string(),
        "",
        "Risk-factor firing rate by claim type (every cell checked against",
        "denial_reasons.RISK_FACTOR_EXPECTATION -- this build passed):",
        rates.round(4).to_string(),
        "",
        f"Claim-type x HCPCS cells with >= {ZERO_CELL_MIN_CLAIMS:,} claims and ZERO denials: {len(zero)}",
        (zero.to_string() if len(zero) else "(none)"),
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_audit.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    if len(zero):
        print(
            f"\nWARNING: {len(zero)} large zero-denial cell(s) above. Each must have a documented "
            "reason (TARGET_DEFINITION.md) -- a cell nobody can explain is how the PRVDR_NUM bug hid."
        )


def main() -> None:
    raw_path = PROCESSED_DIR / "combined_claims_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found -- run load_data.py first.")

    df = pd.read_parquet(raw_path)

    result = sample_denials(df, calibration_scale=CALIBRATION_SCALE)
    calibration_report(result)
    write_label_audit(df, result)

    df = apply_payment_consequence(df, result)
    df = df.join(result)

    rate = df["is_denied"].mean()
    if not (0.05 <= rate <= 0.20):
        print(
            "\nWARNING: denial rate is outside the ~5-20% range typical for "
            "real payer data (Phase 1 Step 2 pitfall). Adjust CALIBRATION_SCALE "
            "at the top of this file and rerun."
        )

    # Persist a slim side-file with the label-construction columns before
    # they're dropped below. Three consumers need this, and a single saved
    # file beats each one recomputing sample_denials() separately (which
    # risks drift from a different CALIBRATION_SCALE, a different seed, or a
    # stale pull -- see the pull-lag incident logged in TARGET_DEFINITION.md):
    #   1. Phase 4/5: denial_reason_carc_1/2 for the SHAP-narrative demo and
    #      the report's "top denial reasons" chart.
    #   2. Phase 1 EDA (run_eda.py): risk-factor co-occurrence and per-factor
    #      lift charts, which need risk_*/is_denied together -- unavailable
    #      once leakage_cols are dropped from train/val/test below.
    #   3. Any future cross-file check needing risk_*/p_denied_model/is_denied
    #      joined back against train/val/test.parquet -- _row_id (added
    #      2026-09-16 in load_data.py) is the safe join key for this, not
    #      CLM_ID (which repeats across claim lines) or a reconstructed
    #      composite key.
    eda_cols = ["_row_id", "BENE_ID", "CLM_ID"] + [
        c for c in df.columns if c.startswith("risk_")
    ] + [
        "p_denied_model",
        "is_denied",
        "denial_reason_carc_1",
        "denial_reason_carc_2",
    ]
    eda_cols = [c for c in eda_cols if c in df.columns]
    eda_path = PROCESSED_DIR / "labeled_claims_for_eda.parquet"
    df[eda_cols].to_parquet(eda_path, index=False)
    print(f"\nSaved label-construction columns for EDA/Phase 4-5 use -> {eda_path}")

    # Target leakage guard: drop everything that was used to construct or is a
    # direct consequence of the label. `risk_*` and `p_denied_model` ARE the
    # target by construction. CLM_PMT_AMT was overwritten as a CONSEQUENCE of
    # is_denied in apply_payment_consequence() above -- it must be dropped too,
    # not just the risk columns. (Phase 1 Common Pitfalls: target leakage.)
    leakage_cols = [c for c in df.columns if c.startswith("risk_")] + [
        "p_denied_model",
        "denial_reason_carc_1",
        "denial_reason_carc_2",
        "CLM_PMT_AMT",
    ]
    leakage_cols = [c for c in leakage_cols if c in df.columns]
    print(f"Dropping target-leakage columns before split: {leakage_cols}")

    # Time-based split if CLM_FROM_DT is usable, else fall back to stratified
    # random split (Phase 1 Step 3).
    date_col = "CLM_FROM_DT"
    if date_col in df.columns and pd.to_datetime(df[date_col], errors="coerce").notna().mean() > 0.9:
        df = df.sort_values(date_col)
        n = len(df)
        train_end = int(n * 0.64)  # 0.8 * 0.8 to match the two-stage 80/20 split's proportions
        val_end = int(n * 0.8)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        print(f"\nTime-based split on {date_col}")
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["is_denied"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.2, stratify=train_df["is_denied"], random_state=42
        )
        print("\nStratified random split (date column unusable for time split)")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.drop(columns=leakage_cols, errors="ignore")
        out_path = PROCESSED_DIR / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(
            f"{name}: {len(split_df):,} rows, "
            f"is_denied rate {split_df['is_denied'].mean():.1%} -> {out_path}"
        )


if __name__ == "__main__":
    main()
