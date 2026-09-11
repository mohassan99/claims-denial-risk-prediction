"""
Phase 1 Step 4 — EDA with saved visualizations.

Produces the 5 required figures into reports/figures/. Run after
build_target_and_split.py. Uses train.parquet only (never look at val/test
during EDA -- avoids the temptation to eyeball-tune against data you'll
evaluate on later).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- safe for scripted / CI runs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORTS_DIR = FIG_DIR.parent

sns.set_theme(style="whitegrid")


def denial_rate_by_procedure(df: pd.DataFrame, top_n: int = 15) -> None:
    """Headline business-insight chart -- denial rate by procedure/HCPCS."""
    grp = (
        df.groupby("HCPCS_CD")["is_denied"]
        .agg(["mean", "count"])
        .query("count >= 5")  # drop noisy low-volume codes
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=grp["mean"], y=grp.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Denial rate")
    ax.set_ylabel("HCPCS code")
    ax.set_title(f"Denial rate by procedure code (top {top_n} by rate, min 5 claims)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "denial_rate_by_procedure.png", dpi=150)
    plt.close(fig)


def denial_rate_by_provider_specialty(df: pd.DataFrame, specialty_col: str = "PRVDR_NUM") -> None:
    """Denial rate by provider (specialty proxy if a real specialty field
    isn't available in your extract -- swap specialty_col once you confirm
    the real column name from your download)."""
    grp = (
        df.groupby(specialty_col)["is_denied"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=grp["mean"], y=grp.index.astype(str), ax=ax, color="#55A868")
    ax.set_xlabel("Denial rate")
    ax.set_ylabel(specialty_col)
    ax.set_title("Denial rate by provider (top 15, min 5 claims)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "denial_rate_by_provider.png", dpi=150)
    plt.close(fig)


def denial_rate_over_time(df: pd.DataFrame, date_col: str = "CLM_FROM_DT") -> None:
    """Flags seasonality / policy-change effects."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    monthly = df.assign(_month=dates.dt.to_period("M"))
    grp = monthly.groupby("_month")["is_denied"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    grp.plot(ax=ax, marker="o", color="#C44E52")
    ax.set_xlabel("Month")
    ax.set_ylabel("Denial rate")
    ax.set_title("Denial rate over time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "denial_rate_over_time.png", dpi=150)
    plt.close(fig)


def missingness_heatmap(df: pd.DataFrame, max_cols: int = 30) -> None:
    cols = df.isna().mean().sort_values(ascending=False).head(max_cols).index
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].isna(), cbar=False, yticklabels=False, ax=ax, cmap="rocket_r")
    ax.set_title(f"Missingness pattern (top {max_cols} columns by missing %)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missingness_heatmap.png", dpi=150)
    plt.close(fig)


def correlation_with_target(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if "is_denied" not in numeric.columns:
        numeric = numeric.join(df["is_denied"])
    corr = numeric.corr(numeric_only=True)[["is_denied"]].drop(index="is_denied")
    corr = corr.sort_values("is_denied", ascending=False)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(corr))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlation with is_denied")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_with_target.png", dpi=150)
    plt.close(fig)


def validate_native_payment_vs_risk_factors() -> None:
    """Phase 1 Step 4 addendum -- see data/TARGET_DEFINITION.md Addendum #5.

    Answers the design question "why not just threshold/target native
    CLM_PMT_AMT instead of engineering is_denied?" empirically, against this
    project's own data, rather than by reasoning alone.

    Runs on data/processed/combined_claims_raw.parquet -- NOT train.parquet.
    train.parquet has both CLM_PMT_AMT and the risk_* columns dropped as
    target leakage by build_target_and_split.py (and even if it didn't,
    CLM_PMT_AMT there would already be the POST-apply_payment_consequence()
    value, not the native one). This function needs CLM_PMT_AMT before
    denial_reasons.py has ever touched it, and the risk-factor flags computed
    independently of is_denied/p_denied -- both of which only exist together,
    pre-consequence, in the raw combined file.

    Cross-tabulates native near-zero payment against each engineered risk
    factor (duplicate_claim, missing_prior_auth, dx_procedure_mismatch,
    provider_outlier). If native zero-pay claims already correlate strongly
    with these flags, that's a genuine, reportable finding -- Synthea's cost
    engine would incidentally be reproducing something denial-shaped. If not,
    it's direct empirical evidence (not just reasoning) that CLM_PMT_AMT
    can't stand in for is_denied, either as a detection heuristic or as the
    modeling target itself.

    Writes a text summary to reports/native_payment_validation.txt and prints
    it. Record the actual numbers back into TARGET_DEFINITION.md Addendum #5
    once this has been run against real downloaded data -- the action item
    there is still open until that happens.
    """
    raw_path = PROCESSED_DIR / "combined_claims_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found -- run load_data.py first.")

    # Imported here, not at module top, so the 5 required EDA figures don't
    # carry a hard dependency on denial_reasons/denial_rules for anyone just
    # re-running the standard plots.
    from denial_reasons import compute_risk_factors

    df = pd.read_parquet(raw_path)

    if "CLM_PMT_AMT" not in df.columns:
        raise KeyError(
            "CLM_PMT_AMT not found in combined_claims_raw.parquet -- can't run "
            "the native-payment validation check without it."
        )

    native_paid = df["CLM_PMT_AMT"].fillna(0)
    # Same $1 threshold as denial_rules.ZERO_PAY_THRESHOLD, kept independent
    # here so this check doesn't silently break if that constant changes.
    near_zero = native_paid < 1.00

    risk_factors = compute_risk_factors(df)

    lines = [
        "Native CLM_PMT_AMT vs. engineered risk-factor flags",
        "=" * 55,
        "Pre-registered question: does Synthea's native payment amount already",
        "correlate with these risk factors, before any is_denied engineering?",
        "See data/TARGET_DEFINITION.md Addendum #5 for the full reasoning.",
        "",
        f"Overall native near-zero-payment rate (< $1.00): {near_zero.mean():.1%}",
        "",
        f"{'risk factor':<22}{'active rate':>13}{'near-zero | active':>22}{'near-zero | inactive':>24}",
    ]
    for factor in risk_factors.columns:
        active = risk_factors[factor]
        rate_active = active.mean()
        pay_given_active = near_zero[active].mean() if active.any() else float("nan")
        pay_given_inactive = near_zero[~active].mean() if (~active).any() else float("nan")
        lines.append(
            f"{factor:<22}{rate_active:>12.1%} {pay_given_active:>21.1%} {pay_given_inactive:>23.1%}"
        )

    lines += [
        "",
        "Interpretation guide:",
        "- If 'near-zero | active' is NOT meaningfully higher than 'near-zero |",
        "  inactive' for any factor, native payment shows no relationship to",
        "  that risk factor -- supports engineering is_denied independently.",
        "- If one or more factors DO show a gap, that's a genuine finding worth",
        "  reporting, not a reason to switch the target -- see Addendum #5 for",
        "  why native payment still conflates denial with deductible/",
        "  coinsurance absorption and bundling even if a correlation exists.",
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "native_payment_validation.txt"
    out_path.write_text(summary)
    print(f"\nSaved validation summary to {out_path}")


def main() -> None:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found -- run build_target_and_split.py first.")

    df = pd.read_parquet(train_path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    denial_rate_by_procedure(df)
    denial_rate_by_provider_specialty(df)
    denial_rate_over_time(df)
    missingness_heatmap(df)
    correlation_with_target(df)

    saved = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f"Saved {len(saved)} figures to {FIG_DIR}:")
    for name in saved:
        print(f"  - {name}")

    # Not one of the 5 required figures -- a text-based validation check
    # (Phase 1 addendum, data/TARGET_DEFINITION.md Addendum #5). Runs against
    # combined_claims_raw.parquet independently of the train.parquet loaded
    # above, since it needs pre-leakage-drop, pre-consequence columns.
    validate_native_payment_vs_risk_factors()


if __name__ == "__main__":
    main()
