"""
Phase 1 Step 4 — EDA with saved visualizations.

Produces the 5 required figures into reports/figures/. Run after
build_target_and_split.py. Uses train.parquet only (never look at val/test
during EDA -- avoids the temptation to eyeball-tune against data you'll
evaluate on later).

Also produces supplementary figures beyond the 5 required ones (added
2026-09-13, per design review): a class-imbalance chart, categorical
cardinality/concentration, numeric feature distributions, and two figures
that need the pre-leakage-drop risk_*/is_denied columns from
labeled_claims_for_eda.parquet (risk-factor co-occurrence and per-factor
lift) -- these exist to answer "what does Phase 2 need to know" rather than
just satisfying the Phase 1 checklist.
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


# ---------------------------------------------------------------------------
# The 5 required Phase 1 Step 4 figures
# ---------------------------------------------------------------------------
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
    """NOTE: this is numeric-only, and most of this dataset's predictive
    signal is categorical (HCPCS_CD, PRVDR_NUM, diagnosis codes) -- expect
    this chart to come back sparse or trivial. It's kept because it's a
    required Phase 1 figure, but categorical_cardinality() and
    denial_rate_by_procedure()/denial_rate_by_provider_specialty() above are
    doing the real categorical-signal work this chart can't."""
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


# ---------------------------------------------------------------------------
# Supplementary figures (added 2026-09-13) -- not required by the Phase 1
# checklist, but answer "what does Phase 2 need to know" rather than just
# "what does the checklist require."
# ---------------------------------------------------------------------------
def class_imbalance_chart(df: pd.DataFrame) -> None:
    """Explicit class-imbalance figure -- puts the actual denial rate in
    front of a reviewer directly, and is the visual justification for the
    Phase 2 PR-AUC-over-accuracy metric choice (Build Guide Step 12)."""
    counts = df["is_denied"].value_counts().sort_index()
    labels = ["Not denied", "Denied"]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.reindex([0, 1], fill_value=0).values, color=["#4C72B0", "#C44E52"])
    for bar, count in zip(bars, counts.reindex([0, 1], fill_value=0).values):
        pct = count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}\n({pct:.1%})", ha="center", va="bottom",
        )
    ax.set_ylabel("Claim count")
    ax.set_title(f"Class balance: is_denied ({total:,} total claims)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_imbalance.png", dpi=150)
    plt.close(fig)


def categorical_cardinality(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("HCPCS_CD", "PRVDR_NUM", "PRNCPAL_DGNS_CD"),
    top_n: int = 20,
) -> None:
    """Cardinality + concentration for the categorical columns Phase 2 will
    need to encode. Answers the actual encoding-strategy question: is this a
    handful of frequent categories (one-hot + 'other' bucket is fine) or a
    long tail (frequency/target encoding needed instead)? Nothing in the 5
    required figures tells you this."""
    present = [c for c in cols if c in df.columns]
    if not present:
        print("\nNone of the expected categorical columns found -- skipping categorical_cardinality.")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))
    if len(present) == 1:
        axes = [axes]

    lines = ["Categorical cardinality and concentration", "=" * 45]
    for ax, col in zip(axes, present):
        vc = df[col].value_counts()
        n_unique = vc.shape[0]
        top_share = vc.head(top_n).sum() / vc.sum()
        lines.append(f"{col}: {n_unique:,} unique values; top {top_n} cover {top_share:.1%} of claims")

        vc.head(top_n).plot(kind="bar", ax=ax, color="#55A868")
        ax.set_title(f"{col}\n{n_unique:,} unique, top {top_n} = {top_share:.1%}")
        ax.set_ylabel("Claim count")
        ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "categorical_cardinality.png", dpi=150)
    plt.close(fig)

    summary = "\n".join(lines)
    print("\n" + summary)
    (REPORTS_DIR / "categorical_cardinality.txt").write_text(summary)


def numeric_feature_distributions(df: pd.DataFrame, max_cols: int = 8) -> None:
    """Distributions of the numeric features that actually survive the
    leakage drop and will reach Phase 2 -- catches skew/outliers before
    fitting the Step 12 baseline logistic regression, which is sensitive to
    this in a way XGBoost isn't."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "is_denied"]
    if not numeric_cols:
        print("\nNo numeric feature columns found (besides is_denied) -- skipping numeric_feature_distributions.")
        return
    if len(numeric_cols) > max_cols:
        print(
            f"\n{len(numeric_cols)} numeric columns found -- plotting the first "
            f"{max_cols} for readability. Adjust max_cols if you need more."
        )
        numeric_cols = numeric_cols[:max_cols]

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        df[col].dropna().plot(kind="box", ax=ax)
        ax.set_title(col, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "numeric_feature_distributions.png", dpi=150)
    plt.close(fig)


def risk_factor_cooccurrence() -> None:
    """Checks the noisy-OR independence assumption (TARGET_DEFINITION.md
    Addendum #1) empirically: are the 5 risk factors actually independent on
    this data, or correlated (e.g. provider_outlier with dx_procedure_mismatch,
    as the doc speculates)? Reads labeled_claims_for_eda.parquet -- the risk_*
    columns aren't in train/val/test (dropped as leakage in
    build_target_and_split.py)."""
    eda_path = PROCESSED_DIR / "labeled_claims_for_eda.parquet"
    if not eda_path.exists():
        raise FileNotFoundError(f"{eda_path} not found -- run build_target_and_split.py first.")
    labeled = pd.read_parquet(eda_path)
    risk_cols = [c for c in labeled.columns if c.startswith("risk_")]
    if not risk_cols:
        print("\nNo risk_* columns found in labeled_claims_for_eda.parquet -- skipping.")
        return

    corr = labeled[risk_cols].astype(int).corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Risk-factor co-occurrence (checks noisy-OR independence assumption)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_factor_cooccurrence.png", dpi=150)
    plt.close(fig)


def risk_factor_lift() -> None:
    """Observed is_denied rate when each risk factor is active vs inactive --
    validates that noisy-OR actually produced graded lift as intended, and
    previews which factors are likely to show up prominently in Phase 2's
    SHAP output. Same data dependency as risk_factor_cooccurrence()."""
    eda_path = PROCESSED_DIR / "labeled_claims_for_eda.parquet"
    if not eda_path.exists():
        raise FileNotFoundError(f"{eda_path} not found -- run build_target_and_split.py first.")
    labeled = pd.read_parquet(eda_path)
    risk_cols = [c for c in labeled.columns if c.startswith("risk_")]
    if not risk_cols:
        print("\nNo risk_* columns found in labeled_claims_for_eda.parquet -- skipping.")
        return

    rows = []
    for col in risk_cols:
        active = labeled[col].astype(bool)
        active_rate = labeled.loc[active, "is_denied"].mean() if active.any() else float("nan")
        inactive_rate = labeled.loc[~active, "is_denied"].mean() if (~active).any() else float("nan")
        rows.append((col, active_rate, inactive_rate))

    lift_df = pd.DataFrame(rows, columns=["factor", "active_rate", "inactive_rate"]).set_index("factor")

    fig, ax = plt.subplots(figsize=(8, 5))
    lift_df.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"])
    ax.set_ylabel("is_denied rate")
    ax.set_title("Observed denial rate: factor active vs inactive")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_factor_lift.png", dpi=150)
    plt.close(fig)

    print("\nRisk factor lift (is_denied rate, active vs inactive):")
    print(lift_df)


def check_missingness(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Missingness by column, sorted descending -- feeds the Phase 1 Step 4
    missingness heatmap and the data dictionary's missing-value decisions."""
    return (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)


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
    factor. If native zero-pay claims already correlate strongly with these
    flags, that's a genuine, reportable finding -- Synthea's cost engine
    would incidentally be reproducing something denial-shaped. If not, it's
    direct empirical evidence (not just reasoning) that CLM_PMT_AMT can't
    stand in for is_denied, either as a detection heuristic or as the
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

    # The 5 required Phase 1 Step 4 figures.
    denial_rate_by_procedure(df)
    denial_rate_by_provider_specialty(df)
    denial_rate_over_time(df)
    missingness_heatmap(df)
    correlation_with_target(df)

    saved = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f"Saved {len(saved)} figures to {FIG_DIR} so far (5 required + any supplementary already run):")
    for name in saved:
        print(f"  - {name}")

    # Supplementary figures (2026-09-13) -- answer "what does Phase 2 need to
    # know", not just the Phase 1 checklist. First three run on train.parquet
    # directly; the last two need labeled_claims_for_eda.parquet.
    class_imbalance_chart(df)
    categorical_cardinality(df)
    numeric_feature_distributions(df)
    risk_factor_cooccurrence()
    risk_factor_lift()

    saved = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f"\nSaved {len(saved)} total figures to {FIG_DIR}:")
    for name in saved:
        print(f"  - {name}")

    # Not one of the 5 required figures -- a text-based validation check
    # (Phase 1 addendum, data/TARGET_DEFINITION.md Addendum #5). Runs against
    # combined_claims_raw.parquet independently of the train.parquet loaded
    # above, since it needs pre-leakage-drop, pre-consequence columns.
    validate_native_payment_vs_risk_factors()


if __name__ == "__main__":
    main()
