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


if __name__ == "__main__":
    main()
