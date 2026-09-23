"""
Diagnose the identical-log-likelihood / LR=0 finding from fit_chow_test.py.

Both the restricted and unrestricted Logit fits converged to bit-identical
log-likelihoods at TWO very different sample sizes (23,039 rows at
--sample-frac 0.02, and 230,391 rows at --sample-frac 0.2 -- roughly 460
events/parameter at the larger size, comfortably above the standard
10-20-events-per-variable danger zone), both showing exp-overflow /
log-divide-by-zero / HessianInversionWarning. Identical results across an
order-of-magnitude difference in sample size rules out "not enough data"
(2026-09-23 finding) -- this points at a specific covariate (or small set
of them) driving quasi-complete or complete separation regardless of
scale, not a sample-size problem.

PRIME SUSPECT, checked first (Step 0): the noisy-OR target-construction
pipeline (denial_reasons.py) assigns `deprecated_code` (CARC 181,
base_prob=0.85 -- the highest of any risk factor) to claims billed with
HCPCS codes CMS stopped recognizing for Part B payment in 2010
(99241-99245, 99251-99255) -- see decisions-and-learnings.md's Addendum #6
investigation. If HCPCS_CD is genuinely near-deterministic of is_denied
for those specific codes, and any of them made the cardinality encoding's
top-20 cut in build_chow_design_matrix.py, that single dummy column
(interacted with claim_type in the unrestricted model) is a textbook,
severe form of near-complete separation -- and critically, that
relationship doesn't get weaker with more data, exactly matching what
was observed.

Step 1 checks something more severe and more general: whether any RAW
risk-factor/target-construction column itself (not just a code that
happens to correlate with the label, but a column that was a literal
INGREDIENT of the noisy-OR formula used to build is_denied) survived into
train_model.parquet as a feature. build_chow_design_matrix.py's Pass 1
loop explicitly skips (leaves untouched, does NOT drop) any column
starting with "risk_" -- if such a column exists and was never removed
from the final feature set, that's outright target leakage, not just a
strong predictor, and would produce this exact sample-size-invariant
degenerate fit. This script does NOT assume "risk_" or any other specific
name from memory of the code/docs -- it reads the LIVE parquet schema and
searches by substring, so a naming mismatch between what the docs
describe and what actually shipped doesn't produce a false negative.

Step 2 crosstabs whatever Step 0/1 found against is_denied, overall and
by claim type.

Step 3 is a general near-complete-separation scan across the design
matrix's actual encoded dummy groups (state_*, hcpcs_*, dgns_*, the 5 NPI
flags) by claim type -- flagging any (value, claim_type) cell with a
denial rate at or near 0%/100%, split into "broad-coverage" (large enough
N to meaningfully destabilize a fit) vs "sparse" (small N -- still a
technically-separating cell, but less consequential) using --min-cell-n.

Run with (from the repo root, inside the venv):
    python src/diagnose_separation.py
    python src/diagnose_separation.py --sample-frac 0.2   # match the run that reproduced the finding
"""

from __future__ import annotations

import argparse

import pandas as pd
import pyarrow.parquet as pq

from build_chow_design_matrix import build_chow_design_matrix
from fit_chow_test import LABEL_COL, TRAIN_PARQUET_PATH, _load_train

# CMS stopped recognizing these consultation codes for Part B payment
# effective 2010-01-01 (99241 itself deleted from CPT entirely effective
# 2021) -- decisions-and-learnings.md's Addendum #6 investigation, the
# origin of the deprecated_code risk factor (base_prob=0.85, the highest
# of any factor). Checked directly here rather than assumed to be
# "handled" just because that risk factor exists in the target-
# construction code -- the question is whether the RAW HCPCS_CD value is
# ALSO doing this work as a raw feature, independent of the label.
_CMS_DEPRECATED_CONSULT_CODES = {
    "99241", "99242", "99243", "99244", "99245",
    "99251", "99252", "99253", "99254", "99255",
}

# Substrings that would flag a column as target-leakage-shaped, based on
# the documented risk-factor/target-construction naming in
# decisions-and-learnings.md and data/TARGET_DEFINITION.md -- matched
# broadly (not as an exact hardcoded list) so a naming difference between
# the docs and the live file doesn't produce a false negative.
_LEAKAGE_SUSPECT_SUBSTRINGS = ("risk", "rule_", "carc", "denial_reason", "reason_priority")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
        help="Fraction of train_model.parquet to check, streamed the same "
        "memory-safe way as fit_chow_test.py (default 0.2, matching the run "
        "that reproduced the identical-log-likelihood finding). This is a "
        "descriptive-crosstab script, not a fit, so pass up to 1.0 for the "
        "full file if you want the most complete picture -- it still goes "
        "through the safe streaming path, not a raw full-file load.",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=50_000,
        help="Same meaning as in fit_chow_test.py (default 50,000).",
    )
    parser.add_argument(
        "--min-cell-n",
        type=int,
        default=30,
        help="Minimum (covariate=value, claim_type) cell size to report as a "
        "'broad-coverage' separation risk (default 30) -- smaller cells are "
        "reported separately as 'sparse', since they matter less to overall "
        "fit stability even when perfectly separated.",
    )
    return parser.parse_args()


def _add_claim_type_label(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a single readable claim_type column from the
    claim_type_carrier/outpatient/dme cell-means dummies, for crosstabs
    below. Doesn't touch the original dummy columns."""
    claim_type_cols = [c for c in df.columns if c.startswith("claim_type_")]
    if not claim_type_cols:
        return df
    label = pd.Series("UNKNOWN", index=df.index)
    for col in claim_type_cols:
        name = col.removeprefix("claim_type_")
        label = label.mask(df[col] == 1, name)
    df = df.copy()
    df["_claim_type"] = label
    return df


def _crosstab(df: pd.DataFrame, col: str, by_claim_type: bool) -> None:
    print(f"\n-- {col} -- dtype={df[col].dtype}, distinct values={df[col].nunique(dropna=False)}")
    overall = df.groupby(col, dropna=False)[LABEL_COL].agg(["mean", "count"]).sort_values("count", ascending=False)
    print("  overall denial rate by value (top 20 by count):")
    print(overall.head(20).to_string())
    if by_claim_type and "_claim_type" in df.columns:
        by_type = (
            df.groupby([col, "_claim_type"], dropna=False)[LABEL_COL]
            .agg(["mean", "count"])
            .sort_values("count", ascending=False)
        )
        print("  denial rate by value x claim_type (top 20 by count):")
        print(by_type.head(20).to_string())


def main() -> None:
    args = parse_args()

    all_columns = pq.ParquetFile(TRAIN_PARQUET_PATH).schema.names
    print(f"train_model.parquet has {len(all_columns)} columns total.\n")

    train = _load_train(args.sample_frac, args.stream_batch_size)
    intermediate = build_chow_design_matrix(train)
    del train
    intermediate = _add_claim_type_label(intermediate)

    # --- Step 0: the specific, already-documented deprecated_code lead ---
    print("=" * 78)
    print("STEP 0 -- CMS-deprecated consultation codes (deprecated_code's own basis)")
    print("=" * 78)
    # HCPCS_CD itself was consumed by build_chow_design_matrix() into
    # hcpcs_* dummies and dropped from `intermediate` -- read it fresh,
    # narrowly (one column), rather than threading it through the encoding
    # call above.
    hcpcs_raw = pd.read_parquet(TRAIN_PARQUET_PATH, columns=["HCPCS_CD", LABEL_COL])
    if args.sample_frac < 1.0:
        hcpcs_raw = hcpcs_raw.sample(frac=args.sample_frac, random_state=42)
    is_deprecated = hcpcs_raw["HCPCS_CD"].isin(_CMS_DEPRECATED_CONSULT_CODES)
    n_deprecated = int(is_deprecated.sum())
    print(f"  Rows with a CMS-deprecated consult HCPCS code: {n_deprecated:,} of {len(hcpcs_raw):,}")
    if n_deprecated:
        rate = hcpcs_raw.loc[is_deprecated, LABEL_COL].mean()
        print(f"  Denial rate among them: {rate:.4f}  (vs {hcpcs_raw[LABEL_COL].mean():.4f} overall)")
        made_top20 = [
            c for c in intermediate.columns
            if c.startswith("hcpcs_") and c.removeprefix("hcpcs_") in _CMS_DEPRECATED_CONSULT_CODES
        ]
        print(f"  Of these, made the top-20 cardinality-encoding cut as their own dummy column: {made_top20 or 'none (absorbed into hcpcs___OTHER__)'}")
    del hcpcs_raw

    # --- Step 1: leakage-shaped column NAMES, in the live schema ---
    print("\n" + "=" * 78)
    print("STEP 1 -- leakage-shaped column names in the live schema")
    print("=" * 78)
    suspects = sorted(
        c for c in all_columns
        if c != LABEL_COL and any(s in c.lower() for s in _LEAKAGE_SUSPECT_SUBSTRINGS)
    )
    if suspects:
        print(f"  Found {len(suspects)}: {suspects}")
    else:
        print(
            "  None found by substring match. Doesn't rule out leakage under a "
            "different naming convention -- the full column list is printed "
            "at the end of this script's output for a manual look if Steps "
            "0/3 don't explain the separation either."
        )

    # --- Step 2: crosstab whatever Step 1 found ---
    if suspects:
        print("\n" + "=" * 78)
        print("STEP 2 -- suspect column crosstabs vs is_denied")
        print("=" * 78)
        present_in_intermediate = [c for c in suspects if c in intermediate.columns]
        missing_from_intermediate = [c for c in suspects if c not in intermediate.columns]
        if missing_from_intermediate:
            print(
                f"  [note] {missing_from_intermediate} matched by name but aren't in the "
                "post-encoding frame (likely consumed/renamed during encoding) -- "
                "reading them fresh from the raw file instead."
            )
            extra = pd.read_parquet(TRAIN_PARQUET_PATH, columns=missing_from_intermediate + [LABEL_COL])
            if args.sample_frac < 1.0:
                extra = extra.sample(frac=args.sample_frac, random_state=42)
            for col in missing_from_intermediate:
                _crosstab(extra, col, by_claim_type=False)
            del extra
        for col in present_in_intermediate:
            _crosstab(intermediate, col, by_claim_type=True)

    # --- Step 3: general near-separation scan across the encoded dummy groups ---
    print("\n" + "=" * 78)
    print("STEP 3 -- near-complete-separation scan (encoded dummies x claim_type)")
    print(f"Flagging cells with denial rate <=2% or >=98%; min N for 'broad' = {args.min_cell_n}")
    print("=" * 78)
    dummy_prefixes = ("state_", "hcpcs_", "dgns_")
    npi_flags = {
        "has_referring_physician", "has_performing_physician",
        "has_attending_physician", "has_operating_physician",
        "has_rendering_physician",
    }
    scan_cols = [
        c for c in intermediate.columns
        if (c.startswith(dummy_prefixes) or c in npi_flags) and c != LABEL_COL
    ]
    broad_flags, sparse_flags = [], []
    for col in scan_cols:
        grouped = intermediate.groupby([col, "_claim_type"])[LABEL_COL].agg(["mean", "count"])
        near_sep = grouped[(grouped["mean"] <= 0.02) | (grouped["mean"] >= 0.98)]
        for (value, ctype), row in near_sep.iterrows():
            if value == 0:
                continue  # the "dummy is absent" side isn't the separation risk
            entry = (col, ctype, float(row["mean"]), int(row["count"]))
            (broad_flags if row["count"] >= args.min_cell_n else sparse_flags).append(entry)

    def _print_flags(flags: list[tuple[str, str, float, int]], title: str) -> None:
        print(f"\n{title} ({len(flags)}):")
        for col, ctype, rate, n in sorted(flags, key=lambda f: -f[3])[:30]:
            print(f"    {col}=1 x claim_type={ctype}: denial rate={rate:.3f}, n={n}")
        if len(flags) > 30:
            print(f"    ... and {len(flags) - 30} more (top 30 by cell size shown)")

    _print_flags(broad_flags, "BROAD-COVERAGE near-separation")
    _print_flags(sparse_flags, "SPARSE near-separation")

    print("\n" + "=" * 78)
    print("Full column list (for a manual look if nothing above explains it):")
    print("=" * 78)
    print(all_columns)


if __name__ == "__main__":
    main()
