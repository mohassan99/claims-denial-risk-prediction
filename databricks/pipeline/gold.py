"""
Gold: the Spark counterpart of the split in build_target_and_split.py and of
src/build_features.py.

gold_labeled_claims_for_eda   label-construction columns for EDA / Phase 4-5
gold_{train,val,test}_model   the modeling tables (= data/processed/*_model.parquet)

Each gold model table carries _split_row, the row's position in its split in
the order pandas writes it. It is not a feature: consumers sort by it so a
model fit from the Delta table sees rows in the same order as one fit from the
parquet (tree histograms can depend on row order).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from .common import write_delta

import build_features as bf  # noqa: E402

LEAKAGE_EXACT = ["p_denied_model", "denial_reason_carc_1", "denial_reason_carc_2", "CLM_PMT_AMT"]
SPLITS = (("train", 0.0, 0.64), ("val", 0.64, 0.8), ("test", 0.8, 1.0))
DATE_COL = "CLM_FROM_DT"


def build_eda_table(labeled: DataFrame) -> None:
    cols = ["_row_id", "BENE_ID", "CLM_ID"] + [c for c in labeled.columns if c.startswith("risk_")] + [
        "p_denied_model", "is_denied", "denial_reason_carc_1", "denial_reason_carc_2"]
    write_delta(labeled.select(*[c for c in cols if c in labeled.columns]), "gold_labeled_claims_for_eda",
                "Label-construction columns (risk factors, P(denied), CARC reasons) for EDA and Phase 4-5; never a model input")


def assign_splits(labeled: DataFrame) -> DataFrame:
    """Reproduces build_target_and_split.py: a true time-based split -- sort by
    the PARSED claim date (stable, so lines sharing a date keep _row_id order;
    unparseable dates last) and cut at 64% / 80%.

    History: until 2026-09-27 both pipelines sorted the raw date STRING
    ("28-Sep-2015"), which orders by day of month first -- a day-of-month
    split mislabeled as time-based. Found during this port; fixed in both
    places together (see TARGET_DEFINITION.md's 2026-09-27 addendum)."""
    n = labeled.count()
    parsed = F.to_date(DATE_COL, "dd-MMM-yyyy")
    parsed_share = labeled.agg(F.avg(parsed.isNotNull().cast("double"))).collect()[0][0]
    if parsed_share <= 0.9:
        raise NotImplementedError("pandas falls back to a stratified random split here; not ported")
    w = Window.orderBy(parsed.asc_nulls_last(), F.col("_row_id"))
    pos = labeled.withColumn("_pos", F.row_number().over(w) - 1)
    cut_train, cut_val = int(n * 0.64), int(n * 0.8)
    return (pos.withColumn("_split", F.when(F.col("_pos") < cut_train, "train")
                           .when(F.col("_pos") < cut_val, "val").otherwise("test"))
            .withColumn("_split_row", F.col("_pos") - F.when(F.col("_split") == "train", 0)
                        .when(F.col("_split") == "val", cut_train).otherwise(cut_val)))


def features_for_split(df: DataFrame) -> DataFrame:
    """build_features.build_features(), column for column, on one split."""
    leakage = [c for c in df.columns if c.startswith("risk_")] + LEAKAGE_EXACT
    base_cols = [c for c in df.columns if c not in leakage and c not in ("_pos", "_split", "_split_row")]
    out = df.select(*base_cols, "_split_row")

    for npi_col, flag in bf.NPI_ROLE_FIELDS.items():
        if npi_col in out.columns:
            out = out.withColumn(flag, F.col(npi_col).isNotNull().cast("long"))

    if "PRVDR_STATE_CD" in out.columns:
        xwalk = F.create_map(*[F.lit(x) for kv in bf.SSA_STATE_CROSSWALK.items() for x in kv])
        out = out.withColumn("provider_state", F.coalesce(xwalk[F.col("PRVDR_STATE_CD")], F.col("PRVDR_STATE_CD")))
        out = out.drop("PRVDR_STATE_CD")

    if "_source_file" in out.columns:
        for src, ct in bf.SOURCE_FILE_TO_CLAIM_TYPE.items():
            out = out.withColumn(f"claim_type_{ct}", (F.col("_source_file") == src).cast("long"))
        out = out.drop("_source_file")

    out = out.drop(*[c for c in bf.DROP_COLUMNS if c in out.columns])

    # fill_claim_type_exclusive_fields: string columns that are 100% null in
    # some claim type (within THIS split) get "NOT_APPLICABLE" for that type.
    skip = bf._ALREADY_HANDLED | bf._DATE_LIKE_COLS | set(bf._CLAIM_TYPE_COLS) | {"_split_row"}
    str_cols = [f.name for f in out.schema.fields
                if isinstance(f.dataType, StringType) and f.name not in skip
                and not f.name.startswith("claim_type_") and not f.name.startswith("risk_")]
    rates = out.agg(*[
        F.avg(F.when(F.col(ct) == 1, F.col(c).isNull().cast("double"))).alias(f"{c}||{ct}")
        for c in str_cols for ct in bf._CLAIM_TYPE_COLS
    ]).collect()[0].asDict()
    fills = []
    for c in str_cols:
        absent = [ct for ct in bf._CLAIM_TYPE_COLS if rates[f"{c}||{ct}"] == 1.0]
        if absent:
            mask = F.lit(False)
            for ct in absent:
                mask = mask | (F.col(ct) == 1)
            fills.append((c, mask))
    for c, mask in fills:
        out = out.withColumn(c, F.when(mask, F.lit("NOT_APPLICABLE")).otherwise(F.col(c)))
    return out


def build_gold(spark, labeled: DataFrame) -> None:
    build_eda_table(labeled)
    split = assign_splits(labeled)
    for name, _, _ in SPLITS:
        feats = features_for_split(split.where(F.col("_split") == name))
        write_delta(feats, f"gold_{name}_model",
                    f"{name} split, model-ready features (= data/processed/{name}_model.parquet); sort by _split_row")
