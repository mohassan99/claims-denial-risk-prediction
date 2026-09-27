"""
Parity check: every Delta table the Spark pipeline writes vs. the parquet file
the pandas pipeline writes for the same step, compared row by row on _row_id.

For each pair it checks
  - identical column names in identical order,
  - identical type per column (string / 64-bit int / double),
  - identical row set (same _row_id values, no extras, none missing),
  - identical value in every cell (null-safe; doubles must match exactly --
    NaN on the pandas side counts as null),
  - for the model tables, identical row ORDER (gold _split_row vs the row's
    position in the parquet file).
The reference parquet files are produced by the pandas scripts in src/ and
uploaded to the `reference` volume (see run_pipeline.py --upload-reference).
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType

from .common import REFERENCE_VOLUME, table

PAIRS = [
    ("silver_claims", "combined_claims_raw.parquet", False),
    ("gold_labeled_claims_for_eda", "labeled_claims_for_eda.parquet", False),
    ("gold_train_model", "train_model.parquet", True),
    ("gold_val_model", "val_model.parquet", True),
    ("gold_test_model", "test_model.parquet", True),
]
HELPER_COLS = {"_split_row"}


def _kind(dt) -> str:
    if isinstance(dt, StringType):
        return "string"
    if isinstance(dt, (LongType, IntegerType)):
        return "int"
    if isinstance(dt, (DoubleType, FloatType)):
        return "double"
    return dt.simpleString()


def compare(spark, delta_name: str, parquet_name: str, check_order: bool) -> dict:
    got = spark.table(table(delta_name))
    ref = spark.read.parquet(f"{REFERENCE_VOLUME}/{parquet_name}").select("*", F.col("_metadata.row_index").alias("_ref_pos"))
    got_cols = [c for c in got.columns if c not in HELPER_COLS]
    ref_cols = [c for c in ref.columns if c != "_ref_pos"]
    report = {"table": delta_name, "columns_match": got_cols == ref_cols}
    if got_cols != ref_cols:
        report["only_in_spark"] = [c for c in got_cols if c not in ref_cols]
        report["only_in_pandas"] = [c for c in ref_cols if c not in got_cols]
        report["order_diff"] = [(i, a, b) for i, (a, b) in enumerate(zip(got_cols, ref_cols)) if a != b][:5]
    common = [c for c in got_cols if c in ref_cols]
    gtypes = {f.name: _kind(f.dataType) for f in got.schema.fields}
    rtypes = {f.name: _kind(f.dataType) for f in ref.schema.fields}
    report["type_mismatches"] = {c: (gtypes[c], rtypes[c]) for c in common if gtypes[c] != rtypes[c]}

    counts = got.agg(F.count(F.lit(1)).alias("n")).collect()[0]["n"], ref.count()
    report["rows"] = {"spark": counts[0], "pandas": counts[1]}
    g = got.select(*[F.col(c).alias(f"g__{c}") for c in got.columns])
    r = ref.select(*[F.col(c).alias(f"r__{c}") for c in ref.columns])
    j = g.join(r, F.col("g___row_id") == F.col("r___row_id"), "full_outer")

    def same(c):
        a, b = F.col(f"g__{c}"), F.col(f"r__{c}")
        if rtypes[c] == "double":
            b = F.when(F.isnan(b), None).otherwise(b)
        if gtypes[c] == "double":
            a = F.when(F.isnan(a), None).otherwise(a)
        if gtypes[c] != rtypes[c]:
            a, b = a.cast("string"), b.cast("string")
        return a.eqNullSafe(b)

    aggs = [F.sum(F.when(F.col("g___row_id").isNull(), 1).otherwise(0)).alias("missing_in_spark"),
            F.sum(F.when(F.col("r___row_id").isNull(), 1).otherwise(0)).alias("missing_in_pandas")]
    aggs += [F.sum(F.when(~same(c), 1).otherwise(0)).alias(f"diff__{c}") for c in common if c != "_row_id"]
    if check_order:
        aggs.append(F.sum(F.when(F.col("g___split_row") != F.col("r___ref_pos"), 1).otherwise(0)).alias("order_diffs"))
    res = j.agg(*aggs).collect()[0].asDict()
    report["missing_in_spark"] = res.pop("missing_in_spark")
    report["missing_in_pandas"] = res.pop("missing_in_pandas")
    if check_order:
        report["row_order_diffs"] = res.pop("order_diffs")
    report["cell_mismatches"] = {k.removeprefix("diff__"): v for k, v in res.items() if v}
    report["PASS"] = (report["columns_match"] and not report["type_mismatches"]
                      and counts[0] == counts[1] and not report["missing_in_spark"]
                      and not report["missing_in_pandas"] and not report["cell_mismatches"]
                      and not report.get("row_order_diffs"))
    return report


def run_all(spark) -> list[dict]:
    reports = []
    for delta_name, parquet_name, order in PAIRS:
        rep = compare(spark, delta_name, parquet_name, order)
        print(f"{'PASS' if rep['PASS'] else 'FAIL'}  {delta_name} vs {parquet_name}: {rep}")
        reports.append(rep)
    return reports
