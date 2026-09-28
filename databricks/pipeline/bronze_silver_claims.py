"""
Bronze + silver claims: the Spark counterpart of src/load_data.py.

bronze_claims_<type>  One table per raw CMS RIF file, every column kept as the
                      exact string that was in the pipe-delimited CSV (an
                      empty field lands as null -- Spark's CSV reader -- and
                      that is the only difference, checked cell by cell
                      against the file), plus _source_file and _file_row (the
                      line's position in the file). Nothing interpreted, so
                      any later step can be re-derived from here.
silver_claims         The three files unioned into one typed table with a
                      global _row_id -- row for row the same thing
                      load_data.load_and_concat() produces in pandas
                      (data/processed/combined_claims_raw.parquet).

Typing rule, mirroring what pandas.read_csv does in load_data.py so the two
pipelines agree without a hand-maintained schema:
  1. Pandas' default missing-value strings ("", "NA", "NULL", "nan", ...) -> null.
     pandas applies these to every column, including the ones forced to str.
  2. Columns load_data forces to str (_CD / _NUM suffix, CODE_DTYPE_COLS) stay
     strings -- leading zeros in codes matter.
  3. Any other column is numeric iff every non-null value in every file parses
     as a number (pandas' own inference). It is a 64-bit integer only if it
     also has no nulls in the combined table and every value is a plain
     integer (pandas upcasts an int column that gains NaN in the concat to
     float64); otherwise double.
Verified against the pandas output column by column in parity_check.py.
"""

from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .common import CLAIM_FILES, RAW_VOLUME, table, write_delta

from load_data import CODE_DTYPE_COLS  # noqa: E402  (src/ on sys.path via common)

# pandas.read_csv's default na_values (pandas/_libs/parsers.pyx STR_NA_VALUES).
PANDAS_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
    "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
]
_NUMERIC_RE = r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$"
_INTEGER_RE = r"^\s*[-+]?\d+\s*$"


def _forced_str(col: str) -> bool:
    return col in CODE_DTYPE_COLS or col.endswith("_CD") or col.endswith("_NUM")


def build_bronze(spark) -> dict[str, str]:
    """Read each raw file as all-strings and land it unchanged. Returns
    {file name: bronze table name}."""
    out = {}
    for fname in CLAIM_FILES:
        raw = (
            spark.read.option("header", True)
            .option("sep", "|")
            .option("inferSchema", False)
            .option("mode", "FAILFAST")
            .csv(f"{RAW_VOLUME}/{fname}")
        )
        # Position of each line within its file. monotonically_increasing_id
        # is increasing in file order for a single file (partition index is
        # the high bits, row-within-partition the low bits); row_number turns
        # it into a dense 0..n-1. Checked against pandas' positional order in
        # parity_check.py.
        w = Window.orderBy("_mono")
        df = (
            raw.withColumn("_mono", F.monotonically_increasing_id())
            .withColumn("_file_row", F.row_number().over(w) - 1)
            .drop("_mono")
            .withColumn("_source_file", F.lit(fname))
        )
        name = f"bronze_claims_{fname.removesuffix('.csv')}"
        write_delta(df, name, f"Raw CMS Synthetic RIF {fname}, every field as the original string")
        out[fname] = table(name)
    return out


def build_silver(spark, bronze_tables: dict[str, str]) -> DataFrame:
    """Union + type the bronze tables into silver_claims."""
    frames = {f: spark.table(t) for f, t in bronze_tables.items()}

    # Column order = pandas.concat(sort=False) order: first file's columns,
    # then each later file's new columns in order of appearance.
    # load_data adds _source_file per file BEFORE the concat, so it lands right
    # after the first file's own columns, ahead of later files' extra columns.
    ordered: list[str] = []
    for f in CLAIM_FILES:
        for c in frames[f].columns:
            if c != "_file_row" and c not in ordered:
                ordered.append(c)

    def normalize(df: DataFrame) -> DataFrame:
        cols = []
        for c in ordered:
            if c == "_source_file":
                cols.append(F.col(c))
            elif c in df.columns:
                cols.append(F.when(F.col(c).isin(PANDAS_NA_VALUES), None).otherwise(F.col(c)).alias(c))
            else:
                cols.append(F.lit(None).cast("string").alias(c))
        return df.select(*cols, "_file_row")

    # Global _row_id: files in CLAIM_FILES order, lines in file order.
    counts = {f: frames[f].count() for f in CLAIM_FILES}
    offsets, running = {}, 0
    for f in CLAIM_FILES:
        offsets[f] = running
        running += counts[f]
    parts = [
        normalize(frames[f]).withColumn("_row_id", F.col("_file_row") + F.lit(offsets[f]))
        for f in CLAIM_FILES
    ]
    # a.unionByName(b), not reduce(DataFrame.unionByName, ...): inside a Databricks
    # serverless Job the frames are Spark Connect DataFrames, and the classic
    # class's unbound method reaches for the JVM (_jdf), which serverless forbids.
    unioned = reduce(lambda a, b: a.unionByName(b), parts)

    # Infer numeric columns exactly once, over the whole union.
    candidates = [c for c in ordered if not _forced_str(c) and c != "_source_file"]
    stats = unioned.agg(
        *[F.sum(F.when(F.col(c).isNotNull() & ~F.col(c).rlike(_NUMERIC_RE), 1).otherwise(0)).alias(f"bad__{c}") for c in candidates],
        *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null__{c}") for c in candidates],
        *[F.sum(F.when(F.col(c).isNotNull() & ~F.col(c).rlike(_INTEGER_RE), 1).otherwise(0)).alias(f"nonint__{c}") for c in candidates],
    ).collect()[0].asDict()

    typed = []
    for c in ordered:
        if c == "_source_file" or _forced_str(c) or stats[f"bad__{c}"] > 0:
            typed.append(F.col(c))
        elif stats[f"null__{c}"] == 0 and stats[f"nonint__{c}"] == 0:
            typed.append(F.trim(F.col(c)).cast("long").alias(c))
        else:
            typed.append(F.trim(F.col(c)).cast("double").alias(c))
    silver = unioned.select(*typed, "_row_id")
    write_delta(
        silver,
        "silver_claims",
        "Carrier + outpatient + DME claim lines, typed, one global _row_id (= src/load_data.py output)",
    )
    return silver
