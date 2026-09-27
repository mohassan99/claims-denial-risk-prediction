"""
Shared plumbing for the Databricks (PySpark + Delta) port of the Phase 1 pipeline.

The business logic is NOT redefined here. Every constant that decides the
label or the features (risk-factor base probabilities, CARC codes, the HCPCS ->
expected-diagnosis table, the provider-key fields, DROP_COLUMNS, the SSA state
crosswalk, ...) is imported from the pandas modules in src/, which stay the
single source of truth. The Spark code only re-expresses the *mechanics*
(groupbys, windows, joins) in a form that scales past one machine's memory.

Runs two ways, same code:
  - from a laptop / cloud sandbox via Databricks Connect (serverless compute),
  - as a Databricks Job task on serverless compute, where `spark` already exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CATALOG = os.environ.get("CLAIMS_CATALOG", "workspace")
SCHEMA = os.environ.get("CLAIMS_SCHEMA", "claims_denial")
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
REFERENCE_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/reference"

# Same default claim set and order as src/load_data.load_and_concat(): the
# order fixes _row_id, which the label's random draws are keyed to.
CLAIM_FILES = ("carrier.csv", "outpatient.csv", "dme.csv")


def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


def get_spark():
    """Native SparkSession inside Databricks; Databricks Connect (serverless)
    everywhere else. Connect reads DATABRICKS_HOST / DATABRICKS_TOKEN from the
    environment (the repo's gitignored .env)."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.serverless(True).getOrCreate()


def write_delta(df, name: str, comment: str | None = None) -> None:
    """Overwrite a managed Delta table (schema included), then set its comment."""
    full = table(name)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full)
    if comment:
        spark = df.sparkSession
        spark.sql(f"COMMENT ON TABLE {full} IS '{comment.replace(chr(39), chr(39) * 2)}'")
    print(f"  wrote {full}")
