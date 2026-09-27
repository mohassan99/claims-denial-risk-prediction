"""
Run the Databricks medallion pipeline (bronze -> silver -> gold) end to end.

    python databricks/run_pipeline.py                 # all layers
    python databricks/run_pipeline.py --steps silver_label gold
    python databricks/run_pipeline.py --steps parity  # compare with pandas outputs
    python databricks/run_pipeline.py --upload-reference   # push pandas parquet outputs for parity

Locally this uses Databricks Connect (serverless) with DATABRICKS_HOST /
DATABRICKS_TOKEN from the repo's .env; as a Databricks Job task it uses the
job's own Spark session. Reads raw CSVs from /Volumes/<catalog>/<schema>/raw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # Databricks serverless python tasks exec() the file without __file__
    _HERE = Path(sys.argv[sys.argv.index("--repo-root") + 1]) / "databricks"
sys.path.insert(0, str(_HERE))

from pipeline.common import REFERENCE_VOLUME, REPO_ROOT, get_spark, table  # noqa: E402

REPORTS_VOLUME = REFERENCE_VOLUME.rsplit("/", 1)[0] + "/reports"

STEPS = ["bronze", "silver", "silver_label", "gold", "parity"]


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        import os

        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _save_report(name: str, text: str) -> None:
    """reports/<name> in the repo checkout, plus a copy in the `reports`
    volume so a scheduled Job's output survives without a writable checkout."""
    try:
        (REPO_ROOT / "reports" / name).write_text(text)
        print(f"  -> reports/{name}")
    except OSError as exc:
        print(f"  (could not write reports/{name} in the checkout: {exc})")
    try:
        from databricks.sdk import WorkspaceClient
        import io

        get_spark().sql(f"CREATE VOLUME IF NOT EXISTS {table('reports')}")
        WorkspaceClient().files.upload(f"{REPORTS_VOLUME}/{name}", io.BytesIO(text.encode()), overwrite=True)
        print(f"  -> {REPORTS_VOLUME}/{name}")
    except Exception as exc:  # report copy is best-effort; the pipeline result is the tables
        print(f"  (could not copy {name} to the reports volume: {exc})")


def upload_reference() -> None:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    spark = get_spark()
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {table('reference')} "
              "COMMENT 'pandas pipeline outputs (src/), used only for parity checks'")
    proc = REPO_ROOT / "data" / "processed"
    for name in ["combined_claims_raw.parquet", "labeled_claims_for_eda.parquet",
                 "train_model.parquet", "val_model.parquet", "test_model.parquet"]:
        with open(proc / name, "rb") as fh:
            w.files.upload(f"{REFERENCE_VOLUME}/{name}", fh, overwrite=True)
        print(f"  uploaded {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=STEPS[:-1], choices=STEPS)
    ap.add_argument("--upload-reference", action="store_true")
    ap.add_argument("--repo-root", help="set by the Databricks Job (no __file__ there)")
    args = ap.parse_args()
    _load_dotenv()

    if args.upload_reference:
        upload_reference()
        return

    from pipeline.bronze_silver_claims import build_bronze, build_silver
    from pipeline.common import CLAIM_FILES

    spark = get_spark()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {table('x').rsplit('.', 1)[0]}")
    bronze_tables = {f: table(f"bronze_claims_{f.removesuffix('.csv')}") for f in CLAIM_FILES}

    if "bronze" in args.steps:
        print("== bronze"); bronze_tables = build_bronze(spark)
    if "silver" in args.steps:
        print("== silver"); build_silver(spark, bronze_tables)
    if "silver_label" in args.steps:
        from pipeline.silver_label import build_silver_labeled

        print("== silver_label")
        _, audit = build_silver_labeled(spark, spark.table(table("silver_claims")))
        print(audit)
        _save_report("label_audit_databricks.txt", audit)
    if "gold" in args.steps:
        from pipeline.gold import build_gold

        print("== gold"); build_gold(spark, spark.table(table("silver_claims_labeled")))
    if "parity" in args.steps:
        from pipeline.parity_check import run_all

        print("== parity")
        reports = run_all(spark)
        _save_report("databricks_parity.json", json.dumps(reports, indent=2, default=str))
        if not all(r["PASS"] for r in reports):
            sys.exit(1)


if __name__ == "__main__":
    main()
