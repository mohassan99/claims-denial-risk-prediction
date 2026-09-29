"""
Build data/processed/val_sample_v2.parquet: a claim sample from the current (time-based) val split.

Reads the Delta table workspace.claims_denial.gold_val_model in Databricks through the SQL Statement
API, taking N claims per claim type in a fixed hash order (deterministic). Needs DATABRICKS_HOST and
DATABRICKS_TOKEN in .env. Uses the workspace's first SQL warehouse (Free Edition has one serverless
starter warehouse, which starts on demand).

    python scripts/make_val_sample.py [--n-per-type 200]

The old data/processed/val_sample.parquet (made from a local val_model.parquet written before the
2026-09-27 split fix) is stale: do not use it for evaluation.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
NUMERIC = {"INT", "LONG", "SHORT", "BYTE", "DOUBLE", "FLOAT", "DECIMAL", "BOOLEAN"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=200)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "val_sample_v2.parquet")
    a = ap.parse_args()

    load_dotenv(REPO / ".env")
    host, token = os.environ["DATABRICKS_HOST"].rstrip("/"), os.environ["DATABRICKS_TOKEN"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    wh = requests.get(f"{host}/api/2.0/sql/warehouses", headers=h, timeout=60).json()["warehouses"][0]["id"]
    sql = f"""SELECT * FROM (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY claim_type_carrier, claim_type_outpatient, claim_type_dme
                                   ORDER BY xxhash64(_split_row)) AS _rn
      FROM workspace.claims_denial.gold_val_model) WHERE _rn <= {a.n_per_type}"""
    r = requests.post(f"{host}/api/2.0/sql/statements", headers=h, timeout=120, json={
        "warehouse_id": wh, "statement": sql, "wait_timeout": "50s", "format": "JSON_ARRAY", "disposition": "INLINE"}).json()
    if r["status"]["state"] != "SUCCEEDED":
        raise SystemExit(f"query did not finish: {r['status']}")
    cols = [c["name"] for c in r["manifest"]["schema"]["columns"]]
    types = {c["name"]: c["type_name"] for c in r["manifest"]["schema"]["columns"]}
    rows, nxt = r["result"]["data_array"], r["result"].get("next_chunk_internal_link")
    while nxt:
        c = requests.get(host + nxt, headers=h, timeout=120).json()
        rows += c["data_array"]
        nxt = c.get("next_chunk_internal_link")
    df = pd.DataFrame(rows, columns=cols)
    for c, t in types.items():
        if t in NUMERIC:
            df[c] = pd.to_numeric(df[c].replace({"true": 1, "false": 0}), errors="coerce")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"wrote {a.out} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
