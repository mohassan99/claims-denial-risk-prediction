"""
Phase 3 -- prove the live Azure ML endpoint returns the same predictions as
the model that produced the reported Phase 2 metrics.

Sends a label-stratified sample of val_model.parquet rows (raw columns, as a
client would send them -- including the label and IDs, which the endpoint
must ignore) to the endpoint over HTTPS and compares each returned
P(is_denied) with the local model's prediction for the same row via
fit_xgboost.py's own categorical transform. Also compares per-claim-type
PR-AUC on the sample, local vs. endpoint.

Pass criterion: max |endpoint - local| <= 1e-6 (the endpoint rounds to 6
decimals, so ~5e-7 is the expected floor).

Needs: az login, .env, reports/xgboost_model*.json, data/processed/val_model.parquet.
Run (repo root):  python deploy/verify_endpoint.py [--n 3000]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fit_xgboost import _apply_categories  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ENDPOINT = "claims-denial-xgb"
BATCH = 500
TOL = 1e-6


def _env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _az(*args: str) -> str:
    az = os.environ.get("AZ", "az")
    return subprocess.run([az, *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    env = _env()
    ws = ["-g", env["AZURE_RESOURCE_GROUP"], "-w", env["AZURE_ML_WORKSPACE"]]
    uri = _az("ml", "online-endpoint", "show", "-n", ENDPOINT, *ws, "--query", "scoring_uri", "-o", "tsv")
    key = _az("ml", "online-endpoint", "get-credentials", "-n", ENDPOINT, *ws, "--query", "primaryKey", "-o", "tsv")

    meta = json.loads((ROOT / "reports/xgboost_model_meta.json").read_text())
    model = XGBClassifier()
    model.load_model(ROOT / "reports/xgboost_model.json")

    val = pd.read_parquet(ROOT / "data/processed/val_model.parquet")
    # label-stratified sample so PR-AUC on it is meaningful
    # (pandas 3's groupby.apply drops the grouping column, so sample per label explicitly)
    frac = args.n / len(val)
    sub = pd.concat([val[val["is_denied"] == k].sample(frac=frac, random_state=42) for k in (0, 1)])
    X_local = _apply_categories(sub[meta["colnames"]], meta["cat_cols"], meta["categories_by_col"])
    p_local = model.predict_proba(X_local)[:, 1]

    records = json.loads(sub.to_json(orient="records"))
    p_ep: list[float] = []
    for i in range(0, len(records), BATCH):
        body = json.dumps({"input_data": records[i : i + BATCH]}).encode()
        req = urllib.request.Request(uri, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError(out["error"])
        p_ep.extend(out["p_denied"])
    p_ep = np.asarray(p_ep)

    diff = np.abs(p_ep - p_local)
    y = sub["is_denied"].to_numpy()
    lines = [
        "Phase 3 -- live endpoint vs. local model, val sample",
        "=" * 56,
        f"endpoint: {ENDPOINT} ({uri.split('/')[2]})",
        f"rows scored: {len(p_ep):,} (label-stratified sample of val, {y.mean():.2%} positive)",
        f"max |endpoint - local|: {diff.max():.2e}   mean: {diff.mean():.2e}   (tolerance {TOL:.0e})",
        "",
        f"{'':12s}{'n':>7s}{'PR-AUC local':>15s}{'PR-AUC endpoint':>18s}",
    ]
    for ct in ["all", "carrier", "outpatient", "dme"]:
        m = np.ones(len(y), bool) if ct == "all" else (sub[f"claim_type_{ct}"] == 1).to_numpy()
        lines.append(f"{ct:12s}{m.sum():7,d}{average_precision_score(y[m], p_local[m]):15.4f}"
                     f"{average_precision_score(y[m], p_ep[m]):18.4f}")
    verdict = "PASS" if diff.max() <= TOL else "FAIL"
    lines += ["", f"RESULT: {verdict}"]
    report = "\n".join(lines)
    print(report)
    (ROOT / "reports/endpoint_verification.txt").write_text(report + "\n")
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
