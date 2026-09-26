"""
Phase 3 -- Azure ML managed online endpoint scoring script for the Phase 2
XGBoost claims-denial model.

The endpoint runs this file inside the Azure ML inference server
(azureml-inference-server-http): init() once per worker at container start,
run() once per request.

WHAT THE MODEL EXPECTS. The registered model folder holds the two files
src/fit_xgboost.py writes:
  - xgboost_model.json       the fitted XGBClassifier (early-stopped; the
                             best_iteration attribute is saved inside it, so
                             predict_proba() uses the same 89 trees the
                             reported val metrics came from)
  - xgboost_model_meta.json  column order, which columns are categorical,
                             and TRAIN's exact category list per categorical
                             column
Requests carry raw claim rows with the same column names as
data/processed/*_model.parquet. This file applies exactly the transform
fit_xgboost.py applies to val (_apply_categories): every categorical column
is cast using TRAIN's category list, so a value the model never saw in
training becomes NaN (xgboost's native missing branch) rather than a new
integer code that would silently map to the wrong category. Columns absent
from a request are NaN, same as a structurally-absent claim-type field in
the training data (e.g. a carrier-only field on an outpatient row). Extra
columns are ignored. The label (is_denied), IDs and raw dates are never
features, so they are ignored if sent.

REQUEST:
    {"input_data": [ {"HCPCS_CD": "99213", "claim_type_carrier": 1, ...}, ... ]}
RESPONSE:
    {"model": "claims-denial-xgboost", "n": 2,
     "p_denied": [0.071, 0.412], "unseen_category_counts": [0, 1]}
unseen_category_counts reports, per row, how many categorical values were
outside train's category list and were therefore treated as missing -- so a
caller can tell a genuine low-risk prediction from one made on inputs the
model has never seen.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODEL_NAME = "claims-denial-xgboost"
MAX_ROWS_PER_REQUEST = 5_000

_model: XGBClassifier | None = None
_colnames: list[str] = []
_cat_cols: list[str] = []
_cat_dtypes: dict[str, pd.CategoricalDtype] = {}


def _find_model_dir() -> Path:
    """AZUREML_MODEL_DIR is the mount point of the registered model; the
    registered folder sits one level below it. Search rather than hard-code
    the folder name so the same file works for a local test too."""
    root = Path(os.environ.get("AZUREML_MODEL_DIR", Path(__file__).resolve().parents[1] / "reports"))
    hits = list(root.rglob("xgboost_model.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one xgboost_model.json under {root}, found {len(hits)}")
    return hits[0].parent


def init() -> None:
    global _model, _colnames, _cat_cols, _cat_dtypes
    model_dir = _find_model_dir()
    meta = json.loads((model_dir / "xgboost_model_meta.json").read_text())
    _colnames = meta["colnames"]
    _cat_cols = meta["cat_cols"]
    _cat_dtypes = {c: pd.CategoricalDtype(categories=meta["categories_by_col"][c]) for c in _cat_cols}

    _model = XGBClassifier()
    _model.load_model(model_dir / "xgboost_model.json")
    if _model.best_iteration != meta["best_iteration"]:
        raise RuntimeError(
            f"loaded model best_iteration {_model.best_iteration} != metadata {meta['best_iteration']} -- "
            "model and metadata files are from different fits"
        )
    logging.info("loaded %s from %s: %d features (%d categorical), best_iteration=%d",
                 MODEL_NAME, model_dir, len(_colnames), len(_cat_cols), _model.best_iteration)


def prepare_features(records: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw request rows -> the exact feature frame the model was fit on.
    Returns (X, unseen_counts)."""
    raw = pd.DataFrame.from_records(records)
    X = raw.reindex(columns=_colnames)  # missing columns -> NaN, extras dropped, train's order
    unseen = np.zeros(len(X), dtype=np.int64)
    num_cols = [c for c in _colnames if c not in _cat_dtypes]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="raise").astype("float64")
    for col, dtype in _cat_dtypes.items():
        s = X[col].astype("object").where(X[col].notna(), None)
        s = s.map(lambda v: v if v is None else str(v))
        known = s.isin(dtype.categories)
        unseen += (s.notna() & ~known).to_numpy()
        X[col] = s.where(known, other=np.nan).astype(dtype)
    return X, unseen


def run(raw_data: str) -> dict:
    try:
        payload = json.loads(raw_data)
        records = payload["input_data"] if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
            raise ValueError('body must be {"input_data": [ {column: value, ...}, ... ]} with at least one row')
        if len(records) > MAX_ROWS_PER_REQUEST:
            raise ValueError(f"at most {MAX_ROWS_PER_REQUEST} rows per request, got {len(records)}")
        X, unseen = prepare_features(records)
        p = _model.predict_proba(X)[:, 1]
        return {
            "model": MODEL_NAME,
            "n": int(len(p)),
            "p_denied": [round(float(v), 6) for v in p],
            "unseen_category_counts": unseen.tolist(),
        }
    except Exception as exc:  # the inference server turns a raise into an opaque 500
        logging.exception("scoring failed")
        return {"error": f"{type(exc).__name__}: {exc}"}
