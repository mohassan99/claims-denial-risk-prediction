"""
Fit the Phase 2 XGBoost model from the Databricks gold tables and track it in
MLflow (Databricks-hosted), registering it in Unity Catalog.

Same model, same code path as src/fit_xgboost.py: the feature selection,
train-governs-val category handling, hyperparameters and early stopping are
imported from that module, not re-typed. The only difference is where the
rows come from (Delta tables gold_train_model / gold_val_model, sorted by
_split_row so XGBoost sees them in the parquet's order). The script then
proves it is the same model: the fitted booster and its metadata must
serialize byte-for-byte identically to reports/xgboost_model.json /
xgboost_model_meta.json when those files are present, and the val metrics
must match reports/xgboost_fit.json. Nothing is logged if any check fails.

What gets registered in Unity Catalog is a pyfunc wrapper
(pipeline/pyfunc_model.py) around deploy/score.py, so the registered model
takes raw claim rows exactly like the Azure ML endpoint and returns the same
numbers.

    python databricks/train_mlflow.py            # fit, verify, log, register
    python databricks/train_mlflow.py --no-register
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.common import REPO_ROOT, SCHEMA, CATALOG, get_spark, table  # noqa: E402
from run_pipeline import _load_dotenv  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXPERIMENT = "/Users/{user}/claims-denial-risk-prediction"
REGISTERED_MODEL = f"{CATALOG}.{SCHEMA}.claims_denial_xgboost"
CLAIM_TYPES = ("carrier", "outpatient", "dme")


def load_split(spark, name: str, columns: list[str] | None = None):
    df = spark.table(table(f"gold_{name}_model"))
    cols = columns or [c for c in df.columns if c != "_split_row"]
    pdf = df.select(*cols, "_split_row").toPandas()
    return pdf.sort_values("_split_row").drop(columns="_split_row").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    _load_dotenv()

    import mlflow
    from xgboost import XGBClassifier

    import fit_xgboost as fx
    from fit_baseline_model import _metrics_block

    spark = get_spark()
    user = spark.sql("select current_user()").collect()[0][0]
    all_cols = spark.table(table("gold_train_model")).columns
    columns = [c for c in all_cols if (c not in fx._EXCLUDE_COLS or c == fx.LABEL_COL) and c != "_split_row"]

    print("=== Loading gold_train_model ===")
    train = load_split(spark, "train", columns)
    y_train, X_train = fx._split_xy(train)
    del train
    cat_cols = fx._object_columns(X_train)
    X_train, categories_by_col = fx._to_train_categories(X_train, cat_cols)
    print(f"train: {X_train.shape[0]:,} x {X_train.shape[1]} ({len(cat_cols)} categorical)")

    print("=== Loading gold_val_model ===")
    val = load_split(spark, "val", columns + [f"claim_type_{ct}" for ct in CLAIM_TYPES if f"claim_type_{ct}" not in columns])
    raw_example = val[[c for c in columns if c != fx.LABEL_COL]].head(20)  # raw rows, for the signature
    y_val, X_val = fx._split_xy(val[columns])
    X_val = fx._apply_categories(X_val, cat_cols, categories_by_col).reindex(columns=list(X_train.columns))

    params = dict(n_estimators=500, max_depth=6, learning_rate=0.1, tree_method="hist",
                  enable_categorical=True, eval_metric="aucpr", early_stopping_rounds=30,
                  random_state=42, n_jobs=-1)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_val = model.predict_proba(X_val)[:, 1]
    y = y_val.to_numpy(dtype=np.float64)
    overall = _metrics_block(y, p_val)
    by_type = {ct: _metrics_block(y[(val[f"claim_type_{ct}"] == 1).to_numpy()],
                                  p_val[(val[f"claim_type_{ct}"] == 1).to_numpy()]) for ct in CLAIM_TYPES}
    print(f"best_iteration={model.best_iteration}  val PR-AUC={overall['pr_auc']:.4f}  ROC-AUC={overall['roc_auc']:.4f}")

    # --- Same model as Phase 2? ------------------------------------------
    checks = {}
    meta = {"colnames": list(X_train.columns), "cat_cols": cat_cols,
            "categories_by_col": {k: list(v) for k, v in categories_by_col.items()},
            "best_iteration": int(model.best_iteration)}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "xgboost_model.json"
        meta_path = Path(tmp) / "xgboost_model_meta.json"
        model.save_model(path)
        meta_path.write_text(json.dumps(meta, indent=2))  # same format as fit_xgboost.py
        new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for fname, fpath in (("xgboost_model.json", path), ("xgboost_model_meta.json", meta_path)):
            ref = REPO_ROOT / "reports" / fname
            if ref.exists():
                checks[f"identical_to_reports_{fname}"] = (
                    hashlib.sha256(fpath.read_bytes()).digest() == hashlib.sha256(ref.read_bytes()).digest())
        ref_fit = json.loads((REPO_ROOT / "reports" / "xgboost_fit.json").read_text())
        checks["best_iteration_matches"] = int(model.best_iteration) == ref_fit["best_iteration"]
        checks["val_metrics_match_1e-9"] = all(
            abs(overall[k] - ref_fit["val_metrics_overall"][k]) < 1e-9 for k in ("pr_auc", "roc_auc", "brier"))
        print("verification:", checks)
        if not all(checks.values()):
            raise SystemExit("Model fit from gold tables does NOT match Phase 2 -- not logging it.")

        # --- MLflow ----------------------------------------------------------
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(EXPERIMENT.format(user=user))
        with mlflow.start_run(run_name="xgboost-phase2-from-gold") as run:
            mlflow.set_tags({"source_tables": f"{table('gold_train_model')}, {table('gold_val_model')}",
                             "git_repo": "mohassan99/claims-denial-risk-prediction",
                             "booster_sha256": new_hash, "phase": "2 model, trained from Databricks gold"})
            mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"} | {
                "n_features": X_train.shape[1], "n_categorical": len(cat_cols), "train_rows": len(y_train)})
            mlflow.log_metric("best_iteration", int(model.best_iteration))
            mlflow.log_metrics({f"val_{k}": v for k, v in overall.items() if k != "n"})
            for ct, m in by_type.items():
                mlflow.log_metrics({f"val_{ct}_{k}": v for k, v in m.items() if k in ("pr_auc", "roc_auc", "brier")})
            mlflow.log_dict(checks, "verification.json")
            # Registered model = pyfunc wrapper around deploy/score.py, so it
            # takes raw claim rows exactly like the Azure ML endpoint.
            from pipeline.pyfunc_model import ClaimsDenialModel

            wrapper = ClaimsDenialModel()
            artifacts = {"xgboost_model": str(path), "xgboost_model_meta": str(meta_path)}
            # Build the signature's example output with the same transform the
            # wrapper will run (deploy/score.py), primed with this fit.
            sys.path.insert(0, str(REPO_ROOT / "deploy"))
            import score as _score

            _score._model = model
            _score._colnames, _score._cat_cols = meta["colnames"], meta["cat_cols"]
            _score._cat_dtypes = {c: pd.CategoricalDtype(categories=meta["categories_by_col"][c]) for c in cat_cols}
            Xr, unseen = _score.prepare_features(raw_example.to_dict(orient="records"))
            example_out = pd.DataFrame({"p_denied": model.predict_proba(Xr)[:, 1], "unseen_category_counts": unseen})
            info = mlflow.pyfunc.log_model(
                name="model", python_model=wrapper, artifacts=artifacts,
                code_paths=[str(REPO_ROOT / "deploy" / "score.py"), str(REPO_ROOT / "databricks" / "pipeline")],
                pip_requirements=["xgboost==3.2.0", "pandas", "numpy", "scikit-learn"],
                signature=mlflow.models.infer_signature(raw_example, example_out),
                input_example=raw_example.head(3),
                registered_model_name=None if args.no_register else REGISTERED_MODEL)
            print(f"MLflow run {run.info.run_id}; model {info.model_uri}"
                  + ("" if args.no_register else f"; registered as {REGISTERED_MODEL} v{info.registered_model_version}"))
    out = {"run_id": run.info.run_id, "experiment": EXPERIMENT.format(user=user), "checks": checks,
           "val_metrics_overall": overall, "val_metrics_by_claim_type": by_type,
           "registered_model": None if args.no_register else REGISTERED_MODEL,
           "registered_version": None if args.no_register else info.registered_model_version}
    (REPO_ROOT / "reports" / "databricks_mlflow_run.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
