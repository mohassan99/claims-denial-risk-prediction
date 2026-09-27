"""
MLflow pyfunc wrapper so the Unity Catalog-registered model accepts RAW claim
rows, exactly like the Azure ML endpoint does.

A plain mlflow.xgboost model expects a DataFrame whose categorical columns are
already pandas `category` dtype with train's category list -- a caller sending
raw strings gets an error. This wrapper reuses deploy/score.py (the Azure ML
scoring script) unchanged, so there is exactly one implementation of the
raw-row -> feature transform, shared by both serving paths.
"""

from __future__ import annotations

import os

import mlflow.pyfunc
import pandas as pd


class ClaimsDenialModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        os.environ["AZUREML_MODEL_DIR"] = os.path.dirname(context.artifacts["xgboost_model"])
        import score  # deploy/score.py, shipped via code_paths

        score.init()
        self._score = score

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        records = model_input.to_dict(orient="records")
        X, unseen = self._score.prepare_features(records)
        p = self._score._model.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_denied": p, "unseen_category_counts": unseen})
