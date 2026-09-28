# Databricks in this project: a study guide

This guide explains how the claims denial pipeline runs on Databricks, why it is built the way it is,
and how each piece maps back to the pandas code in `src/`. It is written to be studied and to answer
interview questions, so every term is defined the first time it appears.

## 1. Vocabulary

| Term | What it means here |
|---|---|
| **Databricks** | A managed platform for running Apache Spark, storing data as Delta tables, and tracking ML work with MLflow. This project uses the **Free Edition** workspace. |
| **Apache Spark / PySpark** | A distributed data engine. PySpark is its Python API. Code describes transformations (filters, joins, groupbys, windows); Spark plans and runs them across many machines, so a dataset never has to fit in one machine's memory. |
| **Serverless compute** | Databricks-managed compute that starts on demand. There is no cluster to size or pay for while idle. Free Edition is serverless only. |
| **Databricks Connect** | A client library that lets Python on a laptop (or a cloud sandbox) send Spark work to Databricks compute. `DatabricksSession.builder.serverless(True)` is the entry point. The client's Python minor version must match serverless (3.12). |
| **Delta Lake / Delta table** | Parquet files plus a transaction log. It adds ACID writes, schema enforcement and table history ("time travel") on top of plain files. Every table in this project is a Delta table. |
| **Unity Catalog (UC)** | Databricks' governance layer. Everything is addressed as `catalog.schema.object`. Here: catalog `workspace`, schema `claims_denial`. UC also holds **volumes** (governed folders for raw files) and **registered models**. |
| **Volume** | A UC-governed folder for non-tabular files, addressed as `/Volumes/<catalog>/<schema>/<volume>/...`. This project has `raw` (the CMS CSVs), `reference` (pandas outputs, used only by the parity check) and `reports`. |
| **Medallion architecture** | A layering convention: **bronze** = raw data exactly as received, **silver** = cleaned, typed, joined, business logic applied, **gold** = consumption-ready tables for a specific use (here, model training). Each layer can be rebuilt from the one before it. |
| **MLflow** | An open-source ML lifecycle tool. **Tracking** records runs (parameters, metrics, artifacts). The **model registry** (here, in Unity Catalog) versions models. A **pyfunc** is MLflow's generic "any Python model" wrapper. |
| **Databricks Job** | A scheduled or on-demand workflow of **tasks** with dependencies. Here each medallion step is one task, running `databricks/run_pipeline.py --steps <step>` on serverless compute. |

## 2. What was built

```
raw CSVs (volume: raw)
   |
   v
bronze_claims_carrier / _outpatient / _dme     every field as the original text + file position
   |
   v
silver_claims                                  3 files unioned, typed, one global _row_id
   |
   v
silver_risk_factors                            the 6 risk-factor rules + noisy-OR P(denied)
   |
   v
silver_claims_labeled                          + is_denied, CARC reason codes
   |
   +--> gold_labeled_claims_for_eda            label-construction columns (never a model input)
   |
   v
gold_train_model / gold_val_model / gold_test_model   time-based split, model-ready features
   |
   v
MLflow run + UC model workspace.claims_denial.claims_denial_xgboost
```

| Databricks code | pandas counterpart | What it does |
|---|---|---|
| `databricks/pipeline/bronze_silver_claims.py` | `src/load_data.py` | reads the pipe-delimited files; lands bronze; unions and types silver |
| `databricks/pipeline/silver_label.py` | `src/denial_reasons.py`, `src/denial_rules.py`, `src/build_target_and_split.py` (label part) | risk factors, P(denied), is_denied, reason codes, label audit |
| `databricks/pipeline/gold.py` | `src/build_target_and_split.py` (split part), `src/build_features.py` | time-based split, feature engineering |
| `databricks/pipeline/parity_check.py` | (none) | compares every Delta table with the pandas parquet output, cell by cell |
| `databricks/train_mlflow.py` | `src/fit_xgboost.py` | fits XGBoost from gold, proves it is the same model, logs to MLflow, registers in UC |
| `databricks/pipeline/pyfunc_model.py` | `deploy/score.py` | lets the UC model take raw claim rows, sharing Azure's preprocessing code |
| `databricks/deploy_job.py` | (none) | publishes the code to the workspace and creates or updates the Job |

## 3. Design decisions worth being able to explain

**One source of truth for business logic.** The Spark modules import every rule parameter and feature
decision from `src/` (base probabilities, CARC codes, the HCPCS-to-diagnosis table, provider-key
fields, `DROP_COLUMNS`, the state crosswalk). Spark only re-expresses the *mechanics*. If a rule
changes in `src/`, both pipelines change together, and the parity check proves they still agree.

**Reproduce the random draws; never re-draw.** The label is a Bernoulli draw from numpy's PCG64
generator (seed 42) in row order, followed by the 30% second-reason draws and a seed-43 payment-noise
draw. Spark cannot reproduce that exact random stream, and drawing again with a Spark random-number
generator would silently change `is_denied` for thousands of claims. So the draws are generated by
the same numpy calls on the Spark driver (one float per row, a few seconds) and joined back on
`_row_id`. Everything else runs distributed.

**Replicate pandas' quirks on purpose.** Matching pandas exactly required naming behaviors pandas
applies silently:
- `read_csv` turns strings like `"NA"`, `"NULL"`, `"nan"` into missing values, even in text columns.
- Type inference: a column is numeric only if every value parses; it is a 64-bit integer only if it
  also has no missing values.
- The prior-authorization lookback uses a stable sort, so ties on date keep original row order.
- `sum()` over all-missing values is 0 in pandas and null in Spark.
- Percentiles: pandas interpolates linearly; Spark's exact `percentile()` does too, while
  `percentile_approx()` would not match.

**Row order is part of correctness.** Gold tables carry `_split_row`, each row's position inside its
split. Tree models can depend on row order through histogram binning, so `train_mlflow.py` sorts by it.
That is why the model trained from Delta tables is byte-identical to the pandas-trained model, not
merely close.

**Verification is a first-class step.** `parity_check.py` checks column names and order, column types,
the exact set of rows, every cell (null-safe, doubles exact), and row order. A negative control with
five planted differences in a copy of a table confirmed the check catches all of them, so a PASS
means something.

## 4. Results

| Check | Result |
|---|---|
| bronze vs the raw CSV text (DME, cell by cell) | identical; an empty field lands as null |
| `silver_claims` vs `combined_claims_raw.parquet` (1,799,924 rows) | every cell equal |
| label audit (Spark) vs `reports/label_audit.txt` | byte-identical |
| `gold_*_model` and `gold_labeled_claims_for_eda` vs the pandas parquet files | every cell equal, same row order |
| XGBoost fit from gold vs `reports/xgboost_model.json` | byte-identical model file and metadata |
| UC-registered model vs the Azure endpoint | same scores on the sample request |

The full run is the Databricks Job `claims-denial-medallion-pipeline`, five tasks from raw CSV to
the parity check. The latest run results are copied into `reports/databricks_parity.json` and
`reports/label_audit_databricks.txt`.

## 5. The bug the port found

Porting forced an exact statement of what every pandas line does. The split code sorted by
`CLM_FROM_DT`, which is text like `28-Sep-2015`. Text sorts character by character, so the "time-based"
split was really a **day-of-month** split: train was days 1-20 of every month, val days 20-25, test days
25-31, each spanning 2015 to 2023. It was fixed in both pipelines (sort by the parsed date) and every
model was refit. Details: `data/TARGET_DEFINITION.md`, addendum of 2026-09-27.

## 6. Running it

**From the Databricks UI:** Workflows (Jobs) → `claims-denial-medallion-pipeline` → Run now. Tables
appear under Catalog → workspace → claims_denial. The MLflow experiment is under Experiments →
`claims-denial-risk-prediction`; the model is under Catalog → Models.

**From a terminal** (Python 3.12, `.env` with `DATABRICKS_HOST` and `DATABRICKS_TOKEN`):

```bash
python3.12 -m venv dbc-venv && dbc-venv/bin/pip install -r databricks/requirements.txt
dbc-venv/bin/python databricks/run_pipeline.py --upload-reference   # after any change to src/
dbc-venv/bin/python databricks/run_pipeline.py                      # bronze -> gold via Databricks Connect
dbc-venv/bin/python databricks/run_pipeline.py --steps parity
dbc-venv/bin/python databricks/train_mlflow.py
dbc-venv/bin/python databricks/deploy_job.py --run                  # publish code, update the Job, run it
```

## 7. Gotchas hit along the way

- **Azure Databricks is not available on an Azure for Students subscription.** The free trial tier
  is refused for that offer type, no usable VM size has quota in any allowed region, and quota
  increases are refused. All five were tested directly; Free Edition was the working path.
- **Serverless Jobs run the entry script without `__file__`.** The Job passes `--repo-root` instead.
- **Serverless forbids direct JVM access.** `reduce(DataFrame.unionByName, frames)` failed inside the
  Job (the unbound classic method touches `_jdf`) but worked through Databricks Connect.
  `reduce(lambda a, b: a.unionByName(b), frames)` works in both.
- **A plain `mlflow.xgboost` model rejects raw strings** for categorical columns. The registered model
  is a pyfunc wrapper around `deploy/score.py`, so it applies the same category mapping as the Azure
  endpoint.
- **Workspace paths can collide.** The MLflow experiment took the name
  `/Users/<me>/claims-denial-risk-prediction`, so the code lives in `/Users/<me>/claims-denial-pipeline`.

## 8. Interview talking points

- Why a medallion design: each layer is rebuildable from the one before it, bronze keeps the raw
  evidence, and consumers read gold without touching business logic.
- Why parity instead of "it looks right": the goal was the *same* dataset at scale, so the proof
  had to be exact, including a negative control.
- Why keep pandas at all: it is the readable reference implementation and the thing parity is
  measured against; Spark is the scalable execution of the same logic.
- What the port surfaced: a mislabeled split. Porting is a form of code review.
