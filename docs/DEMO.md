# Demo runbook: live Azure endpoint + Databricks pipeline

How to bring the project up for a demo or recording, show it, and shut it down so it stops
billing. Every command runs from the repo root in Git Bash (Windows) or any Linux/macOS shell.

## What costs money, and what doesn't

| Piece | Cost while idle | Notes |
|---|---|---|
| Azure ML **deployment** `blue` (1 x Standard_DS2_v2 VM) | about $0.15 per hour, about $3.60 per day | The only thing that bills meaningfully. **Stopped on 2026-09-27.** Start it for a demo, stop it after. |
| Azure ML endpoint `claims-denial-xgb`, registered model, workspace | pennies to a few dollars a month (storage, key vault, container registry) | Kept, so the deployment can come back with one command. |
| Databricks Free Edition | free | Serverless only; no cluster to leave running. |

To check your remaining Azure credit: Azure portal > **Cost Management** (or the Azure for Students
page, "Check your credit"). Optional: Cost Management > **Budgets** > add a budget with an email
alert at, say, $20.

## One-time setup on a machine

1. Azure CLI installed, plus its ML extension: `az extension add -n ml`.
2. `.env` at the repo root (copy `.env.example`), with `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
   and either:
   - a normal login: `az login --tenant <AZURE_TENANT_ID>` (works from your own computer), or
   - the service principal: `AZURE_CLIENT_ID` and `AZURE_CLIENT_SECRET` (needed from a cloud
     sandbox; the secret expires about 90 days after 2026-09-27). `deploy.sh` uses it
     automatically when there is no valid login.
3. For Databricks: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in `.env`, and a Python 3.12 venv
   (`python3.12 -m venv dbc-venv && dbc-venv/bin/pip install -r databricks/requirements.txt`).

## 1. Start the Azure endpoint (about 15 minutes, do this before the demo)

```bash
bash deploy/deploy.sh          # recreates the deployment from the registered model, routes 100% traffic, runs a smoke test
bash deploy/deploy.sh status   # expect: state Succeeded, traffic {"blue": 100}, deployment "blue" listed
```

`deploy.sh` needs no local model files. It deploys the latest registered model
(`claims-denial-xgboost` v2, content hash `11f409f72fc4c45e`, the time-based-split model). **But if
`reports/xgboost_model*.json` exist locally and differ, it registers them as a new version and
deploys that.** Model files from before 2026-09-27 (the old 89-tree model, hash
`1ff691017a5e9b61`) would silently put the old model back. So before step 1, either delete or
rename the two local files, or replace them with the current ones:

```bash
bash deploy/deploy.sh download-model   # pulls v2 into reports/ and prints its hash: expect 11f409f72fc4c45e
```

**Known quirk:** the first image build of a new environment version can fail with
`Identity(object id: ) does not have permissions for .../environments/read`. Run
`bash deploy/deploy.sh` again; the image finished building in the background and the second run
succeeds.

## 2. Show it

**Score sample claims live:**

```bash
bash deploy/deploy.sh test     # sends deploy/sample_request.json (6 real claim lines, 2 per claim type)
```

The response has one P(denied) per row plus `unseen_category_counts` (how many of that row's codes
the model never saw in training).

**Prove the endpoint matches the local model** (optional; needs `data/processed/val_model.parquet`
and `reports/xgboost_model*.json` locally):

```bash
python deploy/verify_endpoint.py   # 3,000 val rows: max difference ~5e-7, identical PR-AUC by claim type
```

**In the Azure portal:** ml.azure.com > workspace `mlw-claims-denial` > Endpoints >
`claims-denial-xgb` shows the deployment, its metrics and a **Test** tab (paste the JSON from
`deploy/sample_request.json`). Models > `claims-denial-xgboost` shows the registered versions.

**Databricks (no startup needed):**
- Workflows > `claims-denial-medallion-pipeline` > **Run now** runs bronze, silver, silver_label,
  gold and parity on serverless compute from the raw CSVs (a few minutes).
- Catalog > workspace > claims_denial: the bronze/silver/gold Delta tables, and Models >
  `claims_denial_xgboost` (v3, the model trained from the Delta tables).
- Experiments > `claims-denial-risk-prediction`: the MLflow run with metrics and the logged model.
- From a terminal instead: `dbc-venv/bin/python databricks/deploy_job.py --run`.

Talking points for each piece are in `docs/DATABRICKS.md` (section 8) and the README's Progress
section.

## 3. Stop the endpoint when done (stops the billing)

```bash
bash deploy/deploy.sh stop     # sets traffic to 0, deletes the deployment; endpoint, model, workspace kept
bash deploy/deploy.sh status   # expect: traffic {} and no deployments listed
```

Bringing it back later is step 1 again. `bash deploy/deploy.sh teardown` deletes the endpoint as
well (only if you are finished with Azure; step 1 then recreates it, which takes a few minutes
longer).
