# Claims Denial Risk Prediction — Claude Code instructions

Portfolio project: predict claim-denial risk on CMS Synthetic Medicare Claims (carrier,
outpatient, DME), with an engineered noisy-OR label (`is_denied`, 14.9% since the 2026-09-24 carrier fix; was 12.1%). Phases: 0 setup,
1 data/EDA (done), 2 baseline logistic + Chow test + XGBoost + SHAP (done 2026-09-25),
**3 Azure ML deploy (done 2026-09-25)**, **3b Databricks port (done 2026-09-27)**, **4 agentic claim explainer (done 2026-09-28)**, **5 report/video (next)**,
6 README/portfolio.

## How the user wants to work (read first)

- **You run, read, fix, and re-run yourself.** The user is not watching the terminal. Do not
  stop to ask them to paste output. Work the loop end to end, then report.
- **At the end of every session**, append a plain-language entry to `reports/SESSION_LOG.md`
  (create it if missing): what you ran, what the results mean, what you changed and why, what's
  next, and any decision you need from the user. Write it for someone reading after the fact who
  did not follow the code. Lead with results and decisions, not process.
- **Stop and ask only for real decisions**: anything that changes the target (`is_denied`), the
  shared `train_model.parquet`, the scope of the Chow test, or deletes/rewrites prior work. Put the
  question in `SESSION_LOG.md` and in chat with your recommendation and the tradeoff.
- Never suggest "start over" as a fix. Fix forward.

## Environment (Windows, Git Bash)

- Repo root: `D:\Project_Claims_Denial\claims-denial-risk-prediction`
- Activate venv: `source venv/Scripts/activate` (Windows layout: `Scripts`, not `bin`).
- Terminal Python: use heredocs (`python << 'EOF' ... EOF`), never `python -c "..."` (breaks on
  pasted leading whitespace).
- **Machine has 8 GB RAM and has hit memory ceilings repeatedly.** Full design matrices are ~3 GB
  each. Rules:
  - Test with `--sample-frac 0.02` first, then `0.2`, then full.
  - Full Chow fits run staged: `--restricted-only`, then `--skip-restricted` (same flags both times).
  - Don't run `python src/build_chow_design_matrix.py` directly on the full file (its `__main__`
    holds both matrices at once). Use `fit_chow_test.py` / `check_rank.py` instead.
  - Never hold both design matrices in memory at once; `del` + `gc.collect()` between.
- Dependencies are pinned in `requirements.txt` (includes `statsmodels==0.15.0`,
  `firthmodels>=0.8.2`). `firthlogist` is dead (pulled from PyPI) — do not use it.

## Azure (Phase 3, set up 2026-09-25)

- Subscription "Azure for Students"; tenant/subscription IDs, RG, workspace and location live in
  `.env` (gitignored; `.env.example` has the keys). RG `rg-claims-denial`, workspace
  `mlw-claims-denial`, region **`canadacentral`** — the subscription's policy allows only
  swedencentral/mexicocentral/francecentral/denmarkeast/canadacentral, and Azure ML isn't in
  mexicocentral. Quota: 4 vCPU per VM family, 6 per region.
- Endpoint `claims-denial-xgb`, deployment `blue` (1 x Standard_DS2_v2), model
  `claims-denial-xgboost` in the workspace registry (tag `content_sha256_16`).
- **All Azure operations go through `deploy/deploy.sh`** (idempotent: `status`, `test`,
  `download-model`, `stop`, `teardown`, or no arg = create whatever is missing). Don't run ad hoc
  `az ml` create/delete commands; if something new is needed, add it to the script.
- **The registry is the durable copy of the model.** In a fresh session run
  `deploy/deploy.sh download-model` rather than refitting.
- New cloud session setup (the sandbox resets between tasks):
  `python3 -m venv ~/azure-cli-venv && ~/azure-cli-venv/bin/pip install azure-cli`,
  `az extension add -n ml`, `az login --use-device-code --tenant <AZURE_TENANT_ID>` (the user
  approves the code), recreate `.env` from the IDs in CLAUDE memory / the user, and export
  `AZ=~/azure-cli-venv/bin/az` for `deploy.sh`. If any full-data step is needed, add swap first
  (`fallocate -l 12G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`):
  the sandbox has ~8 GB, and loading `combined_claims_raw.parquet` alone peaks at 5.7 GB.
- **Auth (2026-09-27):** device-code login from the cloud sandbox is now blocked by Entra security
  defaults (AADSTS530035, "deemed unsafe"). Use the service principal `claims-denial-deployer`
  (Contributor on `rg-claims-denial` ONLY; `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` in `.env`,
  secret expires ~90 days from 2026-09-27). `deploy.sh` logs in with it automatically when the
  current login is expired, and fails fast instead of trying to create the resource group.
- **The deployment bills (~$0.15/h) while it exists, called or not.** `deploy/deploy.sh stop`
  removes only the deployment; `deploy/deploy.sh` restores it in ~15 min.
- Known quirk: the first build of a new environment version fails with
  `Identity(object id: ) does not have permissions for .../environments/read`; re-running
  `deploy.sh` succeeds (image finished building in the background). See SESSION_LOG 2026-09-25
  Phase 3 entry. Test `score.py` in a clean venv built only from `deploy/conda.yml`'s packages
  before any redeploy — the dev venv hid a missing scikit-learn once.

## Databricks (Phase 3b, 2026-09-26)

- **Databricks Free Edition**, not Azure Databricks: the Azure for Students subscription refuses trial
  workspaces for its offer type, has no usable VM size + quota for Databricks in any allowed region,
  and can't request quota increases (all tested 2026-09-26; see SESSION_LOG). Workspace
  `https://dbc-f8758ffb-8443.cloud.databricks.com` (us-east-2, serverless only). `.env` holds
  `DATABRICKS_HOST` / `DATABRICKS_TOKEN` (a personal access token the user created; gitignored).
- Unity Catalog: catalog `workspace`, schema `claims_denial`; volumes `raw` (the 3 CMS CSVs),
  `reference` (pandas outputs used only by the parity check), `reports`.
- Code in `databricks/`: `pipeline/` (bronze_silver_claims, silver_label, gold, parity_check,
  pyfunc_model, common), `run_pipeline.py` (entry point, local or Job), `train_mlflow.py`,
  `deploy_job.py` (publishes code to `/Workspace/Users/<me>/claims-denial-pipeline`, creates/resets
  Job `claims-denial-medallion-pipeline`). Client venv: Python 3.12 + `databricks/requirements.txt`
  (Connect must match serverless's Python minor version).
- **Rules:** business logic stays in `src/` (imported, never re-typed). The label's randomness is
  numpy PCG64 keyed by `_row_id` -- reproduce the draws, never re-draw with a Spark RNG (that would
  change `is_denied`). Any change to `src/` must keep `run_pipeline.py --steps parity` passing
  (re-upload the pandas outputs with `--upload-reference` first).
- Databricks serverless python tasks exec() the entry file without `__file__`: the Job passes
  `--repo-root`.
- MLflow experiment `/Users/<me>/claims-denial-risk-prediction`; UC model
  `workspace.claims_denial.claims_denial_xgboost` (pyfunc wrapper over `deploy/score.py`, takes raw rows;
  v3 = time-based-split model).

## Git

- On your own machine: you commit and push from the local repo. **Always `git pull origin main`
  before pushing.**
- From a cloud session that can't `git push` (proxy 403 — "not in this session's authorized
  repository set"): the GitHub MCP connector (`mcp__Github__*`) has write access to this repo
  independent of that proxy. Push each commit with `mcp__Github__push_files`, one call per local
  commit, reusing its exact message (`git show -s --format='%B' <sha>`) so history stays granular.
  Read each changed file's content straight from git (`git show <sha>:<path>` → a temp file → the
  `Read` tool) and pass that through unmodified — do not retype file content into the tool call by
  hand; a 2026-09-25 session did this and introduced a whitespace bug in one file. After pushing,
  `git fetch origin main && git merge --ff-only origin/main` to bring the session's own local repo
  back in sync (pushing via the API does not update git's local tracking refs). The old
  `git bundle` handoff is a fallback only if the GitHub MCP connector isn't available in session.
  A file too large to comfortably read into a tool call (tens of MB) should not be pushed this way
  at all -- gitignore it if it's regenerable (see `fit_xgboost.py`'s model artifacts, 2026-09-25),
  or ask if it truly needs to be in git. **The "don't retype" rule bit again the same session it
  was written down**: a large JSON file's content was reproduced from what had been read earlier
  rather than passed through as an exact string, and it picked up a stray extra key. Caught within
  the same turn by immediately re-reading the pushed commit and diffing against the local file, and
  fixed with a follow-up commit. Lesson reinforced: after any push whose content wasn't a
  mechanical file-content pass-through (i.e. anything typed or reconstructed by hand, even
  large/structured text), re-fetch and diff the pushed version against the local source before
  moving on, not just after ones that felt risky.
- **Preferred cloud handoff (user's choice, 2026-09-27):** pushing through `push_files` re-sends
  every changed file's full text (the 2026-09-27 session pushed ~250 KB that way), which is slow and
  costly. Instead, commit locally in the sandbox, export with `git format-patch origin/main --stdout
  > /mnt/user-data/outputs/<name>.patch`, send it to the user, and let them run `git pull origin
  main`, `git am <name>.patch`, `git push origin main` on their machine. Use `push_files` only for
  a small change or when the user can't reach their machine.
- Commit incrementally, one verified change per commit, with messages that explain *why*.
- After each phase step, append to README's Progress section (append only, never rewrite it).
- Never commit secrets; `.env` stays gitignored. No coursework references anywhere in the repo.

## Engineering standards (hard-won, don't relearn)

- **`0` is a value, `NaN` is the absence of one.** Never use one to represent the other, for any
  field type. "Present" means non-null; a 0 is a present value.
- Verify every claim against real data before asserting it. This project has repeatedly found its
  own docs wrong by checking (NPI flags, presence counts, df formula, rank). Say "not yet
  verified" when it isn't.
- Build new DataFrame columns into a dict, then one `pd.concat()`. Never assign columns one at a
  time in a loop (fragmentation → memory crashes).
- `.copy()` only when a function mutates its input in place.
- Binary 0/1 columns are `int8`, not `int64`.
- Never build an interaction term that is constant within a claim type (all 0 or all 1).
- `train_model.parquet` is shared with XGBoost: baseline-only exclusions/zero-fills happen in
  `build_chow_design_matrix.py` / `fit_chow_test.py`, never upstream in `build_features.py`,
  unless the user approves.
- Before writing code against a third-party package, verify its actual installed API
  (`inspect.signature`, docstring). Before relying on a package name, confirm it exists on PyPI.
- Test new design-matrix logic on synthetic data with planted cases before running on real data.
- **Audit every rule, feature and check per claim type, never only overall** (added 2026-09-24).
  An overall rate can look right while one claim type is silently exempt: that is how
  `provider_outlier`/`duplicate_claim` never fired on carrier for two weeks. Concretely:
  - before any `groupby` on ID/key columns, confirm each key is populated in every claim type
    (`denial_rules.require_key_coverage`);
  - a new risk factor must get a `denial_reasons.RISK_FACTOR_EXPECTATION` row for every claim
    type ("fires" or "zero: <verified reason>"), or the label build fails;
  - any large cell with an exact 0% or 100% rate (claim type × code, etc.) must be explained in
    writing before moving on (`reports/label_audit.txt` lists them every build).
- A gap between claim types that nobody can explain (e.g. carrier 5.6% vs outpatient 24%) is a
  finding to chase down, not background.
- Any cached fit/intermediate must be keyed to the exact label it was built from (the Chow scripts
  fingerprint the label vector); never reuse a cache across a label rebuild.

## Documentation conventions

- `data/FEATURE_ENGINEERING.md` is the source of truth for every feature decision. Section 6's
  2026-09-23/24 addendum covers the rank-deficiency work. **Append corrections; never edit away
  wrong earlier claims** — the visible correction trail is deliberate (interview prep).
- `data/TARGET_DEFINITION.md` covers the label; `data/data_dictionary.md` is a column reference.
- Read the relevant doc section before re-deriving anything.

## Current state (as of 2026-09-27)

**Split fixed (user's decision, 2026-09-27):** `build_target_and_split.py` and
`databricks/pipeline/gold.py` sort by the PARSED `CLM_FROM_DT` (stable). Train 2015-01-08 to
2020-10-29 (14.4% denied), val to 2021-11-23 (17.8%), test to 2023-03-02 (14.2%). Label unchanged.
Everything refit on it. NEW val numbers (supersede the Phase 2 numbers further down, which were on
the old day-of-month split):
- Chow Stage 1: Firth LR 2,527.4 on 158 df, rejected. Stage 2: same 5 interact / 6 pool.
- Baseline (Firth): overall PR-AUC 0.671 / ROC 0.817; carrier 0.141/0.616; outpatient 0.877/0.920;
  DME 0.264/0.708.
- XGBoost (73 trees): overall 0.691/0.849; carrier 0.191/0.710; outpatient 0.878/0.923; DME
  0.266/0.716.
- Overall metrics rose because val's period has more outpatient lines (35.5% vs 30.6%) and more
  deprecated consult codes (24.9% vs 15.1% of outpatient lines): a claim-mix shift over time, not a
  better model. Carrier/DME flat.
- SHAP provider-ID check: PRVDR_NUM +0.198, CARR_CLM_BLG_NPI_NUM +0.154, ORG_NPI_NUM +0.022,
  PRF_PHYSN_UPIN +0.196 (was -0.150; memorization signature gone), TAX_NUM -0.149 (still negative,
  still minor). HCPCS_CD still dominates (mean |SHAP| 0.870).
- Azure: model v2 registered, deployment `blue` updated, `verify_endpoint.py` PASS (max diff 5e-7).
- Databricks: full Job run from the raw CSVs SUCCESS, parity PASS on every table
  (`reports/databricks_parity.json`); UC model v3 registered, byte-identical to the pandas fit.
- The user's local `combined_claims_raw.parquet` predates `_row_id`; regenerate with `load_data.py`.

**Phase 3b (Databricks port): done.** Study guide `docs/DATABRICKS.md`. Serverless Jobs forbid JVM
access: `reduce(DataFrame.unionByName, ...)` failed there (use `lambda a, b: a.unionByName(b)`).

### Phase 3 state


**Phase 3 (Azure ML deploy): done.** Model rebuilt from scratch in the cloud sandbox first and
reproduced Phase 2 byte-for-byte (label audit, 89 trees, val metrics). Live endpoint verified
against the local model on 3,000 val rows: max |diff| 5e-7, identical PR-AUC per claim type
(`reports/endpoint_verification.txt`). Scoring (`deploy/score.py`) = `fit_xgboost._apply_categories`
exactly (train's category lists; unseen → NaN; per-row `unseen_category_counts` in the response).
Details, failures and fixes: SESSION_LOG.md's Phase 3 entry.

### Phase 2 state (unchanged)

**Label:** fixed 2026-09-24. `provider_outlier` / `duplicate_claim` now key on the billing NPI for
carrier claims (`denial_rules.build_provider_key`). Rate 14.9% overall (carrier 10.0%, outpatient
24.1%, DME 16.2%), `CALIBRATION_SCALE` still 1.4. `build_target_and_split.py` writes
`reports/label_audit.txt` and fails if any risk factor x claim type cell breaks
`RISK_FACTOR_EXPECTATION`. See TARGET_DEFINITION.md's 2026-09-24 addendum.

Key scripts in `src/`:
- `build_chow_design_matrix.py` — restricted/unrestricted design matrices (rank fixes A–D), plus
  (2026-09-25) `build_mixed_design_matrix()` (the baseline model's 5-interact/6-pool design) and
  optional `encoders=`/`return_encoders=` on `build_chow_design_matrix()` so a categorical
  encoding (top-n HCPCS/dx/state categories, provider frequency map, which columns were dropped as
  claim-type-determined) fit on train can be reused unchanged on val/test — needed the first time
  this project encoded two splits into the same feature space.
- `chunked_logit.py` — memory-bounded Newton-Raphson logistic (ordinary or Firth), with
  coefficients optionally held at 0 under the full model's penalty. Validated against
  firthmodels and statsmodels. Peak ~4.9 GB at full size.
- `fit_chow_test.py` — Stage 1. Reparameterizes the unrestricted design as [restricted, Z] and
  tests Z = 0 (Firth = Heinze-Schemper penalized LR). Enforces full rank, span equality, df
  agreement (4 ways), convergence, and a label fingerprint on the cached restricted fit.
- `fit_chow_stage2.py` — per-variable penalized LR tests, Holm-adjusted, resumable.
- `fit_baseline_model.py` (2026-09-25) — fits the Phase 2 baseline mixed model on train, evaluates
  on val (PR-AUC/ROC-AUC/Brier, per claim type). Builds train's array, fits, and frees it BEFORE
  building val's matrix — the first full run without this ordering was OOM-killed holding train's
  and val's matrices alive together (see SESSION_LOG.md's 2026-09-25 entry).
- `fit_xgboost.py` (2026-09-25) — Phase 2 XGBoost, same train/val split and metrics as the
  baseline. Confirmed (not assumed) that none of `build_chow_design_matrix.py`'s machinery is
  needed: top-n/frequency encoding, the rank-deficiency fixes, and the `__x__claim_type`
  interaction terms are all linear-model-only concerns. Categoricals go in at full cardinality via
  pandas `category` dtype + `enable_categorical=True`; NaN is passed through natively. Gains the
  ~130 `fit_chow_test._PENDING_*` columns the baseline couldn't use (no cardinality decision needed
  for a tree). Train's category list per column is fit once and reused on val, same
  train-governs-val discipline as the baseline's encoders. Also saves the fitted booster
  (`reports/xgboost_model.json`, ~21.8 MB) and its column/category metadata
  (`reports/xgboost_model_meta.json`, ~605 KB) so downstream scripts can load-and-verify instead of
  refitting. Both are gitignored (too large for this session's read-then-push-via-API discipline)
  but fully deterministic (`random_state=42`) from `train_model.parquet` — rerun this script to
  regenerate them if missing.
- `fit_shap.py` (2026-09-25) — loads the saved XGBoost model + metadata (never refits), re-verifies
  its predictions reproduce `reports/xgboost_fit.json`'s reported val metrics to 1e-6 before doing
  anything else, then runs `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")`
  on a 20,000-row label-stratified sample of val. Validated the explainer against XGBoost's native
  categorical splits/missing values on synthetic data first (reconstruction error ~1.67e-6) before
  running on real data. Reports mean |SHAP| importance overall and per claim type, and answers the
  provider-ID question quantitatively: for each of the 5 flagged ID columns, correlates each
  category's mean |SHAP| with log(its train claim count) — positive/flat says bigger claim history
  gets at least as much weight (consistent with a real signal); negative says low-count categories
  carry more weight (consistent with memorizing small-sample noise). See results below.
- statsmodels' lbfgs is NOT used for fitting anymore: it never left beta = 0 on these matrices.

**Chow test results on the corrected label (full train set):** Firth LR 2,642.1 on 158 df, H0
rejected. Standard MLE agrees within 0.1% but is formally non-convergent because of one claim (dx
N186 x carrier). Stage 2: interact provider_state, HCPCS_CD, PRNCPAL_DGNS_CD, prvdr_num_freq,
CARR_CLM_CASH_DDCTBL_APLD_AMT; pool the other 6. `--exclude-separating-codes` is obsolete (no
separation left). Details in FEATURE_ENGINEERING.md Section 6's last addendum.

**Baseline mixed model results (2026-09-25, val set):** overall PR-AUC 0.557, ROC-AUC 0.772
(14.9% positive rate). Per claim type: outpatient 0.815/0.893 (deprecated_code is near-
deterministic there), DME 0.262/0.706, carrier 0.141/0.612 (its risk factors are the
weakest-grounded ones). Calibration tight in every claim type. Fit with Firth (standard MLE stalls
on a rare-category near-separation; not investigated further since Firth converges cleanly and
fast). Full table in SESSION_LOG.md and `reports/baseline_model_results__firth.txt`.

**XGBoost results (2026-09-25, val set, 89 trees, early-stopped):** overall PR-AUC 0.583, ROC-AUC
0.816 — beats the baseline everywhere, most on carrier (PR-AUC 0.188 vs 0.141, ROC-AUC 0.706 vs
0.612), its weakest claim type either way. Outpatient 0.820/0.903, DME 0.260/0.715. Confirmed
XGBoost needs none of the baseline's encoding (see `fit_xgboost.py`'s docstring); it also gets to
use ~130 columns the baseline couldn't. Top features: HCPCS_CD by a wide margin, then several raw
high-cardinality provider-identifier columns (ORG_NPI_NUM, TAX_NUM, CARR_CLM_BLG_NPI_NUM,
PRF_PHYSN_UPIN, PRVDR_NUM) — plausibly a legitimate "this provider is denied often" signal (the
baseline's `prvdr_num_freq` carried the same signal, just pre-aggregated), but worth confirming
with SHAP rather than assuming, since raw ID splitting can also memorize small-sample provider
noise. `LINE_PRMRY_ALOWD_CHRG_AMT` ranks #2; checked and it is NOT new leakage — it's an exact
DME-only duplicate of `LINE_ALOWD_CHRG_AMT` (already in the baseline's pooled features,
`build_chow_design_matrix.py`'s fix B), so it adds no information beyond what the baseline already
used, just weighted more heavily by the tree. Full table in `reports/xgboost_results.txt`.

**SHAP results (2026-09-25, 20,000-row label-stratified sample of val):** loaded model reproduced
the reported val metrics to 1e-6 before explaining; SHAP reconstruction error ~5.2e-6. By mean
|SHAP| the ranking differs from gain-based importance in one notable way: `HCPCS_CD` dominates
even more clearly (0.781, next is `LINE_PLACE_OF_SRVC_CD` at 0.166), and **`LINE_PRMRY_ALOWD_CHRG_AMT`
— gain-based importance's #2 feature (0.199) — does not even make SHAP's overall top 20**, only
showing up at #10 within DME specifically (0.028). Read together with the leakage check already
done, this is not a leakage retraction — it confirms gain can overweight a feature that wins a few
very effective splits without moving most individual predictions much, while SHAP reflects average
per-prediction impact; the two metrics are answering different questions and this project's
top-line "important features" claim should cite SHAP, not gain, going forward.

The provider-ID question (task queue item 4) has a mixed, not a single, answer — reported per
column rather than resolved overall, per this project's per-claim-type auditing standard extended
here to per-feature:
- `PRVDR_NUM` (corr +0.234): high-claim-count providers get *more* SHAP weight than low-claim-count
  ones (0.274 vs 0.102 mean |SHAP|) — the clearest case of a genuine, volume-supported signal, and
  it corroborates the baseline model's large `prvdr_num_freq` coefficient (same underlying signal,
  pre-aggregated there).
- `CARR_CLM_BLG_NPI_NUM` (corr +0.065) and `ORG_NPI_NUM` (corr +0.017): essentially flat — no
  evidence either way of memorization, no strong evidence of a volume-driven signal either.
- `TAX_NUM` (corr −0.186) and `PRF_PHYSN_UPIN` (corr −0.150): negative — low-claim-count categories
  carry more SHAP weight, the signature the memorization concern predicted. Both are minor overall
  contributors (mean |SHAP| 0.068 and 0.009 respectively), so this is a real but small-magnitude
  finding, not a reason to distrust the model's headline numbers.
Net: the flagged provider-ID features are not uniformly one thing or the other; `PRVDR_NUM` looks
safe and informative, `TAX_NUM`/`PRF_PHYSN_UPIN` show a real but minor memorization signature worth
naming in the writeup rather than acting on (no evidence it's driving the reported metrics — it's a
small fraction of total importance). Full tables in `reports/shap_results.txt` /
`reports/shap_fit.json`.

### Phase 4 state (2026-09-28)

**Phase 4 (agentic claim explainer): done.** Code in `src/agent/` (tools, agent loop, grounding check,
eval runner), guide in `docs/PHASE4.md`, all transcripts in `reports/phase4/runs/` (re-gradable offline).
- Grounding on 30 val claims (Sonnet 5.5): v1 prompt 97.5% (355/364 checkable tokens, 2 empty answers);
  after precomputing numbers in the tools 99.3%; prompt v2 100% (540/540), and 100% (542/542) on 30
  held-out claims. Grounded = traceable to a tool output, not correct. Failures and fixes (model
  arithmetic, invented code meanings, over-stated policy, empty answers, and bugs in my own grader) are
  in `docs/PHASE4.md`.
- Claims come from Databricks `gold_val_model` via `scripts/make_val_sample.py` (writes
  `data/processed/val_sample_v2.parquet`). The older `val_sample.parquet` is stale (pre-split-fix).
- The Azure endpoint stayed stopped; `score_claim` runs `deploy/score.py` locally on model v2.
- API spend $2.21 (`reports/phase4/cost_ledger.json`); runner has a `--budget` cap.
- Tools need no API key (`python -m src.agent.tools ...`); only live agent runs do. Tests:
  `python -m pytest tests/test_agent.py`.
- Local env: `.env` (gitignored) holds every secret; `bash scripts/make_env.sh` builds it; the full list of
  gitignored items and how to rebuild them is `docs/LOCAL_SETUP.md`. The Claude desktop bridge can read
  `.env` but refuses to write it.
- **Sourced definitions added 2026-09-29** (`data/agent/code_reference.json`, section "Adding sourced definitions"
  in `docs/PHASE4.md`): field meanings (ResDAC), place of service (CMS), HCPCS Level II and ICD-10-CM (NLM
  serving the CMS code sets), joined into `score_claim` / `explain_shap` output with a source per entry. Same 30
  held-out claims, prompt v2 unchanged: grounding 511/511, features with a meaning 68/180 to 178/180, code values
  with a meaning 0/74 to 56/74. CPT codes (AMA licensed) are deliberately left without descriptions; 5 items
  whose lookups were rate limited (a 6th, A4604, is covered by a CMS policy article) are listed in the reference under `not_verified_left_out` and a test keeps them out
  until verified. Rule kept: never add an entry that was not checked against a source in the session.
- Open: CPT descriptions, the 5 unverified items and rare diagnosis codes still get an honest decline.

## Task queue (do in order; log each in SESSION_LOG.md)

1. ~~Build the Phase 2 baseline mixed model~~ — done 2026-09-25 (`fit_baseline_model.py`).
2. ~~Baseline metrics on val~~ — done 2026-09-25, see above.
3. ~~XGBoost~~ — done 2026-09-25 (`fit_xgboost.py`), see above.
4. ~~SHAP~~ — done 2026-09-25 (`fit_shap.py`), see above. Provider-ID question answered per
   column, not with a single verdict: `PRVDR_NUM` looks like a real signal, `TAX_NUM`/
   `PRF_PHYSN_UPIN` show a minor memorization signature, `CARR_CLM_BLG_NPI_NUM`/`ORG_NPI_NUM` are
   ambiguous.
5. ~~Phase 3: Azure ML deploy~~ — done 2026-09-25 (`deploy/`), see Azure section above.
6. ~~Phase 3b: Databricks port~~ -- done 2026-09-27; split fixed and everything refit.
7. ~~Phase 4: agentic claim explainer~~ -- done 2026-09-28, see 'Phase 4 state' above and `docs/PHASE4.md`. Scope (revised 2026-09-28, user's decision): Replaces the
   earlier "SHAP narrative + RAG chunking comparison" plan, which duplicated the user's separate RAG
   project. Claim explanation is structured lookup (SHAP values, feature meanings, CARC codes are
   fetched by exact key), so RAG is used only where it fits: policy questions.
   - **Build:** an agent on the Anthropic API using tool use (the model decides which tool to call;
     key is `ANTHROPIC_API_KEY` in `.env`). Four tools:
     1. `score_claim`: run the LOCAL XGBoost model on a claim row (the Azure endpoint stays stopped;
        do not restart it unless the user asks).
     2. `explain_shap`: top SHAP contributions for that claim, joined to feature meanings from
        `data/data_dictionary.md`.
     3. `lookup_carc`: CARC (Claim Adjustment Reason Code) to short description for the codes this
        project uses. X12 copyrights the official list, so use paraphrases and link to
        https://x12.org/codes/claim-adjustment-reason-codes. 11: diagnosis doesn't support the
        procedure billed. 16: missing information or billing error. 18: duplicate of a claim or
        service already submitted. 181: procedure code wasn't valid on the date of service. 197:
        required prior authorization wasn't obtained.
     4. `search_policy`: small RAG (retrieval-augmented generation: fetch relevant text chunks, hand
        them to the model) over the policy documents behind the six risk rules (CMS consultation-code
        rule, Claims Processing Manual Ch. 12 Sec. 30.6.10; LCD L33718 + article A52467 for CPAP; CMS
        DMEPOS prior-authorization program documents; others cited in `data/TARGET_DEFINITION.md`).
        Basic chunking only; no chunking comparison study.
   - **Evaluate grounding:** on about 30 val claims, the share of numbers and codes in each
     explanation that trace back to tool outputs. Document at least one real failure and its fix.
   - **Cut rule:** if time runs short, drop `search_policy` and keep tools 1 to 3.
   - **Model:** get it with `bash deploy/deploy.sh download-model` (expect content hash
     `11f409f72fc4c45e`). Never refit.
   - Everything gitignored is rebuildable locally: see `docs/LOCAL_SETUP.md`. `bash
     scripts/make_env.sh` creates `.env` with hidden prompts for secrets.

## Open decisions for the user (don't act on these alone)

- **Keep the endpoint deployment running?** It costs ~$0.15/h (~the whole student credit per
  month). Recommended: keep until the demo/video is recorded, then `deploy/deploy.sh stop`.

- **CALIBRATION_SCALE**: kept at 1.4 (rate 14.9%, inside the 10–15% target) so the fix changed
  carrier labels only. Retuning to ~1.0 (~12.4%) would also move every outpatient/DME label.
- **Minimum support for interaction terms**: dx N186 x carrier has one claim. A rule like "skip
  interaction terms with < N nonzero rows" would make standard MLE converge, but changes the
  Chow test's tested set and df.
- 3 fixed-constant fields (`CARR_CLM_PMT_DNL_CD`, `CLM_DISP_CD`, `CLM_MDCR_NON_PMT_RSN_CD`):
  move to `build_features.py` `DROP_COLUMNS`?
- ~130 raw columns excluded from the baseline pending an encoding decision (see `_PENDING_*` in
  `fit_chow_test.py`); secondary diagnosis/procedure codes are the one group likely worth encoding.
