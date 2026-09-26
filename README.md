# Claims Denial Risk Prediction

**Predicting which Medicare claims are likely to be denied — and why — built on the real CMS
Synthetic Medicare Claims PUF, an engineered denial-risk label grounded in documented adjudication
logic, XGBoost + SHAP explainability, and a live Azure ML deployment.**

Public CMS claims data has no real claim-level denial-outcome field anywhere in its release —
that's proprietary payer adjudication data CMS doesn't publish. This project uses the PUF's real
claim structure, procedure/diagnosis codes, and billing amounts for realism, then engineers a
defensible, probabilistically-calibrated denial-risk label from documented adjudication rules
(CMS payment-policy transmittals, CARC/RARC codes, published payer denial-rate benchmarks) rather
than assuming a shortcut exists. Every non-obvious design decision — including the wrong turns and
corrections along the way — is kept in full in `data/TARGET_DEFINITION.md` and
`data/FEATURE_ENGINEERING.md`, not cleaned up to look like the right answer was obvious from the
start.

Built to bring the same "own the metric" analytical discipline behind 10+ years of payer analytics
work (HEDIS/STARS gap closure, risk adjustment) to a full ML build: data engineering → target
construction → modeling → deployment.

**Status: Phases 1–3 complete.** Data + engineered label (Phase 1), a Chow-tested baseline logistic
model and an XGBoost model with SHAP explainability (Phase 2), and the XGBoost model deployed as a
live, key-authenticated Azure ML managed online endpoint whose predictions are verified to match
the local model (Phase 3). Next: a GenAI layer on top of the endpoint (Phase 4).

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### Azure ML deployment (Phase 3)

Everything Azure-side is scripted in `deploy/` and is idempotent: re-running it skips whatever
already exists, so it's safe from a fresh clone, a new machine, or a new session.

One-time, per machine: install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli),
then

```bash
az extension add -n ml
cp .env.example .env          # fill in AZURE_TENANT_ID / AZURE_SUBSCRIPTION_ID
az login --tenant <AZURE_TENANT_ID>
```

Then, from the repo root:

```bash
bash deploy/deploy.sh status           # what's deployed right now
bash deploy/deploy.sh test             # score deploy/sample_request.json against the live endpoint
bash deploy/deploy.sh download-model   # pull the registered model into reports/ (no refit needed)
python deploy/verify_endpoint.py  # live endpoint vs. local model on 3,000 val rows
bash deploy/deploy.sh                  # (re)create anything missing: RG, workspace, model, endpoint, deployment
bash deploy/deploy.sh stop             # delete the deployment -> stops VM billing (endpoint + model kept)
```

The registered model in the Azure ML workspace is the durable copy of the fitted model; the local
`reports/xgboost_model*.json` files are gitignored and can always be re-pulled with
`download-model` (or regenerated with `python src/fit_xgboost.py`).

## Progress

### Phase 1 — Data Sourcing, Wrangling & EDA (complete)

Downloaded the CMS Synthetic Medicare Claims PUF (carrier, outpatient, DME — ~89% of claim volume
per the CMS user guide) and built the full pipeline from raw claims to a modeling-ready feature
set:

- **Target (`is_denied`):** no public CMS claims PUF contains a real denial outcome field —
  verified directly against the CMS user guide before writing any code. Engineered a probabilistic
  (noisy-OR) label from 5 documented risk factors instead of a hard rule, so the label carries
  graded risk rather than being a deterministic function of the same fields used to model it. One
  factor (`deprecated_code`, Medicare-non-payable consultation codes) was discovered mid-build
  while investigating an unrelated anomaly, not planned in advance. Full reasoning, including
  corrected wrong turns, in `data/TARGET_DEFINITION.md`.
- **Leakage inventory:** caught and fixed a real leakage bug (`CLM_PMT_AMT` is both a natural
  feature and, structurally, a consequence of denial) before it reached modeling.
- **EDA:** 5 required figures plus 5 supplementary ones (class imbalance, categorical cardinality,
  numeric feature distributions, risk-factor co-occurrence, risk-factor lift) answering not just
  the checklist but "what does Phase 2 need to know."
- **Feature engineering:** identifier re-typing, a real dtype bug fix generalized to close a whole
  category of future bugs, a data-quality fix for a dual-coded state field, structural-missingness
  handling for claim-type-specific fields, and `claim_type` derivation with the specific
  cell-means encoding Phase 2's planned interaction terms require. Full reasoning, including two
  corrected verification mistakes, in `data/FEATURE_ENGINEERING.md`.
- **Data dictionary:** `data/data_dictionary.md`.

Deliverables: `src/load_data.py`, `src/denial_rules.py`, `src/denial_reasons.py`,
`src/build_target_and_split.py`, `src/run_eda.py`, `src/build_features.py`.

*Note, 2026-09-22: a 6th risk factor (`missing_hcpcs`) was discovered during Phase 2 feature-audit
work and folded back into the label — the earlier 9.4% denial rate cited here at Phase 1's close
recalibrated to 12.1% as a result. Kept as a visible correction rather than silently editing the
number above — see `data/TARGET_DEFINITION.md`'s addenda for the full trail.*

### Phase 2 — Baseline Logistic Regression (Chow Test) + XGBoost/SHAP (in progress)

Building the design matrix for a formal Chow test — deciding, per shared covariate, whether its
effect on denial risk should be pooled across claim types (carrier/outpatient/DME) or given a
separate coefficient per claim type, rather than assuming either answer. A complete claim-type
presence audit (`data/full_claim_type_presence_audit.csv`) replaced an earlier, incomplete
column-by-column classification once cross-checks caught it missing real cases — including several
covariates that looked shared but turned out to be empirically claim-type-specific once actual
per-category variance was checked, not just null rates. Full reasoning in
`data/FEATURE_ENGINEERING.md` Section 6. Fitting code and results not yet built.

*Progress, 2026-09-24:* first valid Chow-test result. On the full train set (1.15M claims),
Firth's penalized likelihood-ratio test rejects pooled coefficients: **LR = 4,732.5 on 158 df**.
df is cross-checked four independent ways, and both design matrices are verified full rank.
Per-variable follow-up tests (Holm-adjusted) show 6 of 11 shared variables need claim-type-specific
effects: HCPCS, principal diagnosis, provider frequency, state, service count, and carrier cash
deductible. The other 5 can be pooled.

Getting there meant catching two silent failures in the earlier fitting code, now fixed and kept
on the record:
- The optimizer never left its starting point. Every earlier log-likelihood equals n·ln(0.5)
  exactly.
- Two separately-penalized Firth fits don't form Firth's likelihood-ratio test.

The fix is a memory-bounded Newton-Raphson solver (`src/chunked_logit.py`), validated against
`firthmodels` and `statsmodels`. It puts the test in "full model, tested coefficients = 0" form.
The work also surfaced that 34 of 37 carrier procedure codes have zero denials. The likely cause
is two label rules that key on a provider field that is empty for every carrier claim. That is an
open label decision, not yet changed. Full trail in `data/FEATURE_ENGINEERING.md` Section 6.

*Progress, 2026-09-24 (later the same day): label fix.* Chasing the Chow test's separation led to
a real bug in the engineered label. Two risk rules (outlier provider, duplicate claim) identified
providers by a field that is empty on every professional (carrier) claim, so they could never
fire on 62% of the data. Carrier claims now use the billing NPI.
- Carrier's denial rate went from 5.6% to 10.0%, and the overall rate from 12.1% to 14.9%.
  Outpatient and DME labels are unchanged claim for claim.
- The separation disappeared.
- The Chow test still rejects pooling: Firth LR = 2,642 on 158 df, with standard MLE agreeing
  within 0.1%. 5 of 11 shared variables need claim-type-specific effects.

Because the overall denial rate had looked right, the label build now also fails if any rule
silently stops firing for a claim type, or if a grouping key is missing for one. See
`data/TARGET_DEFINITION.md` (2026-09-24 addendum) and `reports/label_audit.txt`.

*Progress, 2026-09-25: Phase 2 baseline mixed model.* Built and evaluated the model the Chow test
selected: 5 shared variables (state, HCPCS, principal diagnosis, provider claim frequency, carrier
cash deductible) kept as claim-type interaction terms, the other 6 pooled to one coefficient each.
Fit with `chunked_logit`'s Firth-penalized MLE (`src/fit_baseline_model.py`) on the full train set
(1,151,951 rows), evaluated on val, never only overall:

| | PR-AUC | ROC-AUC | positive rate |
|---|---|---|---|
| Overall | 0.557 | 0.772 | 14.9% |
| Carrier | 0.141 | 0.612 | 10.1% |
| Outpatient | 0.815 | 0.893 | 24.0% |
| DME | 0.262 | 0.706 | 16.0% |

Calibration is tight in every claim type (predicted mean within 0.2 points of the actual rate).
Outpatient is by far the easiest claim type to predict, because its `deprecated_code` risk rule is
close to deterministic; carrier is the hardest, consistent with its risk factors being the
weakest-grounded ones in `denial_reasons.py`. Full results in
`reports/baseline_model_results__firth.txt`.

*Progress, 2026-09-25: XGBoost.* Fit `src/fit_xgboost.py` on the same train/val split, this time
confirming (not assuming) what a tree-based model actually needs: none of the baseline's top-n/
frequency encoding, rank-deficiency fixes, or claim-type interaction terms, since those all exist
to work around linear-model limitations trees don't share. Categorical features go in at full
cardinality (pandas `category` dtype, xgboost's native categorical splits) and missing values are
passed through natively rather than zero-filled — and the model gets to use roughly 130 columns
the baseline couldn't (secondary diagnosis/procedure codes, legacy provider IDs, other CMS
categorical codes) because a tree needs no cardinality-reduction decision for them first.

| | PR-AUC | ROC-AUC | positive rate |
|---|---|---|---|
| Overall | 0.583 | 0.816 | 14.9% |
| Carrier | 0.188 | 0.706 | 10.1% |
| Outpatient | 0.820 | 0.903 | 24.0% |
| DME | 0.260 | 0.715 | 16.0% |

Beats the baseline logistic model in every claim type, most on carrier — still the hardest claim
type for either model, but XGBoost closes a meaningful share of the gap. The single most
important feature by a wide margin is `HCPCS_CD`; several raw provider-identifier columns
(billing/organization NPI, tax number, referring-physician UPIN) also rank highly, a plausible
"this provider is denied more often" signal in the same spirit as the baseline's `prvdr_num_freq`
— flagged for a closer look with SHAP rather than assumed, since raw high-cardinality ID splits
can also memorize individual providers' small-sample noise. Full results in
`reports/xgboost_results.txt`.

*Progress, 2026-09-25: SHAP (Phase 2 complete).* Explained the XGBoost model with
`shap.TreeExplainer` (`src/fit_shap.py`) on a 20,000-row label-stratified sample of val, loading
the already-fitted model rather than refitting and re-verifying it reproduced the reported val
metrics to 1e-6 first. Two findings:

1. **Gain-based importance and SHAP importance disagree for one feature.** `LINE_PRMRY_ALOWD_CHRG_AMT`
   is XGBoost's #2 feature by gain (0.199) but doesn't crack SHAP's overall top 20 at all (only
   reaching #10 within DME, 0.028). This isn't new leakage — it was already checked and confirmed
   to be an exact DME-only duplicate of a feature the baseline model already used — but it is a
   real methodological finding: gain can overweight a feature that wins a few very effective
   splits without moving most individual predictions much, while SHAP reflects actual average
   per-prediction impact. `HCPCS_CD` dominates even more clearly under SHAP than under gain (mean
   |SHAP| 0.781, next closest 0.166).

2. **The provider-ID question has a mixed answer, not a single verdict.** For each of the 5
   flagged raw provider-identifier columns, correlated each category's mean |SHAP| with
   log(its training claim count) — positive means higher-volume providers get at least as much
   weight (a real, volume-supported signal); negative means low-volume categories carry more
   weight (the signature of memorizing small-sample noise):

   | Column | corr(log count, \|SHAP\|) | Read |
   |---|---|---|
   | `PRVDR_NUM` | +0.234 | Genuine signal — high-volume providers get *more* weight (0.274 vs 0.102), corroborating the baseline's large `prvdr_num_freq` coefficient |
   | `CARR_CLM_BLG_NPI_NUM` | +0.065 | Flat — no evidence either way |
   | `ORG_NPI_NUM` | +0.017 | Flat — no evidence either way |
   | `PRF_PHYSN_UPIN` | −0.150 | Minor memorization signature (small overall contributor, mean \|SHAP\| 0.009) |
   | `TAX_NUM` | −0.186 | Minor memorization signature (mean \|SHAP\| 0.068) |

   Net: not every raw provider-ID feature is doing the same thing. `PRVDR_NUM` looks safe and
   informative; `TAX_NUM` and `PRF_PHYSN_UPIN` show a real but small-magnitude memorization
   signature that's worth naming rather than acting on, since neither is a major contributor to
   the reported metrics.

Full tables in `reports/shap_results.txt` / `reports/shap_fit.json`. This closes out Phase 2. Next:
Phase 3 (Azure ML deploy).

### Phase 3 — Azure ML Deployment (complete)

The Phase 2 XGBoost model is live as an Azure ML **managed online endpoint** (`claims-denial-xgb`,
Canada Central): an HTTPS REST API, key-authenticated, that takes raw claim rows as JSON and
returns P(denied) per row.

- **Verified, not assumed.** `deploy/verify_endpoint.py` sends 3,000 validation rows to the live
  endpoint and compares each prediction with the local model's: max difference 5×10⁻⁷ (the
  endpoint rounds to 6 decimals), identical PR-AUC overall and per claim type
  (`reports/endpoint_verification.txt`). Before deploying, the model artifacts were regenerated
  from scratch and reproduced Phase 2's reported metrics byte-for-byte.
- **Same preprocessing as training.** The scoring script applies exactly the transform the model
  was evaluated with: each categorical column is cast using the training set's category list, so a
  code never seen in training is treated as missing rather than silently mapped to the wrong
  category. Each response also reports how many of a row's categorical values were unseen, so a
  caller can tell a confident low-risk score from one made on unfamiliar inputs.
- **Model registry as the source of truth.** The model is registered in the Azure ML workspace with
  a content hash tag; the deploy script only registers a new version when the model files change.
- **Real constraints, documented.** The Azure for Students subscription only allows five regions
  (not the usual `eastus`), and its 4-vCPU-per-family quota — with Azure's 20% reserve for rolling
  upgrades — caps the deployment at a single 2-vCPU `Standard_DS2_v2` instance. Three deploy-time
  failures (a first-build permission check, a stale managed identity from a too-fast
  delete/recreate, and a missing `scikit-learn` in the inference image) are written up in
  `reports/SESSION_LOG.md` along with what now prevents each.

Infrastructure is fully scripted and idempotent (`deploy/deploy.sh`); see Setup above.
