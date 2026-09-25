# Claims Denial Risk Prediction

**Predicting which Medicare claims are likely to be denied — and why — built on the real CMS
Synthetic Medicare Claims PUF, an engineered denial-risk label grounded in documented adjudication
logic, and (in progress) XGBoost + SHAP explainability with an Azure ML deployment.**

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

**Status: Phase 1 of 6 complete.** Phase 2 (baseline logistic regression — including a formal Chow
test to decide, per shared feature, whether claim-type effects should be pooled or interacted —
then XGBoost + SHAP) is in progress.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

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
