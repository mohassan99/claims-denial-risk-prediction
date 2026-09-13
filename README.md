# Claims Denial Risk Prediction

Healthcare claims denial risk prediction — data wrangling, XGBoost/SHAP modeling, Azure ML deployment, and a SHAP-narrative + RAG + minimal agentic GenAI layer, built on the CMS Synthetic Medicare Claims PUF.

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
  while investigating an unrelated anomaly, not planned in advance. Final real-data denial rate:
  9.4%, within the 5-20% range typical of real payer data. Full reasoning, including corrected
  wrong turns, in `data/TARGET_DEFINITION.md`.
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
