# Data Dictionary

Reflects the final Phase 1 pipeline as built, against the real downloaded CMS Synthetic Medicare
Claims PUF (carrier + outpatient + DME claim files, 2015-2023). For full reasoning behind any
decision below, see `data/TARGET_DEFINITION.md` (the label) and `data/FEATURE_ENGINEERING.md`
(the features) — this file is a concise column-by-column reference, not a restatement of that
reasoning.

## Target

| Column | Meaning | Source |
|---|---|---|
| `is_denied` | **Engineered target**, not a native CMS field. Probabilistically sampled via noisy-OR across 5 risk factors — not a hard rule, not ground truth. | Constructed (`denial_reasons.py`); see `TARGET_DEFINITION.md` |

## Label-construction columns — dropped from the modeling feature set (target leakage)

These columns exist only in `combined_claims_raw.parquet` and `labeled_claims_for_eda.parquet`,
never in `train`/`val`/`test.parquet` or `*_model.parquet` — they define or are a direct
consequence of `is_denied`, so including them as features would be leakage by construction.

| Column | Meaning | Why dropped |
|---|---|---|
| `risk_deprecated_code`, `risk_duplicate_claim`, `risk_missing_prior_auth`, `risk_dx_procedure_mismatch`, `risk_provider_outlier` | Per-factor boolean flags feeding the noisy-OR label | These *are* the label's inputs |
| `p_denied_model` | The noisy-OR probability used to sample `is_denied` | Literally the label's generating probability |
| `denial_reason_carc_1` / `_2` | Sampled CARC reason code(s) for denied claims | Dropped from modeling, but kept in `labeled_claims_for_eda.parquet` — used by the Phase 4 SHAP-narrative demo and the Phase 5 "top denial reasons" chart |
| `CLM_PMT_AMT` (native) | Amount Medicare paid | Overwritten to near-$0 for denied claims by `apply_payment_consequence()` for internal consistency — a near-deterministic *consequence* of the label, not an independent feature |

## Engineered features — this project's derived columns (`src/build_features.py`)

| Column | Meaning | Source |
|---|---|---|
| `claim_type_carrier`, `claim_type_outpatient`, `claim_type_dme` | Cell-means one-hot encoding (all 3 dummies, no dropped reference) of which RIF file a claim came from | Derived from `_source_file`; see `FEATURE_ENGINEERING.md` Section 3 for why cell-means specifically (needed as interaction terms for claim-type-specific fields, not just a plain categorical) |
| `provider_state` | Billing provider's state, normalized to 2-letter USPS form | Derived from `PRVDR_STATE_CD` via the official SSA-numeric-to-alpha crosswalk — the raw field mixed alpha and numeric codes for the same states (e.g. `CA` and `05` both meaning California); see `FEATURE_ENGINEERING.md` Section 1. Verified clean: 51 unique values (50 states + DC), no leftover unmapped codes, no territories present in this download. |
| `has_referring_physician`, `has_performing_physician`, `has_attending_physician`, `has_operating_physician`, `has_rendering_physician` | Binary flags: was this NPI role field populated on the claim | Derived from `RFR_PHYSN_NPI`/`PRF_PHYSN_NPI`/`AT_PHYSN_NPI`/`OP_PHYSN_NPI`/`RNDRNG_PHYSN_NPI` presence — the raw NPI *values* are dropped (redundant with `PRVDR_NUM`, synthetic and unresolvable against real NPPES records), but *whether a role was populated at all* carries real signal (e.g. a populated referring-physician field plausibly indicates a referral-based encounter) |

## Key native CMS fields retained as features

| Column | Meaning | Source |
|---|---|---|
| `BENE_ID` | Unique synthetic beneficiary identifier | All files — join key |
| `CLM_ID` | Unique claim identifier | All claim files |
| `CLM_FROM_DT` | Claim service start date | All claim files — drives the time-based train/val/test split and the denial-over-time EDA figure |
| `HCPCS_CD` | Procedure/service code billed | Carrier, outpatient, DME — kept as `str` (leading-zero risk); 144 unique values in this data, top 20 cover 93.4% — concentrated enough for one-hot + "other" bucket |
| `PRNCPAL_DGNS_CD` | Principal diagnosis (ICD-10-CM) | Carrier, DME — 277 unique values, top 20 cover 79.2%, same one-hot treatment as `HCPCS_CD` |
| `PRVDR_NUM` | Billing provider identifier | All claim files — used as `provider_outlier`/`dx_procedure_mismatch`'s grouping key; 8,460 unique values, top 20 cover only 4.8% (genuine long tail) — needs frequency or target encoding for Phase 2, one-hot would explode the feature space |

## Dropped columns

| Column(s) | Why | Documented in |
|---|---|---|
| `RFR_PHYSN_NPI`, `PRF_PHYSN_NPI`, `AT_PHYSN_NPI`, `OP_PHYSN_NPI`, `RNDRNG_PHYSN_NPI`, `PRVDR_NPI` | Raw NPI values dropped (redundant with `PRVDR_NUM`); presence captured separately, see above | `FEATURE_ENGINEERING.md` Section 1 |
| `PRVDR_ZIP` | Superseded by `provider_state` (correct field for MAC-jurisdiction reasoning; ZIP3 would have been a noisier, redundant proxy) | `FEATURE_ENGINEERING.md` Section 1 |
| `PRVDR_STATE_CD` | Superseded by `provider_state` after crosswalk normalization | `FEATURE_ENGINEERING.md` Section 1 |
| `_source_file` | Superseded by `claim_type_*` dummies | `FEATURE_ENGINEERING.md` Section 3 |
| `LINE_SERVICE_DEDUCTIBLE`, `FI_CLM_PROC_DT`, `OT_PHYSN_UPIN`, `OT_PHYSN_NPI`, `ICD_PRCDR_CD25`, `PRCDR_DT25`, `RSN_VISIT_CD1/2/3`, `REV_CNTR_NDC_QTY` | 100% null across the entire dataset, regardless of claim type — carry zero information | `FEATURE_ENGINEERING.md` Section 2 |
| `PRVDR_SPCLTY` | Zero-variance (constant `01`/General Practice across every populated row, carrier + DME) — confirmed a genuine Synthea realism limitation, not a null field; `01` is a real, meaningful code, just never varied | `FEATURE_ENGINEERING.md` Section 2 |
| `ICD_DGNS_VRSN_CD1`-`CD12` | Zero-variance (constant `0`/ICD-10 across every populated row) — confirmed *expected*, not a limitation: this dataset (2015-2023) falls almost entirely after the real Oct-2015 ICD-9→ICD-10 transition | `FEATURE_ENGINEERING.md` Section 2 |

## Remaining native numeric fields (~166 columns)

The bulk of the remaining columns in `train_model.parquet` are claim-type-specific dollar/count/
rate fields native to the CMS RIF schema — e.g. `REV_CNTR_TOT_CHRG_AMT` (outpatient-only),
`DMERC_LINE_MTUS_CNT` (DME-only), `NCH_CARR_CLM_ALOWD_AMT` (carrier-only). Their partial nullness
(null for every claim outside their applicable claim type) is **structural, not a data-quality
issue** — see `FEATURE_ENGINEERING.md` Section 2 for the full explanation and why these are kept
rather than dropped or imputed, and Section 3 for how Phase 2's baseline logistic regression plans
to handle them (an interaction construction against the `claim_type_*` dummies, gated by a
Chow-test-driven decision about which fields genuinely need claim-type-specific coefficients
versus which can be shared). Full official field-by-field definitions are available via the CMS
RIF/CCW documentation (ResDAC's variable lookup, `resdac.org/cms-data/variables`) — not
individually re-documented here, since XGBoost consumes them natively regardless of encoding and
the baseline's exact handling is still an open Phase 2 decision rather than a settled mapping.