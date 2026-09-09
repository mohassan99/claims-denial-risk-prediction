# Data Dictionary (starter — extend once you've opened the real files)

Source: CMS Synthetic Medicare Claims PUF, `carrier.csv` / `outpatient.csv` (start here per
Table 3-1 claim volume: carrier 59%, outpatient 30% of all claims). Field names follow the
Medicare RIF / CCW "long name" convention documented in the CMS user guide, Section 3.4.

Run `df.columns.tolist()` on your actual downloaded file first — CMS has occasionally renamed
fields between releases, and the synthetic PUF may not populate every column the real RIF
codebook defines. Update this table to match what you actually see before treating it as final.

| Column | Meaning | Source file(s) | Notes |
|---|---|---|---|
| `BENE_ID` | Unique synthetic beneficiary identifier | all | Join key across all claim + beneficiary files |
| `CLM_ID` | Unique claim identifier | all claim files | |
| `CLM_FROM_DT` | Claim service start date | all claim files | Used for time-based split + denial-over-time EDA |
| `CLM_THRU_DT` | Claim service end date | all claim files | Not yet used in this pipeline; add for length-of-stay features |
| `HCPCS_CD` | Procedure/service code billed | carrier, outpatient, DME | Leading-zero risk — kept as `str` in `load_data.py` |
| `PRNCPAL_DGNS_CD` / `ICD_DGNS_CD1` | Principal diagnosis (ICD-10-CM) | carrier / outpatient resp. | Field name differs by claim type — check both |
| `PRVDR_NUM` | Billing provider identifier | all claim files | Used as provider-outlier grouping key; real specialty field TBD once file is inspected |
| `CLM_PMT_AMT` | Amount Medicare paid on the claim | all claim files | Core signal for `rule_zero_payment` — verify this is the right payment field for your claim type (carrier files sometimes carry payment at the line level instead) |
| `is_denied` | **Engineered target**, not a native CMS field | constructed (`denial_reasons.py`) | See `data/TARGET_DEFINITION.md` — probabilistically sampled, not a hard rule; do not treat as ground truth |
| `risk_*` | Per-factor risk flags feeding the noisy-OR label | constructed | **Drop before modeling** — these define the target, so including them as features is target leakage by construction |
| `p_denied_model` | The noisy-OR probability used to sample `is_denied` | constructed | **Drop before modeling** — this literally is the label's generating probability |
| `denial_reason_carc_1` / `_2` | Sampled CARC reason code(s) for denied claims | constructed | **Drop from the modeling split**, but keep a `BENE_ID`/`CLM_ID`-joinable copy — genuinely useful for the Phase 4 SHAP-narrative demo and the Phase 5 "top denial reasons" chart |
| `CLM_PMT_AMT` | Amount Medicare paid | all claim files, **overwritten** by `apply_payment_consequence()` | **Drop before modeling** — denied claims get this zeroed out for internal consistency, making it a near-deterministic consequence of the label, not an independent feature |

## Still to document once you've opened the real files
- Real provider-specialty field name (used as `PRVDR_SPCLTY` or similar in standard CCW claims —
  confirm exact name/values in your download; `run_eda.py`'s `denial_rate_by_provider_specialty`
  currently groups on `PRVDR_NUM` as a placeholder)
- Whether `CLM_PMT_AMT` exists at claim level for carrier files or only at line level
  (`LINE_NCH_PMT_AMT` in the standard CCW codebook) — carrier claims can have multiple line items
- Full column list per claim type, once downloaded (`inpatient.csv`, `dme.csv`, `snf.csv`,
  `hospice.csv`, `hha.csv`, `pde.csv` each have their own schemas per Table 3-3 of the CMS user
  guide)
