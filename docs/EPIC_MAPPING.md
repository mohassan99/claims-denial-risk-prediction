# Epic mapping (provider-side view)

**What this is.** A mapping from this project's claim fields and built denial reasons to the places they would sit in a hospital's
Epic billing data. It is built from Epic's public table documentation (the electronic health information export schema at
https://open.epic.com/EHITables). It is **not** hands-on work with Epic, and this project has never been run on Epic data.

**One caveat.** The public schema documents the Epic-released tables used for patient data export. An organization's version, its
custom values and its reporting database (Clarity) can differ. This is not the full Clarity data dictionary.

**Date verified:** October 2, 2026. Every table page below was read on that date. Source pattern:
`https://open.epic.com/EHITables/GetTable/<TABLE>.htm`.

Status words: **Verified** (the page text supports the row), **Table exists, description not read**, **To confirm** (not stated as fact).

The report's Appendix D is the short version of this file.

## 1. Concepts

| Term | Meaning | Status | Source |
|---|---|---|---|
| Resolute | Epic's revenue cycle application. Hospital Billing covers facility fees, Professional Billing covers professional fees. | Verified | https://epicsupport.sites.uiowa.edu/node/111 |
| Hospital billing (HB) tables, prefix HSP_ | Counterpart of this project's outpatient (facility) lines. | Verified (definitions); the pairing is my reading | same |
| Professional billing (PB) tables, prefix ARPB_ | Counterpart of this project's carrier (clinician) lines. | Verified (definitions); the pairing is my reading | same |

## 2. Field mapping (report Table D1)

| My field | Hospital billing: HSP_TRANSACTIONS | Professional billing: ARPB_TRANSACTIONS | Status |
|---|---|---|---|
| Claim line | One row per transaction (key TX_ID), tied to an account by HSP_ACCOUNT_ID. TX_TYPE_HA_C_NAME is Charge, Payment, Debit Adjustment or Credit Adjustment. | One row per transaction (key TX_ID), tied to an account by ACCOUNT_ID. TX_TYPE_C_NAME is charge, payment or adjustment. | Verified: tables. To confirm: building one line from several rows. |
| Procedure code | PROC_ID_PROC_NAME (name), MODIFIERS, UB_REV_CODE_ID, HCPCS_CODE (an override code) | PROC_ID_PROC_NAME (name), MODIFIER_ONE to MODIFIER_FOUR. No code column in this table (all 67 columns read). | Verified: columns. To confirm: where the base CPT/HCPCS code lives. |
| Place of service | PLACE_OF_SVC_ID_LOC_NAME | POS_ID_LOC_NAME | Verified |
| Billed amount | BILLED_AMOUNT | AMOUNT ("the original amount of this transaction") | Verified |
| Allowed amount | ALLOWED_AMOUNT | None in this table | Verified (HB). To confirm (PB). |
| Paid amount | TX_AMOUNT on Payment rows; CLM_PMT_HISTORY (claim-level history of how a payment was processed) | Payment-type rows with INSURANCE_AMT (the insurance portion), TOTAL_MATCH_AMT, and ARPB_TX_MATCH_HX | Verified: columns exist. To confirm: joining payment rows to charge rows. CLM_PMT_HISTORY: table exists, description not read in full. |
| Provider | BILLING_PROV_ID_PROV_NAME, PERFORMING_PROV_ID_PROV_NAME | BILLING_PROV_ID_PROV_NAME, SERV_PROVIDER_ID_PROV_NAME | Verified |
| Payer, financial class | PAYOR_ID_PAYOR_NAME, FIN_CLASS_C_NAME (includes Medicare) | PAYOR_ID_PAYOR_NAME, ORIGINAL_FC_C_NAME | Verified |
| Service date | SERVICE_DATE, TX_POST_DATE | SERVICE_DATE, POST_DATE | Verified |
| Diagnosis | Not located in HSP_TRANSACTIONS | PRIMARY_DX_ID_DX_NAME, DX_TWO_ID_DX_NAME to DX_SIX_ID_DX_NAME (names); ARPB_CHG_ENTRY_DX lists all diagnoses on a charge entry session | Verified (PB). To confirm (HB). |

## 3. Denial mapping (report Table D2)

The project's six built reasons and the CARC each carries come from `src/denial_reasons.py` (REASON_CATALOG). Appeals are not one of the six.

| Built reason | CARC | Epic location | Status |
|---|---|---|---|
| Procedure code missing | 16 | Not located (HCPCS_CODE is an override in HB; no code column in ARPB_TRANSACTIONS) | To confirm |
| Retired consultation code | 181 | Not located (the denial event is recorded, see below; the remittance reason table was not verified) | To confirm |
| Duplicate of an earlier line | 18 | ARPB_PMT_RELATED_DENIALS excludes denials with remittance reason 18, so duplicates are not counted in that denial-rate table | Verified: the exclusion. To confirm: where duplicates are recorded. |
| Prior authorization missing | 197 | ARPB_AUTH_INFO ("authorization information for a charge transaction", keys TX_ID, LINE; AUTH_ID links authorization records) and ATB_AUTH_DENIAL_RSNS ("reasons for denial when the authorization decision is denied", keys AUTH_BUNDLE_ID, LINE; column AUTH_PYR_DENIAL_REASON_C_NAME). Both describe the authorization, not the claim denial. | Verified: both tables. Match to CARC 197 is the closest, to confirm. |
| Diagnosis does not support procedure | 11 | Not located (diagnosis and procedure-name columns are in section 2) | To confirm |
| Provider bills far above peers | 16 (a proxy in this project) | Not a recorded reason; would be computed from provider and billed-amount columns | To confirm |

**Where a denial itself is recorded (not tied to one reason):**

| What | Location | Status |
|---|---|---|
| HB denial action | HSP_TRANSACTIONS ACTION_STRING (1 Not Allowed Adjustment, 2 Next Responsible Party, 3 Claim Denied); TX_SOURCE_HA_C_NAME includes Electronic Remittance, Claim Denial Payment, DBC Denied Invoice | Verified |
| PB denial records | ARPB_PMT_RELATED_DENIALS (keys TX_ID, LINE; column RELATED_BDC_ID), for denial-rate metrics. Only denials converted to Epic's Denial/Remark/Correspondence framework are included. | Verified |

## 4. Appeals (not modeled in this project)

| Table | Description read | Status |
|---|---|---|
| APPEAL_GRV | "An individual appeal or grievance." Columns include the subject (an authorization or a claim), review level, decision and a member (PAT_ID). These look like the health-plan side of Epic. | Verified description; which a hospital uses: to confirm |
| ARPB_TX_VENDOR_APPEAL | "Status of Vendor Appeals." Per-transaction response values: Correct Claim, Claim Needs Attention, Under Appeal, Successful Appeal, Denied Appeal, Pending Further Action, Partial Recovery; also VENDOR_REIMB and VENDOR_VARIANCE. | Verified |

## 5. Other tables read

| Table | What it holds | Status |
|---|---|---|
| HSP_ACCOUNT | Hospital account information from the Hospital Account and Claim master files (key HSP_ACCOUNT_ID) | Verified (description) |
| ARPB_TX_MATCH_HX | A line is added whenever two transactions are matched together; stores matched amounts (total, insurance, self-pay) and match and unmatch dates | Verified |

## 6. Step 0 status changes (October 2, 2026)

- Promoted to Verified: ARPB_TRANSACTIONS (all 67 columns read), ATB_AUTH_DENIAL_RSNS, ARPB_AUTH_INFO, ARPB_CHG_ENTRY_DX, ARPB_TX_MATCH_HX, APPEAL_GRV, ARPB_TX_VENDOR_APPEAL.
- Moved to To confirm: allowed amount on the professional side (no such column in ARPB_TRANSACTIONS); the base procedure code on the professional side (the table has the procedure name only, and no column name contains ALLOW, CPT, HCPCS or ICD); how payment rows join to charge rows.

## 7. To confirm (not stated as fact anywhere)

- Where diagnosis codes live for hospital billing.
- Where the base CPT/HCPCS code lives (HCPCS_CODE in HB is an override).
- Which tables hold remittance reason codes (CL_RMT_SVCE_LN_INF was seen only in third-party material and was not read).
- CLM_VALUE_RECORD, CL_PB_CLM, PAT_ENC, PAT_ENC_HSP, CLARITY_SER, CLARITY_TDL_TRAN, and Caboodle (Epic's data warehouse).
- Whether a hospital uses APPEAL_GRV, or only the transaction-level appeal status.

## 8. Open questions

- **Claim line versus billing transaction.** My unit is a claim line; Epic's is a billing transaction. A line-level view would need charge, payment and adjustment rows joined by account and procedure. This join is an open question, not a solved step.
- **Provider-side cost question.** On the provider side the question becomes the cost of working or appealing a denial versus the amount recovered. Whether the results here would carry over to Epic data has not been tested and is not claimed.

No SQL against Epic tables is included, and none was run.
