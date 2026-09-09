# Target Variable Definition — `is_denied`

## Decision (2026-08-03)

**Data source:** CMS Synthetic Medicare Enrollment, Fee-for-Service Claims, and Prescription
Drug Event PUF ("CMS Synthetic Claims PUF"), 8,671 synthetic beneficiaries, Synthea-generated,
pipe-delimited RIF-format CSVs. Source: https://data.cms.gov/collection/synthetic-medicare-enrollment-fee-for-service-claims-and-prescription-drug-event

**Why not use a native CMS denial field:** Verified directly against the CMS user guide
(May 2023) before writing any code. Every field that would carry a denial/non-payment signal in
this release is a **fixed constant across all 8,671 beneficiaries**, not real variation:

| File | Field | Documented value |
|---|---|---|
| Carrier / DME | `CARR_CLM_PMT_DNL_CD` (Payment Denial Code) | fixed `1` for every row |
| Inpatient/Outpatient/HHA/Hospice/SNF | `CLM_MDCR_NON_PMT_RSN_CD` | `[Blank]` for every row |
| DME / Carrier | `CLM_DISP_CD` (Disposition Code) | fixed `1` for every row |

This isn't specific to this release — it's structural. Synthea only simulates clinical encounters
that get billed and paid, so the RIF exporter never populates a real denial outcome. The older
DE-SynPUF has the same limitation for a related reason: it's built from finalized/paid NCH claims
history. **No public CMS claims PUF contains a genuine claim-level denial/approval outcome** —
that's proprietary payer adjudication data, not something CMS republishes. (Kaggle's "Healthcare
Provider Fraud Detection" dataset doesn't fill this gap either — its label is a provider-level
fraud flag, not a claim-level denial outcome.)

## Chosen approach

Use the real CMS synthetic claims (real HCPCS/ICD-10 codes, provider specialties, beneficiary
demographics, claim timing, payment amounts) for feature realism, and construct `is_denied` via a
**documented, rule-based label** grounded in real adjudication logic rather than a native CMS
field. This is disclosed as an explicit, engineered modeling decision — not presented as ground
truth CMS denial data — in the README and report.

### Label mechanism: probabilistic (noisy-OR), not deterministic (see `src/denial_reasons.py`)

**Revised 2026-08-05.** The first pass used a hard boolean-OR across rules (any rule firing =
100% denied). Two problems with that: (1) it makes the Phase 2 modeling step close to
meaningless — if `is_denied` is a deterministic function of the same fields fed to the model, the
model just re-derives the rule, and an unrealistically perfect PR-AUC is itself a leakage red
flag; (2) it's not how real adjudication behaves — payers don't deny 100% of claims that trip a
risk factor (reviewer discretion, retroactive documentation, minor-gap tolerance).

Current approach: each risk factor contributes a **conditional probability of denial given the
factor is active**, combined via **noisy-OR** — `P(denied) = 1 − Π(1 − pᵢ)` over active factors —
then the actual outcome is Bernoulli-sampled from that probability. This produces graded risk and
realistic multi-reason denials (when 2+ factors are active, their probabilities compound) without
requiring a real joint-probability table, which doesn't exist publicly (see below).

Risk factors (with each one's illustrative CARC code and base conditional probability):

1. **Missing prior authorization proxy** (CARC 197, `base_prob=0.22`) — high-cost procedure
   categories (DME, certain outpatient procedure codes) with no linked prior encounter/diagnosis
   support in the beneficiary history.
2. **Diagnosis–procedure mismatch** (CARC 11, `base_prob=0.15`) — procedure code's typical
   diagnosis category doesn't match any diagnosis billed on the same claim (coarse
   medical-necessity proxy).
3. **Provider outlier billing pattern** (CARC 16, `base_prob=0.08`) — provider bills in the top
   percentile of claim volume or payment amount relative to peers (rough proxy for audit-flagged
   providers, weakest-grounded rule, deliberately given the lowest base rate).
4. **Timely filing violation** — not implemented; `FI_CLM_PROC_DT` (claim processing date) is
   blank/fixed in this synthetic release, so days-between-service-and-submission can't be
   computed. Documented as a disclosed limitation, not silently dropped.

**Where the base probabilities came from, and why they're assumptions, not facts:** searched
specifically for published joint/conditional denial-reason probabilities — none exist publicly.
What's published (Experian Health 2025 State of Claims, Kodiak Solutions/HFMA 2024, MGMA, Aptarro
industry compilations) are self-reported **marginal** survey shares — e.g. "~35% of revenue-cycle
leaders cite prior authorization as a top denial driver" — which is a frequency-of-mention stat,
not a measured claim-level incidence rate, and these don't sum to 100% since respondents cite
multiple reasons. Real CARC/RARC co-occurrence data lives inside payer adjudication systems and
isn't released, for the same underlying reason no public CMS PUF has a real denial field. The
`base_prob` values in `REASON_CATALOG` (`src/denial_reasons.py`) are loosely anchored to those
marginal benchmarks but are ultimately reasoned, disclosed assumptions — a single
`CALIBRATION_SCALE` knob tunes them uniformly to hit the target 10-15% overall rate rather than
hand-tuning each one to a number that would falsely imply real-world precision.

**Multi-reason assignment:** for denied claims, the recorded reason(s) are sampled from the active
risk factors, weighted by each factor's contribution — claims with 2+ active factors have a
documented 30% chance of carrying a second CARC code. This is also a disclosed assumption, not a
measured multi-reason rate.

**Leakage note this design surfaces:** `CLM_PMT_AMT` is overwritten to ~$0 for denied claims
(`apply_payment_consequence()`) for internal consistency — this makes it a near-deterministic
*consequence* of the label, not an independent feature, so it must be dropped from the Phase 2
feature set along with the `risk_*` / `p_denied_model` / `denial_reason_carc_*` columns used to
construct the label. See `data/data_dictionary.md`.

### What this means for the report / interview story

State this plainly, don't obscure it: *"Public CMS claims data doesn't expose real denial
outcomes — that's proprietary payer data. I used real CMS claims structure and codes for realism
and engineered a rule-based denial target grounded in documented adjudication logic, then
validated that the resulting label produces a realistic ~5-20% minority-class rate consistent with
published payer denial-rate ranges."* This is a stronger interview answer than a shortcut, because
it shows you understood the data landscape well enough to know a native denial field doesn't
exist anywhere in the public domain — most candidates wouldn't catch that.

### Limitations to disclose in Phase 5 report

- `is_denied` is an engineered proxy, not observed ground truth — SHAP/model findings describe
  what predicts the *rule-based* label, not real payer denial behavior.
- Timely-filing logic couldn't be included (processing date not populated in this release).
- Rule thresholds (e.g., the $0-payment cutoff, outlier billing percentile) are documented,
  arbitrary-but-reasoned choices — sensitivity to these should be spot-checked, not assumed robust.
