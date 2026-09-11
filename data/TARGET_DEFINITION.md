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

---

## Addendum: Design Q&A log (2026-09-10)

Working notes from a design-review conversation, kept verbatim-in-substance rather than
summarized, per the standing rule that this file doubles as interview prep — these are exactly
the follow-up questions a hiring manager might ask.

### 1. Noisy-OR independence assumption — disclosed limitation, not previously named

`P(denied) = 1 − Π(1 − pᵢ)` treats `missing_prior_auth`, `dx_procedure_mismatch`, and
`provider_outlier` as **statistically independent** given each is active. This was implicit in
the math above but never stated as a limitation. It should be: in reality these are plausibly
correlated (a provider with an outlier billing pattern is plausibly *more* likely to also produce
diagnosis/procedure mismatches — same underlying sloppy-billing cause), so the true joint denial
probability when multiple factors co-occur may differ from what independence implies. This is a
disclosed modeling simplification, not a measured fact.

### 2. `CALIBRATION_SCALE` mechanics — where in the pipeline scaling actually happens

Implementation detail worth being precise about: `calibration_scale` multiplies each **individual
factor's `base_prob` before noisy-OR combination**, not the combined `p_denied` after combination
(see `sample_denials()` in `src/denial_reasons.py`):

```python
base_probs = {k: v["base_prob"] * calibration_scale for k, v in REASON_CATALOG.items()}
base_probs = {k: min(p, 0.95) for k, p in base_probs.items()}  # safety clamp
p_denied = noisy_or_probability(risk_factors, base_probs)
```

This is not mathematically equivalent to scaling the final combined probability (`1 − Π(1−pᵢ) )
× scale`), because noisy-OR is nonlinear — scaling before combination keeps each `pᵢ`
individually interpretable as "this factor's own conditional denial probability" (consistent with
how `REASON_CATALOG` documents it) and lets claims with more active factors still compound off the
scaled-down `pᵢ` values, rather than applying one flat multiplier regardless of how many factors
fired. The `min(p, 0.95)` clamp prevents any single factor from being treated as deterministic
even under aggressive scaling.

**No closed-form guarantee of landing in 10–15%.** There's no algebraic solution for "what scale
gives exactly 12%" — the population-level rate depends on the calibrated `pᵢ` values *and* the
distribution of how many claims have 0/1/2/3 active factors, which is itself a property of the
real CMS data. Calibration is empirical: guess a `calibration_scale`, run `calibration_report()`,
check `is_denied.mean()`, adjust, repeat — not something the formula guarantees on its own.

### 3. Reason-code selection — moved from weighted-random to deterministic priority

**Original implementation:** for denied claims with 2+ active factors, the primary/secondary
reason was chosen via weighted random sampling, weights = each active factor's calibrated
`base_prob`.

**Problem identified:** this column (`denial_reason_carc_1/2`) is excluded from the Phase 2
feature set entirely (see data dictionary), so the selection mechanism has zero effect on
modeling. Its only real consumer is Phase 4 (SHAP narrative, RAG corpus, grounding check) and
Phase 5 (report language) — neither of which is a modeling target. For those consumers,
randomizing *which* factor gets blamed when the same set of factors is active on different runs
actively hurts the "here's why this specific claim was flagged" narrative — it makes demo output
non-reproducible without adding anything realistic, since real adjudication systems generally use
a fixed rule-priority hierarchy, not a probability-weighted lottery, to decide which edit gets
recorded when several fire.

**Revised approach: deterministic priority order, justified by processing stage, not by
`base_prob` rank.** Verified against several claims-adjudication process descriptions: real
adjudication runs format/eligibility checks first, then medical-policy edits (prior auth, medical
necessity), then bundling/frequency edits, with provider-level audit/outlier review happening
**retrospectively** (post-payment), not as a real-time adjudication edit at all. This gives a
principled fixed order:

1. `duplicate_claim` (if implemented — see #4 below) — earliest, typically one of the cheapest/
   first system checks, ahead of most medical-policy edits.
2. `missing_prior_auth` (CARC 197) — front-end, pre/early-adjudication gate.
3. `dx_procedure_mismatch` (CARC 11) — medical-policy/medical-necessity edit, mid-adjudication.
4. `provider_outlier` (CARC 16) — **always last**, not because it's the lowest-probability
   factor, but because it's structurally a retrospective audit mechanism — by the time an
   outlier-billing review flags a claim, any real-time adjudication reason would already have
   been recorded first. This is a *causal/temporal* justification, not a statistical one, and
   it happens to coincide with `provider_outlier` already having the lowest `base_prob`, which
   is a coincidence worth noting rather than treating as confirmation.

Coincidentally, `base_prob`-rank and this processing-order rank agree on direction for the three
original factors — but **agreement in ranking direction does not mean the resulting population
proportions (e.g., "62% of denials primary-attributed to CARC 197") are realistic.** That
proportion is driven entirely by each rule's *activation rate* on the actual CMS data
(`compute_risk_factors()` output), not by the tie-break rule. Check
`calibration_report()`'s per-rule activation rates against Experian's ~35%-cite-prior-auth
framing empirically — don't assume the tie-break choice moves that number, because it doesn't.

**Status: recommended, not yet implemented in `denial_reasons.py`.** The weighted-random sampling
code is still what's running. This section documents the design decision and rationale; the code
change (replace weighted `rng.choice` with a fixed-priority argmax over active factors) is still
open.

### 4. Candidate additional risk factor: duplicate claim — recommended, not yet implemented

**Why:** duplicate-claim denials (CARC 18) are one of the best-grounded categories available —
stronger than `provider_outlier`'s grounding. A Louisiana Medicaid transparency report showed
duplicate claims as **30.95% of all denials**, the second-largest single category (behind invalid
procedure/modifier combinations at 39.51%, ahead of "no authorization on file" at 18.62%).
Multiple industry sources independently describe duplicate billing as one of the most commonly
cited administrative denial reasons. (One additional source claimed the three most common
real-world CARC drivers are 197/11/16 — i.e. exactly the three factors already chosen here; that
source reads as aggregator/SEO content rather than a primary report like Kodiak/HFMA, so treat as
a nice-to-have citation to verify, not load-bearing.)

**Why it's cheap:** `load_data.py` already loads and correctly dtypes `BENE_ID`, `HCPCS_CD`, and
`CLM_FROM_DT`; `data_dictionary.md` confirms `CLM_ID` (unique claim identifier) is present in
every claim file. A duplicate rule is: group by `(BENE_ID, HCPCS_CD, CLM_FROM_DT, PRVDR_NUM)`,
flag any group with more than one distinct `CLM_ID`. No new fields to source.

**Caveat to document if implemented:** not every same-code/same-date repeat billing is a true
duplicate — recurring services (dialysis, physical therapy, some DME rentals) legitimately repeat.
Real duplicate-detection logic typically also checks exact-match units/modifiers. Document this
as a simplification, same pattern as the existing three rules.

**Why not implement every plausible factor instead of stopping at 3–4:** each additional factor
is another `base_prob` anchored to the same thin marginal survey data the existing three already
stretch, another `risk_*` column requiring leakage-inventory discipline, and time spent on
marginal realism that doesn't change the modeling task (Phase 1 is already running over its
original estimate). Duplicate-claim clears the bar because it's both unusually well-grounded *and*
free given fields already loaded — that combination doesn't generalize to "add more."

**Also considered, more effort, not "cheap":** an eligibility/coverage-lapse check (claim date
outside the beneficiary's enrollment window, CARC 27) is well-grounded but requires joining the
`beneficiary_2015.csv`…`beneficiary_2023.csv` enrollment files, which `load_data.py` doesn't
currently load. Documented as possible future work, not pursued now.

### 5. Why `CLM_PMT_AMT` cannot be used as the label — heuristic *or* target

Two distinct claims get conflated if this isn't spelled out separately:

**(a) As a detection heuristic for `is_denied` (thresholding `CLM_PMT_AMT ≈ 0`):** wrong even in
real-world data, because $0 insurer payment has causes other than denial — deductible/coinsurance
absorption (patient owes the money, provider *does* get paid, just not by the insurer), and
bundling (payment rolled into another line). Both are correctly-adjudicated, non-denied claims
that a raw threshold would mislabel.

**(b) As the target itself (redefine the problem as "predict $0-pay lines," not "predict
denial"):** doesn't remove the problem, relocates it — this framing conflates three operationally
different events (true denial, deductible absorption, bundling) under one label. A SHAP story
built on that target would be muddied by whatever drives deductible-exhaustion or bundling
patterns in the data, unrelated to denial risk.

**Synthea-specific reason, stated with appropriate confidence:** this project's `is_denied` is
engineered specifically *because* Synthea has no denial concept in its claims export (see
"Decision" section above). By the same logic, there's no strong reason to expect Synthea's
*native* `CLM_PMT_AMT` — driven by a synthetic insurance-plan cost/payment-split module — to
already encode anything resembling `missing_prior_auth`, `dx_procedure_mismatch`, or
`provider_outlier`-style risk. This is a reasoned expectation, not a verified fact about Synthea's
internals.

**Recommended validation (not yet run):** before any engineered `is_denied` logic is applied,
cross-tabulate native, unmodified `CLM_PMT_AMT ≈ 0` claims against the three (or four, if
duplicate-claim is added) risk-factor flags. If native zero-pay claims already correlate with
these factors, that's a genuinely interesting, reportable finding. If not, it's direct empirical
evidence — stronger than reasoning alone — for why the native payment field can't be used as
either a detection heuristic or a target. **Action item: run this check during Phase 1 EDA and
record the result here.**