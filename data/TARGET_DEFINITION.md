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

Risk factors (with each one's illustrative CARC code and base conditional probability). Updated
2026-09-12 to 5 factors (see Addendum #4 for the 4th, Addendum #6 for the 5th). **Final real-data
calibration confirmed 2026-09-12 — see Addendum #7: overall 9.4% denial rate, 3 of 5 factors
genuinely contributing.**

1. **Deprecated / Medicare-non-payable procedure code** (CARC 181, `base_prob=0.85`) — CPT
   consultation codes CMS stopped recognizing for Medicare Part B payment effective 2010-01-01.
   Highest-confidence factor in the set — anchored to an exact federal policy, not a survey
   estimate. Added 2026-09-12; see Addendum #6.
2. **Duplicate claim** (CARC 18, `base_prob=0.35`) — same beneficiary + procedure + service date +
   provider recorded under more than one distinct claim ID. Added 2026-09-11; see Addendum #4.
3. **Missing prior authorization proxy** (CARC 197, `base_prob=0.22`) — high-cost procedure
   categories (DME, certain outpatient procedure codes) with no linked prior encounter/diagnosis
   support in the beneficiary history.
4. **Diagnosis–procedure mismatch** (CARC 11, `base_prob=0.15`) — procedure code's typical
   diagnosis category doesn't match any diagnosis billed on the same claim (coarse
   medical-necessity proxy). Real HCPCS mapping built 2026-09-12; see Addendum #7.
5. **Provider outlier billing pattern** (CARC 16, `base_prob=0.08`) — provider bills in the top
   percentile of claim volume or payment amount relative to peers (rough proxy for audit-flagged
   providers, weakest-grounded rule, deliberately given the lowest base rate).
6. **Timely filing violation** — not implemented; `FI_CLM_PROC_DT` (claim processing date) is
   blank/fixed in this synthetic release, so days-between-service-and-submission can't be
   computed. Documented as a disclosed limitation, not silently dropped.

**Where the base probabilities came from, and why they're assumptions, not facts:** searched
specifically for published joint/conditional denial-reason probabilities — none exist publicly.
What's published (Experian Health 2025 State of Claims, Kodiak Solutions/HFMA 2024, MGMA, Aptarro
industry compilations) are self-reported **marginal** survey shares — e.g. "~35% of revenue-cycle
leaders cite prior authorization as a top denial driver" — which is a frequency-of-mention stat,
not a measured claim-level incidence rate, and these don't sum to 100% since respondents cite
multiple reasons. Real CARC/RARC co-occurrence data lives inside payer adjudication systems and
isn't released, for the same underlying reason no public CMS PUF has a real denial field. Most
`base_prob` values in `REASON_CATALOG` (`src/denial_reasons.py`) are loosely anchored to those
marginal benchmarks but are ultimately reasoned, disclosed assumptions — a single
`CALIBRATION_SCALE` knob tunes them uniformly to hit the target 10-15% overall rate rather than
hand-tuning each one to a number that would falsely imply real-world precision. `deprecated_code`
is the one exception — see Addendum #6 for why it's grounded differently and set higher.

**Multi-reason assignment:** for denied claims, the recorded reason(s) are assigned by fixed
processing-order priority (changed 2026-09-11 from weighted-random — see Addendum #3), and claims
with 2+ active factors have a documented 30% chance of carrying a second CARC code. The 30% figure
is a disclosed assumption, not a measured multi-reason rate.

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

`P(denied) = 1 − Π(1 − pᵢ)` treats all active risk factors as **statistically independent** given
each is active. This was implicit in the math above but never stated as a limitation. It should
be: in reality these are plausibly correlated (a provider with an outlier billing pattern is
plausibly *more* likely to also produce diagnosis/procedure mismatches — same underlying
sloppy-billing cause), so the true joint denial probability when multiple factors co-occur may
differ from what independence implies. This is a disclosed modeling simplification, not a
measured fact. **Empirically checked 2026-09-13 — see Addendum #8: `duplicate_claim` and
`deprecated_code` are confirmed to co-occur heavily (98.6% of `duplicate_claim` claims are also
`deprecated_code`), so the independence assumption is measurably violated for at least this pair.**

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
check `is_denied.mean()`, adjust, repeat — not something the formula guarantees on its own. **This
was confirmed concretely on real data 2026-09-12 — see Addendum #6/#7: the first real run landed
at 1.4%, then 7.0% after the DME fix, then 9.4% once `dx_procedure_mismatch`'s mapping was
actually running (see Addendum #7) — the cause was never the calibration scale itself; it was two
under-firing rules, both now fixed.**

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
principled fixed order, implemented as `REASON_PRIORITY` in `src/denial_reasons.py`. Updated
2026-09-12 to include `deprecated_code` (see Addendum #6), placed FIRST — code-validity is a
harder, more upfront system check than even duplicate detection, since a system needs a
currently-recognized procedure code before it's even meaningful to check whether that code was
billed twice:

1. `deprecated_code` — earliest: a system needs a currently-recognized, payable procedure code
   before duplicate-checking or any other edit is even meaningful.
2. `duplicate_claim` — early/front-end system check, ahead of most medical-policy edits.
3. `missing_prior_auth` (CARC 197) — front-end, pre/early-adjudication gate.
4. `dx_procedure_mismatch` (CARC 11) — medical-policy/medical-necessity edit, mid-adjudication.
5. `provider_outlier` (CARC 16) — **always last**, not because it's the lowest-probability
   factor, but because it's structurally a retrospective audit mechanism — by the time an
   outlier-billing review flags a claim, any real-time adjudication reason would already have
   been recorded first. This is a *causal/temporal* justification, not a statistical one, and
   it happens to coincide with `provider_outlier` already having the lowest `base_prob`, which
   is a coincidence worth noting rather than treating as confirmation.

Coincidentally, `base_prob`-rank and this processing-order rank agree on direction across all
factors — but **agreement in ranking direction does not mean the resulting population proportions
(e.g., "62% of denials primary-attributed to CARC 197") are realistic.** That proportion is driven
entirely by each rule's *activation rate* on the actual CMS data (`compute_risk_factors()`
output), not by the tie-break rule. Check `calibration_report()`'s per-rule activation rates
against Experian's ~35%-cite-prior-auth framing empirically — don't assume the tie-break choice
moves that number, because it doesn't.

**Status: implemented** in `src/denial_reasons.py` — `REASON_PRIORITY` constant plus a
deterministic walk down it in `sample_denials()`, replacing the weighted `rng.choice` call. The
~30% second-reason chance is kept (see the "Multi-reason assignment" note above the addendum) —
that's the one piece of randomness kept because it reflects real population variety in *how many*
factors are active, not *which* one is blamed.

### 4. Candidate additional risk factor: duplicate claim — implemented

**Why:** duplicate-claim denials (CARC 18) are one of the best-grounded categories available —
stronger than `provider_outlier`'s grounding. A Louisiana Medicaid transparency report showed
duplicate claims as **30.95% of all denials**, the second-largest single category (behind invalid
procedure/modifier combinations at 39.51%, ahead of "no authorization on file" at 18.62%).
Multiple industry sources independently describe duplicate billing as one of the most commonly
cited administrative denial reasons. (One additional source claimed the three most common
real-world CARC drivers are 197/11/16 — i.e. exactly the three factors already chosen here; that
source reads as aggregator/SEO content rather than a primary report like Kodiak/HFMA, so treat as
a nice-to-have citation to verify, not load-bearing.)

**Why it was cheap:** `load_data.py` already loads and correctly dtypes `BENE_ID`, `HCPCS_CD`, and
`CLM_FROM_DT`; `data_dictionary.md` confirms `CLM_ID` (unique claim identifier) is present in
every claim file. The rule is: group by `(BENE_ID, HCPCS_CD, CLM_FROM_DT, PRVDR_NUM)`, flag any
group with more than one distinct `CLM_ID`. No new fields needed.

**Caveat, RESOLVED 2026-09-12 — see Addendum #6 for the full investigation.** Not every
same-code/same-date repeat billing is a true duplicate — recurring services (dialysis, physical
therapy, some DME rentals) legitimately repeat, and real duplicate-detection logic typically also
checks exact-match units/modifiers, not available here. On real data the activation rate came out
at 0.057% — investigated in full, and the finding was more specific and more interesting than
either possibility originally anticipated: see Addendum #6.

**Status: implemented** — `rule_duplicate_claim()` in `src/denial_rules.py`, wired into
`REASON_CATALOG`/`RISK_FACTOR_FUNCS`/`compute_risk_factors()` in `src/denial_reasons.py`.

**Why not implement every plausible factor instead of stopping at a handful:** each additional
factor is another `base_prob` anchored to the same thin marginal survey data the existing ones
already stretch, another `risk_*` column requiring leakage-inventory discipline, and time spent on
marginal realism that doesn't change the modeling task. Duplicate-claim cleared the bar because it
was both unusually well-grounded *and* free given fields already loaded. `deprecated_code`
(Addendum #6) cleared an even higher bar — exact federal policy grounding, also free — which is
why it was added despite this same scope-discipline argument; that combination still doesn't
generalize to "add more."

**Also considered, more effort, not "cheap":** an eligibility/coverage-lapse check (claim date
outside the beneficiary's enrollment window, CARC 27) is well-grounded but requires joining the
`beneficiary_2015.csv`…`beneficiary_2023.csv` enrollment files, which `load_data.py` doesn't
currently load. Documented as possible future work, not pursued now — and deliberately NOT
downloaded "just in case" (2026-09-11 decision): downloading data against a speculative,
not-yet-decided feature is the same scope creep this section already argues against for risk
factors generally.

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

**Synthea-specific reason, originally stated with appropriate confidence, PARTIALLY WRONG per
Addendum #8:** this project's `is_denied` is engineered specifically *because* Synthea has no
denial concept in its claims export (see "Decision" section above). By that same logic, it was
reasoned there was "no strong reason to expect" Synthea's *native* `CLM_PMT_AMT` to already
correlate with the risk factors. **That reasoning held for `deprecated_code` (confirmed, no
correlation) but was WRONG for `missing_prior_auth` (a strong, real correlation was found on
actual data, since fully explained — see Addendum #8) — see Addendum #8 for the numbers and the
resolved explanation.** Worth keeping this correction visible rather than quietly fixing the
original claim: reasoning from first principles about what a data-generation process "shouldn't"
do is a hypothesis, not a fact, and this is a concrete example of that hypothesis being checked
and partly failing.

**Validation check: implemented AND RUN 2026-09-13 — see Addendum #8 for full results.** The
action item that was open since 2026-09-11 is now closed. `validate_native_payment_vs_risk_factors()`
in `src/run_eda.py` cross-tabulated native, unmodified `CLM_PMT_AMT ≈ 0` claims against the
risk-factor flags on real data, using `combined_claims_raw.parquet` before
`apply_payment_consequence()` had touched it.

---

## Addendum: Phase 1 real-data run log (2026-09-11 to 2026-09-12)

First run of the pipeline against real downloaded CMS data. Two bugs found and fixed, one
significant discovery from investigating a diagnostic question, one design confirmation — kept
here in full since this is exactly the kind of debugging narrative worth having ready for an
interview ("walk me through a bug you hit and how you found it").

### Bug 1: parquet write failure on `PRVDR_STATE_CD` dtype mismatch

`load_data.py`'s first real run (`carrier.csv` + `outpatient.csv`, 1,696,096 combined rows) raised
`pyarrow.lib.ArrowTypeError: ("Expected bytes, got a 'int' object", ...)` writing
`combined_claims_raw.parquet`. Root cause: `PRVDR_STATE_CD` (an SSA state code, can carry a
leading zero) wasn't on the explicit `CODE_DTYPE_COLS` list, so pandas inferred its dtype
independently per source file — `carrier.csv` and `outpatient.csv` disagreed (one int, one not),
and `pd.concat` produced a mixed-type object column pyarrow couldn't serialize.

**Fix:** generalized the dtype-forcing rule in `load_claim_file()` — force `str` on every column
ending in `_CD` or `_NUM`, not just the explicit list, since CMS RIF fields with that suffix
pattern are always categorical identifiers, never numeric quantities. This closes the whole
category rather than requiring a one-off fix per column as more claim files get added later
(inpatient/SNF/hospice/hha still to come).

### Bug 2: `missing_prior_auth` fired at exactly 0.000000%

Not a code bug — a data-coverage gap. `rule_missing_prior_auth` flags HCPCS codes starting with
`"E"`/`"K"` (DME Level II codes), but only `carrier.csv` and `outpatient.csv` were loaded, and
those claim types essentially never use DME coding.

**Fix:** added `dme.csv` (already downloaded, not yet loaded) to `load_data.py`'s default claim
file set (now 1,799,924 combined rows across 3 claim types). Reasoned as more realistic, not just
a patch: CMS's actual "Required Prior Authorization for Certain DMEPOS Items" program is a real
prior-auth mechanism specifically for DME, so pairing `missing_prior_auth` with DME claims is
arguably a *better* fit than the original carrier/outpatient-only scope, not merely a bug
workaround.

### Calibration numbers — three runs, tracked in full

**Run 1 (pre-DME-fix):**
```
Overall is_denied rate: 1.4%  (target: 10-15%, hard bound 5-20%)

Risk factor activation rates (all claims):
risk_provider_outlier         12.7107%
risk_duplicate_claim           0.0571%
risk_dx_procedure_mismatch     0.0184%
risk_missing_prior_auth        0.0000%

Of denied claims, 0.2% carry a second reason code

Primary reason distribution (of denied claims):
16 (provider_outlier)    97.87%
18 (duplicate_claim)      1.87%
11 (dx_procedure_mismatch) 0.26%
```

**Run 2 (post-DME-fix, post-deprecated_code, PRE dx_procedure_mismatch mapping fix — this run
also happened to be executed against a stale local file, see Addendum #7's note on the pull-lag
diagnostic; numbers below are what that stale state actually produced):**
```
Overall is_denied rate: 7.0%

Risk factor activation rates (all claims):
risk_provider_outlier         20.0152%
risk_deprecated_code           5.3337%
risk_missing_prior_auth        0.0798%
risk_duplicate_claim           0.0546%
risk_dx_procedure_mismatch     0.0173%

Of denied claims, 10.9% carry a second reason code

Primary reason distribution (of denied claims):
181 (deprecated_code)    72.1032%
16  (provider_outlier)   27.4992%
197 (missing_prior_auth)  0.3377%
11  (dx_procedure_mismatch) 0.0529%
18  (duplicate_claim)     0.0071%
```

**Run 3 (final — all fixes actually loaded, including the real `dx_procedure_mismatch` mapping):**
```
Overall is_denied rate: 9.4%  (target: 10-15%, hard bound 5-20%)

Risk factor activation rates (all claims):
risk_provider_outlier         20.0152%
risk_dx_procedure_mismatch    11.6172%
risk_deprecated_code           5.3337%
risk_missing_prior_auth        0.0798%
risk_duplicate_claim           0.0546%

Of denied claims, 11.2% carry a second reason code

Primary reason distribution (of denied claims):
181 (deprecated_code)      53.9418%
11  (dx_procedure_mismatch) 29.1001%
16  (provider_outlier)     16.6931%
197 (missing_prior_auth)    0.2597%
18  (duplicate_claim)       0.0053%
```

**This is the number to cite going forward.** 9.4% overall, inside the 5-20% hard bound and just
under the 10-15% soft target (itself derived from a marginal survey stat, not a precise number —
see the base-probability sourcing discussion above). Three of five factors are now genuinely
contributing to the primary-reason distribution (53.9% / 29.1% / 16.7%), a real improvement over
Run 1's single-factor-dominance problem. `missing_prior_auth` and `duplicate_claim` remain small
but are now fully explained rather than mysterious (see Addendum #6 for `duplicate_claim`;
`missing_prior_auth`'s proxy is inherently narrow by design — DME codes + a 90-day no-prior-claim
window).

**Decision point, left open deliberately:** whether to bump `CALIBRATION_SCALE` (try 1.15-1.25)
to push from 9.4% toward the center of the 10-15% band, or leave it as-is since 9.4% is
already close to Kodiak/HFMA's cited ~11.8% industry-average denial rate. Both are defensible;
record whichever is chosen and why once decided.

### `provider_outlier` — two separate explanations now on record

**At 12.7% (Run 1, carrier+outpatient only):** the rule flags *providers* in the top 5th
percentile by claim count OR total paid, but the rate is measured at the *claim* level. These
diverge because claim-volume is itself one of the flagging criteria: a provider flagged partly
*for* being high-volume then contributes disproportionately many claim-level rows.

**Jump to 20.0% after adding DME (Run 2/3):** expected, not a new bug. The percentile cutoff is
computed *within the current claim population* — adding 103,828 DME claims changed who's in that
population and how volume is distributed across providers, which shifts the 95th-percentile
threshold itself. Revisit the percentile cutoff only if this proves too high relative to real
audit-flag rates once the model is built; not a code fix, a calibration question for later.

### Split logic — confirmed correct as designed, no change made

Reviewed `build_target_and_split.py`'s split logic against the question "why not just check if a
time-based split is possible, and fall back to random if not" — that's exactly what the existing
code already does (checks whether `CLM_FROM_DT` parses for >90% of rows before choosing a branch).
The real run against real data took the time-based branch, confirming dates were usable. No code
change needed; this was a design-review confirmation, not a bug.

---

## Addendum: Discovery of Risk Factor 5 — `deprecated_code` (2026-09-12)

This is the most interesting finding of the whole build so far, and it wasn't planned — it was
discovered by chasing down a diagnostic question about `duplicate_claim`'s near-zero rate. Keeping
the full trail here because "tell me about a time you found something you weren't looking for" is
a real interview question this answers well.

### The investigation trail

1. `duplicate_claim`'s real-data activation rate was 0.057% (982 flagged rows). Inspecting the
   flagged rows directly showed **968 of 982 (98.6%) shared a single HCPCS code: `99241`**, with
   group sizes mostly of 2 (some 3-4).
2. First hypothesis: maybe `99241` is just a very common code, and duplicate-style collisions
   scale with volume. **Ruled out** — `99241` (96,002 total claims) is not even the most common
   code in the dataset; `90935` (222,786 claims, over 2x the volume) showed a 0% duplicate-flag
   rate.
3. Second, corrected hypothesis, controlling properly for claim structure: compared `99241`
   against three other high-volume codes on *distinct-`CLM_ID` flagging* (matching
   `rule_duplicate_claim`'s actual logic, not a naive row-count check, which was tried first and
   gave misleading results for multi-line codes like `G0444`/`99408` — see below).
   - `99241`: 96,002 claims, avg 1.00 lines/claim (single-line), **1.008% flagged**
   - `90935`: 222,786 claims, avg 1.00 lines/claim (single-line), **0.000% flagged**
   - `G0444`: 166,324 claims, avg 1.98 lines/claim (multi-line — same `CLM_ID` shared across
     lines), 0.000% flagged (correctly not counted as duplicate, since `rule_duplicate_claim`
     checks *distinct* `CLM_ID`s, and a multi-line claim only has one)
   - `99408`: 94,609 claims, avg 1.49 lines/claim (multi-line), 0.000% flagged, same reason
4. With claim structure matched (single-line vs. single-line: `99241` vs. `90935`), `99241` was
   the only one that duplicated at all, consistently, at ~1%. That's specific to the code itself,
   not volume or multi-line billing structure.

### The real-world explanation

Searched for what's special about CPT code 99241. Effective January 1, 2010, CMS stopped
recognizing CPT consultation codes — office/outpatient codes 99241-99245 and inpatient codes
99251-99255 — for Medicare Part B payment (CMS Transmittal 1875 / MLN Matters MM6740; codified in
the Medicare Claims Processing Manual, IOM Publication 100-04, Chapter 12, Section 30.6.10).
Physicians were instructed to bill standard evaluation-and-management codes instead. Code `99241`
was further **deleted from the CPT code set entirely**, effective January 1, 2021 (code `99251`
was similarly deleted effective 2023).

This project's data spans 2015-2023 (per the beneficiary files), meaning **every one of the
96,002 claims billed with `99241` in this dataset falls within a period where Medicare either
would not pay for it (2015-2020) or the code should not have existed as a valid submission at all
(2021-2023).** This is almost certainly a Synthea code-generation quirk — its consultation-type
encounter logic appears to reference a CPT table that doesn't reflect Medicare's actual
payment-recognition status for these codes — not a real-world billing pattern, and not
representative of anything a real payer's system would actually receive at meaningful volume from
Medicare-participating providers.

### Decision: built as a new risk factor, not just documented as a limitation

Given how unusually well-grounded this is — an exact, dated federal policy rather than a marginal
survey estimate, the same category of evidence quality none of the other four factors have —
**Risk Factor 5, `deprecated_code`, was added** rather than only noting it as a `duplicate_claim`
caveat. This is a genuine scope decision, not something added automatically just because a factor
was plausible (see Addendum #4's "why not implement every plausible factor" argument, which this
factor is the deliberate, justified exception to).

**Implementation** (`rule_deprecated_code()` in `src/denial_rules.py`): flags any claim with
`HCPCS_CD` in `{99241-99245, 99251-99255}` where `CLM_FROM_DT >= 2010-01-01`. The date check is
kept explicit even though this project's data is entirely post-2010 (so it always evaluates true
here) — the rule should state its actual real-world condition, not bake an assumption silently
into which codes are listed.

**`base_prob = 0.85`** — the highest of any factor in `REASON_CATALOG`, deliberately: this is the
one rule anchored to an exact policy rather than a self-reported survey share, so it's given
correspondingly higher confidence, while staying under the global `0.95` clamp to avoid a fully
deterministic single-feature relationship (the same reason `is_denied` is probabilistic at all —
see the "Label mechanism" section above).

**Priority order:** placed FIRST in `REASON_PRIORITY` (see Addendum #3) — code-validity/
recognition is a harder, more upfront system check than duplicate detection: a claims system needs
a currently-recognized, payable code before it's even meaningful to check whether that code has
been billed twice.

**CARC code:** `181` — "Procedure code was invalid on the date of service." This is the correct
real CARC for this exact scenario, not a repurposed code from elsewhere in the catalog.

**Effect on `duplicate_claim`'s own story:** this resolves Addendum #4's open caveat. The 0.057%
rate wasn't primarily catching real duplicate billing OR the anticipated false-positive pattern
(recurring legitimate services like dialysis/PT) — it was catching a narrower, more specific
issue: one deprecated code that happens to also occasionally get billed twice. `duplicate_claim`
remains implemented as-is; its low real-data activation rate is now fully explained rather than
an open question.

---

## Addendum: Real HCPCS mapping for `dx_procedure_mismatch` (2026-09-12)

`dx_procedure_mismatch`'s original `PROCEDURE_TO_EXPECTED_DX_PREFIX` used 3 illustrative
placeholder codes (`93000`, `71046`, `80053`) that turned out not to appear in this project's real
data at any meaningful volume — which is exactly why the rule fired at only 0.018% against real
claims (see the Phase 1 real-data run log addendum above). Rebuilt from the actual top-30 HCPCS
codes in this project's data (`df["HCPCS_CD"].value_counts()`), with each mapping verified against
real Medicare coverage/billing sources — not guessed, same discipline as every other rule in this
file.

### Verified mappings

| HCPCS code | Description | Required diagnosis | Source |
|---|---|---|---|
| `G0444` | Annual depression screening | `Z13.31` (preferred since Oct 2021), `Z13.39`, `Z13.89`, `Z00.00` | AAPC/CodingIntel/Aetna depression-screening billing guidance |
| `G0442` | Annual alcohol misuse screening | Same Z-code billing pattern as `G0444` | AAPC forum discussion confirming identical G0438/G0439-bundling and diagnosis behavior |
| `96127` | Brief emotional/behavioral assessment | `Z13.31`/`Z13.39`/`Z13.89` if screening; F-code (mental/behavioral chapter) if a positive result is coded | Connected Mind behavioral-health billing guide |
| `90935` | Hemodialysis, single physician evaluation | `N18.x` (ESRD/CKD) required; `I12.x`/`I13.x` (hypertensive CKD) or `E11.22` (diabetic nephropathy) as legitimate comorbid alternates | OmniMD / Coding Clarified / Quest National Services nephrology billing references |
| `A7030`, `A7031`, `A7034`, `A7035`, `A7037`, `A7038`, `A4604`, `E0601` | CPAP mask/cushion/interface/headgear/tubing/filter/heated-tubing/device | `G47.33` (obstructive sleep apnea) — near-universal requirement | CMS LCD L33718 + companion Policy Article A52467; multiple independent DME billing sources confirm `G47.30` (unspecified) is a documented common cause of denial for these specific codes |

That's 12 of the real top-30 codes now covered with cited, verified mappings — a meaningful
upgrade from 3 codes that weren't even present in the data.

### Deliberately left unmapped — not guessed

The remaining high-volume codes in the top 30 (`96156`, `99408`, `99495`, `99401`, `M1069`,
`99397`, `A4253`, `A4259`, `94010`, `45378`, `G8839`, `G9573`, `G9572`, `88155`, `H2001`, `H2010`,
and others) were **not** added to the mapping. Two different reasons, worth distinguishing:

- **Genuinely too broad for a chapter-level check** — e.g. `99495` (transitional care management)
  can legitimately follow almost any admitting diagnosis; there's no single expected ICD-10
  chapter to check against without turning this into a much more elaborate, code-specific medical
  necessity engine than this rule is scoped to be.
- **Not independently verified** — several of these (`96156`, `99408`, `99401`) plausibly follow a
  similar Z-code/F-code screening pattern to the codes that were verified, but that similarity
  wasn't confirmed against a real source in this pass, so they're left out rather than extended by
  analogy. `M1069` in particular doesn't match standard CPT/HCPCS numbering conventions and wasn't
  identified with confidence — left out entirely rather than guessed at.

**The standing rule for extending this table further:** add a code only when you can cite where
its required-diagnosis logic is coming from, the same bar every other mapping here (and every
other rule in this file) was held to — extending by analogy or plausibility, without a source, is
exactly the kind of unverified assumption this project has been careful to avoid elsewhere.

### Verification of the mapping's own accuracy, before trusting the activation rate

Before trusting the jump this mapping produced, the actual `PRNCPAL_DGNS_CD` distribution was
checked directly against three of the mapped codes on real data:

```
G0444: 166,324 claims -- PRNCPAL_DGNS_CD first-char distribution:
  Z 45.4%, T 10.3%, E 7.6%, J 7.5%, N 6.3%

90935: 222,786 claims:
  N 96.1%, J 0.9%, Z 0.8%, E 0.6%, D 0.6%

A7038: 16,295 claims:
  Z 65.7%, T 11.7%, N 3.4%, E 2.8%, G 2.7%
```

**Interpretation:** `90935` (hemodialysis) matches its expected `N` prefix 96.1% of the time —
Synthea appears to correctly pair this code with an ESRD/CKD diagnosis, consistent with real
billing convention. `G0444` and `A7038`, by contrast, match their expected prefix (`Z`, `G`
respectively) only 45.4% and 2.7% of the time — Synthea does **not** reliably enforce the
screening/DME-specific diagnosis pairing real billing rules require for these codes. This is a
genuine, useful finding about the synthetic data's fidelity, not a rule bug: `rule_dx_procedure_mismatch`
is working exactly as designed against real Medicare billing requirements — it's Synthea's
generation logic for these specific code families that doesn't fully replicate real-world coding
discipline. Worth a sentence in the Phase 5 report's limitations section.

### Status

**Implemented, pulled, and confirmed working on real data** — `PROCEDURE_TO_EXPECTED_DX_PREFIX` in
`src/denial_rules.py`. A pull-lag diagnostic is worth recording too: the mapping code was pushed
well before it was actually pulled locally, so an intermediate `build_target_and_split.py` run
(Run 2 above) executed against the *old* 3-code placeholder without anyone realizing it until the
activation rate (0.017%, essentially unchanged from before) didn't match what the verified
mismatch rates above would predict. Confirmed via `git log --oneline -- src/denial_rules.py`
showing the mapping commit missing locally, then resolved with a fresh `git pull`. Lesson: when a
number doesn't move the way a change should predict, check whether the change is actually running
before assuming a logic bug.

---

## Addendum: Phase 1 Step 4 EDA results (2026-09-13)

First full run of `run_eda.py` against real data, including the 5 required figures plus 5
supplementary ones added specifically to answer "what does Phase 2 need to know" (not just satisfy
the checklist) and the long-open native-payment validation check (Addendum #5).

### Categorical cardinality — direct input to Phase 2 encoding strategy

```
HCPCS_CD:        144 unique values; top 20 cover 93.4% of claims
PRVDR_NUM:     8,460 unique values; top 20 cover  4.8% of claims
PRNCPAL_DGNS_CD: 277 unique values; top 20 cover 79.2% of claims
```

`HCPCS_CD` and `PRNCPAL_DGNS_CD` are concentrated enough for one-hot encoding + an "other" bucket.
`PRVDR_NUM` is a genuine long tail (top 20 of 8,460 covers under 5% of claims) — one-hot encoding
this would explode the feature space for almost no per-category signal. **Decision for Phase 2:
use frequency or target encoding for `PRVDR_NUM` specifically**, one-hot (+ other-bucket) for
`HCPCS_CD`/`PRNCPAL_DGNS_CD`.

### 77 numeric columns found — needs an inventory pass before Phase 2

`numeric_feature_distributions()` found 77 numeric columns surviving the leakage drop, far more
than expected. CMS RIF files carry many overlapping dollar/quantity fields (deductible, coinsurance,
line-level vs. claim-level payment amounts); several are likely near-duplicates or mostly
zero/null. **Action item: run `.describe()` across all 77 before Phase 2 feature selection** —
don't feed all 77 into the baseline model without first checking which ones carry real variance
and which are structurally redundant or empty.

### Risk-factor lift — mostly validates the pipeline; the one anomaly is now fully resolved

```
factor                      active_rate   inactive_rate
risk_deprecated_code           95.2%          4.6%
risk_duplicate_claim           97.6%          9.4%
risk_missing_prior_auth        30.8%          9.4%
risk_dx_procedure_mismatch     23.6%          7.5%
risk_provider_outlier          25.3%          5.4%
```

`missing_prior_auth` (30.8% observed vs. `base_prob=0.22 × CALIBRATION_SCALE=1.4` ≈ 30.8%
predicted-in-isolation) and `dx_procedure_mismatch` (23.6% vs. ≈21% predicted) land close to what
their calibrated probability alone would produce — a good sign the noisy-OR mechanics are working
as designed.

`duplicate_claim` at 97.6% sat far above its own ≈49% isolated prediction (`base_prob=0.35 ×
CALIBRATION_SCALE=1.4`, clamped at 0.95). **Resolved 2026-09-13, not just hypothesized:** of the
982 `duplicate_claim`-flagged claims, **98.6% are also flagged `deprecated_code`**. This is the
same claim population Addendum #6 already identified — a `99241` claim billed twice trips
`duplicate_claim` on the pair, and both copies are also Medicare-non-payable consultation codes,
tripping `deprecated_code` independently. The two rules aren't providing two pieces of independent
evidence toward the same conclusion; they're both firing on the same underlying data artifact.
Combining `deprecated_code` (0.85) and `duplicate_claim`'s scaled probability (≈0.49) via noisy-OR
predicts `1 − (1−0.85)(1−0.49) ≈ 92%` denial probability for a claim with both active — close to
the 97.6% observed, with the remaining gap explained by `provider_outlier` also co-occurring on
44.6% of these same claims. The correlation matrix (`risk_factor_cooccurrence.png` /
`labeled_claims_for_eda.parquet`) shows this pair's Pearson correlation at only 0.097, which
understates how tightly linked they are in practice — Pearson correlation on very rare binary
flags (`duplicate_claim` fires on ~0.05% of claims) is diluted by the base rates and is a much
weaker signal here than direct conditional overlap (the 98.6% figure). **Worth noting for future
diagnostics: when investigating rare-flag co-occurrence, compute the conditional overlap rate
directly rather than relying on the correlation coefficient alone.**

### Native-payment validation — CLOSES Addendum #5's open action item, with a real correction

```
Overall native near-zero-payment rate (< $1.00): 5.9%

risk factor             active rate    near-zero | active    near-zero | inactive
deprecated_code               5.3%           0.0%                  6.2%
duplicate_claim                0.1%           1.4%                  5.9%
missing_prior_auth              0.1%          95.8%                  5.8%
dx_procedure_mismatch          11.6%          27.3%                  3.1%
provider_outlier               20.0%           6.6%                  5.7%
```

**`deprecated_code`: 0.0% near-zero-pay when active (below the 6.2% baseline) — confirms Addendum
#6's finding.** Synthea pays these consultation-code claims completely normally, with zero
awareness they're Medicare-non-payable. Expected result, strong confirming evidence.

**`missing_prior_auth`: 95.8% near-zero-pay when active, vs. 5.8% baseline — a real, large
correlation, and it directly contradicted the "no strong reason to expect" reasoning in Addendum
#5's original text (now corrected there). Investigated and fully explained, 2026-09-13, not left
as an open hypothesis.** Pulled the 1,436 flagged claims directly and sorted by `CLM_PMT_AMT`: the
HCPCS codes involved are `E0260`/`E0261` (hospital beds), `K0001`-`K0004` (wheelchairs, standard
through heavy-duty/custom), and `E1038` (transport chair) — Medicare's **capped-rental DME**
equipment category, billed monthly over a rental period rather than paid in full upfront. The
payment distribution matches that structure exactly: 75% of flagged claims show a native
`CLM_PMT_AMT` of exactly $0.00 (median $0, mean pulled to ~$1.95 only by a handful of claims up to
$90.18). Capped-rental billing genuinely produces $0-paid lines as a normal, structural feature —
administrative/setup lines or specific months within the rental cycle can legitimately show no
payment without the claim being denied. So `missing_prior_auth`'s proxy (E/K HCPCS codes with no
recent prior claim) happens to heavily overlap with this specific DME billing pattern, and that
billing pattern has its own independent, well-documented reason for showing frequent $0 native
payment — unrelated to whether the claim represents genuine denial risk. This confirms the
hypothesis from the first pass rather than surfacing a bug or a hidden real signal.

`dx_procedure_mismatch` shows a moderate real gap (27.3% vs. 3.1%) worth a passing mention.
`duplicate_claim` and `provider_outlier` show no meaningful gap, as expected.

**Net effect on the original design argument:** the core conclusion — `CLM_PMT_AMT` can't be used
as the label, either as a heuristic or as the target itself — still holds regardless of this
finding, since the reasoning in Addendum #5(a)/(b) about deductible absorption and bundling
conflating with true denial doesn't depend on whether native payment happens to correlate with any
specific risk factor. What changes is the *supporting* claim about *why* no correlation was
expected — that turned out to be right for one factor (`deprecated_code`) and wrong for another
(`missing_prior_auth`, now explained by capped-rental DME billing structure), and both outcomes,
plus the explanation for the miss, are on record rather than only the confirming one.