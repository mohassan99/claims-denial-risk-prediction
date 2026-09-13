# Feature Engineering & Data Cleaning Decisions

Companion to `TARGET_DEFINITION.md` (which covers the label, `is_denied`). This file covers
decisions about the *features* — how columns are typed, which are dropped, and how
claim-type-specific structural missingness is handled going into modeling.

Written in the same spirit as `TARGET_DEFINITION.md`: kept in full, including the wrong turns and
corrections, not cleaned up to look like the right answer was obvious from the start. Several of
the positions below were argued for, then retracted under challenge, in the same conversation —
that trail is preserved deliberately, since "here's a claim I made, here's why it was wrong, here's
the corrected version" is a stronger interview answer than a single polished conclusion.

---

## 1. Identifier columns miscategorized as numeric

**What happened:** the `_CD`/`_NUM` suffix-based dtype-forcing fix (see `TARGET_DEFINITION.md`'s
Phase 1 real-data run log, Bug 1) generalized well for CMS code columns, but missed a real
category: physician/provider identifier fields ending in `_NPI` (`RFR_PHYSN_NPI`, `PRF_PHYSN_NPI`,
`AT_PHYSN_NPI`, `OP_PHYSN_NPI`, `RNDRNG_PHYSN_NPI`, `PRVDR_NPI`) and `PRVDR_ZIP`. These were left
to pandas' default numeric inference, so `train.parquet`'s `.describe()` output showed nonsensical
statistics for them (e.g. a mean NPI value), the same class of problem the `_CD`/`_NUM` rule was
built to prevent, just for a suffix pattern that rule didn't anticipate.

**Decision: re-type as string/categorical in Phase 2 feature-prep code, not by re-running
`load_data.py` a fourth time.** Regenerating the whole pipeline again for a typing fix that doesn't
change any actual values (just how they're read) wasn't worth the time cost — cast these
explicitly to `str` when building the Phase 2 feature matrix instead. Documented here as a known
gap in the "generalized" `_CD`/`_NUM` rule, since it wasn't actually fully general.

**NPI fields — considered, then dropped.** Initially proposed keeping and encoding these
(frequency encoding, since raw NPI has no repeat structure to one-hot). Retracted once confirmed
`PRVDR_NUM` already captures the provider-level grouping information these fields would add, at
lower cardinality and with real repeat structure `provider_outlier`/`dx_procedure_mismatch` already
use. Five NPI role-fields (referring/performing/attending/operating/rendering) pointing at
essentially the same redundancy wasn't worth the added complexity. **All 6 NPI fields dropped from
the Phase 2 feature set.**

**`PRVDR_ZIP` — considered, corrected, replaced.** Initially proposed binning to ZIP3 to capture
regional Medicare Administrative Contractor (MAC) jurisdiction effects — MACs administer different
LCDs and can have genuinely different denial patterns across jurisdictions, so this wasn't an
implausible hypothesis. **Corrected:** MAC jurisdiction is assigned by *state*, not ZIP code, so
ZIP3 would be a noisier, redundant proxy for information `PRVDR_STATE_CD` already carries directly
and correctly (already string-typed from the earlier `_CD` fix). **`PRVDR_ZIP` dropped; use
`PRVDR_STATE_CD` for any regional/MAC-jurisdiction feature instead.**

---

## 2. Structural missingness across claim-type-specific fields

**The finding:** of the 77 numeric columns surviving the leakage drop, most of the non-trivial
nullness lines up almost exactly with claim-type proportions in the data (carrier ~62%, outpatient
~32%, DME ~5.8%). `CARR_CLM_PRMRY_PYR_PD_AMT`-family fields are null for every non-carrier row,
`REV_CNTR_*`-family fields are null for every non-outpatient row, `DMERC_LINE_*`-family fields are
null for every non-DME row. This isn't random missingness or a data-quality problem — it's
structural: these fields don't apply outside their claim type, by construction of the RIF schema
itself.

**Why this is a stronger, cleaner case than ordinary missing data:** in the missing-data
literature, this is *structural missingness* (sometimes "structural zero") — a value is null **if
and only if** claim type doesn't have the concept, fully deterministic, no ambiguity about the
mechanism. That's a materially different (and easier) situation than MNAR (missing-not-at-random)
data where the missingness mechanism itself is uncertain or informative in unclear ways.

**Why imputation is the wrong tool here — a real correction, not a stylistic choice.** Imputation
(mean/median/model-based) is appropriate when a value exists in reality but wasn't recorded — you
are estimating something real but unobserved. That's not this situation: there is no
revenue-center concept for a carrier claim at all. Imputing a mean `REV_CNTR_TOT_CHRG_AMT` for a
carrier row would manufacture a number with no real referent and distort whatever relationship the
model learns. An earlier draft of this reasoning suggested imputation as an option; that was wrong
and is corrected here rather than silently dropped.

**Decision: drop columns that are 100% null across the entire dataset (not claim-type-specific,
genuinely empty everywhere) — keep everything that's claim-type-specific.** The two categories
need different treatment:

- **Always-100%-null, regardless of claim type — DROP:** `LINE_SERVICE_DEDUCTIBLE`,
  `FI_CLM_PROC_DT` (already documented in `TARGET_DEFINITION.md` as blank/fixed in this release),
  `OT_PHYSN_UPIN`, `OT_PHYSN_NPI`, `ICD_PRCDR_CD25`, `PRCDR_DT25`, `RSN_VISIT_CD1`,
  `RSN_VISIT_CD2`, `RSN_VISIT_CD3`, `REV_CNTR_NDC_QTY`. These carry zero information regardless of
  claim type — dropping them loses nothing.
- **Claim-type-specific (partial nullness that matches claim-type proportion) — KEEP, handle per
  Section 3 below, do NOT drop and do NOT impute.** The partial nullness pattern is itself
  informative (which claim type a row belongs to) and dropping these columns would throw away real
  claim-type-specific signal, not just noise.

**Still open, not yet checked:** `PRVDR_SPCLTY` showed a suspicious constant mean of exactly
`1.0` in the `.describe()` output, and `ICD_DGNS_VRSN_CD1`-`CD12` all showed a mean of exactly
`0.0` — both patterns consistent with zero-variance (constant-value) columns, which would make them
droppable too, but this hasn't been directly confirmed with `.value_counts()` yet. **Action item:
run that check before finalizing the drop list.**

---

## 3. Handling claim-type-specific features in the Phase 2 baseline logistic regression

**The actual question, stated precisely (a real clarification, not just rewording):** a categorical
dummy shifts a logistic regression's intercept, and an interaction term lets one continuous
predictor's slope vary by category — both are standard, well-understood tools. But neither answers
the question actually at stake here, which isn't "does this variable's effect differ by claim
type" (interactions handle that fine) — it's "does this variable exist at all for this claim
type." You can't multiply an interaction coefficient against an undefined value; a carrier claim
doesn't have a small or zero `REV_CNTR_TOT_CHRG_AMT`, it has *no* `REV_CNTR_TOT_CHRG_AMT`.

**Three approaches were considered. The evaluation of two of them changed substantially under
challenge during this discussion — both changes are kept below, not smoothed over.**

### Approach 1 — Restrict the baseline to a common feature set (fields present across all claim
types); let XGBoost use everything

**Initially recommended, then retracted as methodologically weak.** The original justification was
"the baseline's job is just to show why gradient boosting is needed." That reasoning is circular:
deliberately impoverishing the baseline's features to make XGBoost look better by comparison isn't
a fair contrast, it's a strawman with a predetermined conclusion. The correct standard is to give
the linear model the best feature set the GLM framework can actually support — if XGBoost still
wins under that fair comparison, that's a real, defensible finding about nonlinearity and
native missing-value handling, not a manufactured one. **Rejected as the primary approach for this
reason**, though it remains available as a fallback if time runs out before Approach 2 or 3 can be
properly implemented.

### Approach 2 — Missing-indicator / applicability interaction, single pooled model

**Reframed more precisely during this discussion than in the first pass at it.** This is not
"missing-indicator plus a separate strategy" — the construction *is* literally an interaction term:
for each claim-type-specific field, include both an `applicable` binary indicator and the
interaction `value × applicable`, with the raw value zero-filled where not applicable. When not
applicable, both terms vanish (contribute nothing to the log-odds); when applicable, the
interaction term reduces to a normal linear effect and the indicator absorbs any baseline shift for
"this claim type doesn't have this field."

This is the textbook "missing indicator method." It's genuinely controversial and often
discouraged for *ordinary* missing data, where the missingness mechanism is uncertain or itself
informative in unclear ways — but it's specifically well-justified for **deterministic, structural
missingness**, which is exactly this case (claim type determines applicability with 100%
certainty). This isn't a compromise or a simplification; it's the correct tool for this specific
situation, applied inside one pooled model that shares statistical power across claim types for
any effect that behaves consistently across them.

### Approach 3 — Fully stratified: separate logistic regression per claim type

**Initially dismissed as "too noisy" given DME is only 5.8% of the data — this claim was wrong and
is corrected here, not quietly dropped.** Checked against actual numbers rather than intuition:
5.8% of ~1.15M training rows ≈ 66,700 DME claims; at the ~9.4% overall denial rate, that's roughly
6,270 denial events in the DME subset alone. Standard practice for logistic regression reliability
uses an events-per-variable (EPV) rule of thumb of ≥10 — with ~6,270 events, the DME subset alone
could reliably support **hundreds** of predictors, far more than would realistically be used. The
original "too small/noisy" claim does not survive contact with the actual arithmetic.

Is this the theoretically best approach, or just an easier-to-explain one? In the
statistics/econometrics literature, when a data-generating process is genuinely structurally
different across subgroups — which this is, given different applicable variables and different
institutional billing mechanisms per claim type, not merely a superficial difference — a fully
stratified model is considered the *most* flexible, least-biased option, not a simplification made
for narrative convenience. The real tradeoff is **statistical efficiency, not correctness**: any
effect that behaves similarly across all three claim types gets independently re-estimated three
times on smaller subsets instead of once on the full pooled data, which is a real, quantifiable
cost, but not a validity problem.

**Combining three claim-type-specific models into one PR-AUC is standard, defensible evaluation
practice, not a hack.** Route every held-out test claim to its matching claim-type model, pool all
resulting probability scores and true labels together, and compute PR-AUC over that pooled set
exactly as for a single model. This is how segmented/mixture-of-experts scoring is routinely
evaluated in production risk models (insurance underwriting, credit scoring) — nothing
methodologically irregular about it.

### Decision — left open, intentionally, as a real design choice for Phase 2 rather than resolved
here

**Approach 2 is the current lean as a default**, since it shares statistical power across claim
types for effects that behave consistently, while still giving claim-type-unique fields their own
genuinely separate coefficients through the interaction structure, all inside one coherent,
easier-to-maintain model. **Approach 3 is an equally legitimate alternative**, not a fallback, if
the goal is letting *every* variable — including the ones shared across claim types — have fully
claim-type-specific effects; the sample size comfortably supports it, and "let each claim type have
its own model" has a real theoretical grounding, not just a practical convenience one. Approach 1
is documented but rejected as the primary path, kept only as a time-pressure fallback.

**This will be decided when the Phase 2 baseline script is actually built**, with this document
providing the reasoning trail — not decided speculatively now, ahead of writing that code.

### `claim_type` feature — derive from already-computed data, not from field-presence patterns

`load_data.py` already tags every row with `_source_file` (the origin CSV name) at concatenation
time — this is free, already-computed claim-type information that hasn't been used yet. **Decision:
map `_source_file` to a clean `claim_type` categorical (`carrier`/`outpatient`/`dme`) for Phase 2**,
rather than having the model infer claim type indirectly from which fields happen to be populated.

Its role differs by which approach from above is chosen: under Approach 2, the collection of
per-field `applicable` indicators already collectively encodes claim type (their 1/0 pattern is
unique per claim type), but an explicit `claim_type` main-effect term is still useful for a single,
cleanly interpretable "baseline shift by claim type" coefficient. Under Approach 3, `claim_type` is
the literal routing key used to split the training data into the three per-claim-type subsets.

---

## 4. Scope correction: feature engineering is Phase 1 work, not Phase 2 spillover

**A real correction, checked against the actual Build Guide rather than assumed.** Partway through
this discussion, this entire line of work was framed as possibly being Phase 2 spillover ("Phase 2
is titled baseline/XGBoost/SHAP, no feature engineering"). That framing was wrong. The Build
Guide's Phase 1, **Step 9 ("Data wrangling")**, explicitly states: *"Engineer claims-domain
features: provider specialty, CPT/HCPCS category, ICD category, prior claim history for the same
member/provider, days-between-service-and-submission, whether prior authorization was on file."*
Feature engineering was Phase 1 scope from the original plan, not something being pulled forward
from Phase 2.

**Implication:** the remaining items below are Phase 1 completion work, not scope creep:

- Derive `claim_type` from `_source_file`
- Re-type NPI fields (drop) and `PRVDR_ZIP` (drop, use `PRVDR_STATE_CD` instead) correctly
- Drop the always-100%-null columns listed in Section 2
- Confirm (or refute) the `PRVDR_SPCLTY`/`ICD_DGNS_VRSN_CD*` zero-variance suspicion
- Decide between Approach 2 and Approach 3 for claim-type-specific field handling (can be finalized
  either now or deferred to the start of Phase 2's baseline script, but the analysis above is
  Phase 1's output either way)