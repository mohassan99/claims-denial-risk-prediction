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

**NPI *values* — considered, then dropped. NPI *presence/absence* — a separate decision, kept.**
Initially proposed keeping and encoding the raw NPI values (frequency encoding, since raw NPI has
no repeat structure to one-hot). Retracted once confirmed `PRVDR_NUM` already captures the
provider-level grouping information these fields would add, at lower cardinality and with real
repeat structure `provider_outlier`/`dx_procedure_mismatch` already use. **But this only argues
against the raw NPI *value* — whether a given NPI role-field is populated at all is a separate,
genuinely useful question that got conflated with it initially.** Whether `RFR_PHYSN_NPI` is
populated plausibly indicates a referral-based encounter (different coordination/prior-auth
pattern than a direct visit); whether `OP_PHYSN_NPI` is populated indicates a procedural/surgical
claim versus not. **Decision: drop the raw NPI values (all 6 fields), but keep a binary presence
flag per role** (`has_referring_physician`, `has_operating_physician`, etc.) as its own engineered
feature, since the *fact* of population carries information the *value* doesn't.

**Can NPI be rolled up to specialty/sub-specialty instead? Considered and rejected, precisely.** In
the real world this crosswalk exists — NPPES (the actual national NPI registry) publishes a
taxonomy/specialty code per real NPI. Two things break it for this project: (1) these are
Synthea-generated synthetic NPIs with no reason to resolve to real NPPES records — a join against
the real registry would fail or return meaningless matches; (2) even if it worked, it would be
redundant, since `PRVDR_SPCLTY` already exists natively in the carrier file as a direct, more
reliable field. Not pursued.

**`PRVDR_ZIP` — considered, corrected, replaced.** Initially proposed binning to ZIP3 to capture
regional Medicare Administrative Contractor (MAC) jurisdiction effects — MACs administer different
LCDs and can have genuinely different denial patterns across jurisdictions, so this wasn't an
implausible hypothesis. **Corrected once:** MAC jurisdiction is assigned by *state*, not ZIP code,
so ZIP3 would be a noisier, redundant proxy for information `PRVDR_STATE_CD` already carries
directly (already string-typed from the earlier `_CD` fix). `PRVDR_ZIP` dropped in favor of
`PRVDR_STATE_CD`.

**Corrected a second time — the whole premise was unverified.** The MAC-jurisdiction reasoning
implicitly assumed this dataset spans multiple states/jurisdictions the way a real national payer
dataset would. That was checked directly against the actual CMS user guide for this dataset
(which includes a "Beneficiaries Per State Code" figure, confirming multi-state coverage by
design) — so the premise holds at the dataset level. **A separate, real data-quality issue
surfaced during this check, though:** `PRVDR_STATE_CD`'s `value_counts()` on real data showed a mix
of alpha USPS codes (`CA`, `FL`, `NY`) and numeric codes (`05`, `09`, `53`) for what are very
plausibly the *same* states represented two different ways. Confirmed against the official CMS/CCW
code system spec: `PRVDR_STATE_CD` is documented as a two-digit **numeric** SSA state code
(`05 = California`, `09 = District of Columbia`, `53 = Wyoming`, etc.) — the alpha values in this
dataset are not part of the official spec, and are almost certainly the same states double-counted
across two coding conventions (likely inconsistent handling across the carrier/outpatient/DME RIF
layouts during Synthea's generation). **Action item before this field is usable for anything,
MAC-jurisdiction or otherwise:** normalize every value through the official SSA-numeric-to-alpha
crosswalk before re-running `value_counts()` to see the true state distribution — using the raw,
un-normalized column would silently understate concentration for any state affected by the
dual-coding bug.

**Precise expectation for the post-normalization unique-value count, corrected before it was
asserted as fact:** the shorthand "50 states + DC = 51 jurisdictions" was floated as a sanity check
(observed 102 raw unique values, hypothesized as 51 jurisdictions × 2 codings = 102). That
shorthand undercounts what the official SSA crosswalk actually covers — it includes **Puerto Rico**
(`40`) and the **Virgin Islands** (`48`) in addition to the 50 states and DC, i.e. 53 real domestic
jurisdictions, not 51. There's also no guarantee every jurisdiction is duplicated in both coding
conventions in this specific dataset — some may appear in only one. **Don't assert a specific
expected post-fix count in advance; run the crosswalk and read off the actual number**, the same
verify-before-asserting discipline applied everywhere else in this project.

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

**Three approaches were considered. The evaluation of several points, including some terminology,
changed substantially under challenge during this discussion — those changes are kept below, not
smoothed over.**

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
for each claim-type-specific field, include both an applicability indicator and the interaction
`value × applicable`, with the raw value zero-filled where not applicable. When not applicable,
both terms vanish (contribute nothing to the log-odds); when applicable, the interaction term
reduces to a normal linear effect and the indicator absorbs any baseline shift for "this claim type
doesn't have this field."

This is the textbook "missing indicator method." It's genuinely controversial and often
discouraged for *ordinary* missing data, where the missingness mechanism is uncertain or itself
informative in unclear ways — but it's specifically well-justified for **deterministic, structural
missingness**, which is exactly this case (claim type determines applicability with 100%
certainty). This isn't a compromise or a simplification; it's the correct tool for this specific
situation. **See the corrected implementation note under "`claim_type` feature" below — the
applicability indicator and the `claim_type` dummy are the SAME piece of information and must not
both be included as separate terms.**

### Approach 3 — Stratified regression: separate logistic regression per claim type

**Correction on the name itself:** this was originally called "fully stratified" and its
evaluation-combination step was described as "segmented/mixture-of-experts scoring." The correct
standard statistical term for the modeling approach is **stratified regression** (or "stratified
modeling" / "subgroup-specific models") — fitting a separate model per subgroup defined by an
observed categorical variable. "Mixture of experts" was imprecise for the scoring step too: that
term properly refers to a *learned, soft* gating function deciding how much weight each sub-model
gets, whereas routing here is *hard and deterministic* — `claim_type` is already known for every
claim, no learned gating involved. The more accurate description is "stratified models with
deterministic routing."

**Initially dismissed as "too noisy" given DME is only 5.8% of the data — this claim was wrong and
is corrected here, not quietly dropped.** Checked against actual numbers rather than intuition:
5.8% of ~1.15M training rows ≈ 66,700 DME claims; at the ~9.4% overall denial rate, that's roughly
6,270 denial events in the DME subset alone. Standard practice for logistic regression reliability
uses an events-per-variable (EPV) rule of thumb of ≥10 — with ~6,270 events, the DME subset alone
could reliably support **hundreds** of predictors, far more than would realistically be used. The
original "too small/noisy" claim does not survive contact with the actual arithmetic.

**The "efficiency vs. correctness" framing of the tradeoff was also wrong, and this is a more
substantive correction than the arithmetic one above.** The original framing said the cost of
stratifying was "recomputing shared effects three times on smaller subsets" — that's a
runtime/effort cost, not a statistical one, and it understated what's actually at stake. The
precise version: **a fully claim-type-interacted pooled model and three separately-fit stratified
models are mathematically identical** — same fit, same coefficients, same predictions, just
organized differently in code. So "pooled vs. separate" is not itself a bias/efficiency tradeoff
when both are fully saturated with interactions.

The real tradeoff is **per-variable, not per-modeling-strategy**: does a given shared field's
effect on denial actually vary by claim type, or not? Take `dx_procedure_mismatch` specifically —
it's entirely plausible that a mismatch means something different in a DME billing context than in
a carrier consultation context, in which case forcing one shared coefficient across all three claim
types (as Approach 2 does by default for any variable *not* given its own claim-type interaction)
is not merely inefficient, it's **model misspecification** — a real bias, potentially even
producing the wrong sign in the pooled estimate (the kind of aggregation distortion Simpson's
paradox describes). **The scientifically correct practice is not to pick one blanket strategy for
every variable, but to test the assumption per shared variable** — fit the claim-type interaction
for a candidate variable, run a likelihood-ratio test (or check the interaction term's
significance / compare AIC) against the version without it, and only impose a shared coefficient
where that assumption survives the test. Variables that pass get one shared coefficient; variables
that fail get their own per-claim-type coefficient, structurally no different from what full
stratification would have given them anyway.

**A note on precisely what "efficient" means for the variables that pass — a distinct, technical
sense of the word, not the runtime/effort sense retracted above.** When a shared coefficient is
statistically justified (the interaction test doesn't support letting the effect vary), estimating
it as one pooled parameter is more *statistically* efficient: it draws on the full training set
(~1.15M rows) rather than only one claim type's subset, so its standard error is smaller — a lower-
variance estimate of the same true effect, in the formal sense tied to the Cramér-Rao bound on
minimum achievable estimator variance. This is a real, separate benefit from anything about human
effort or code runtime, and it only applies *when the homogeneity assumption is actually correct*
for that variable — if the test fails, there is no efficiency benefit to weigh against anything;
pooling would just be wrong (misspecified), not efficient-but-biased.

**Combining three claim-type-specific models into one PR-AUC is standard, defensible evaluation
practice, not a hack.** Route every held-out test claim to its matching claim-type model (hard,
deterministic routing on the known `claim_type`, not a learned gate), pool all resulting
probability scores and true labels together, and compute PR-AUC over that pooled set exactly as
for a single model. Stratified models with deterministic routing are evaluated this way routinely
in production risk models (insurance underwriting, credit scoring) — nothing methodologically
irregular about it.

### Decision — left open, intentionally, as a real design choice for Phase 2 rather than resolved
here

**Revised recommendation, given the correction above:** rather than picking Approach 2 or Approach
3 as a blanket strategy, **test each shared (non-claim-type-exclusive) variable for a claim-type
interaction and let the data decide, variable by variable**, per the likelihood-ratio approach
above. Structurally this is closest to Approach 2 (one pooled model, claim-type-specific fields
handled via the interaction construction from that section) but without assuming homogeneity for
every shared variable by default — each shared variable earns its "single coefficient" status
through a test, not through convenience. Approach 3 (stratified regression) remains the
theoretically cleanest fallback if time doesn't allow per-variable testing, since the sample size
comfortably supports it and it makes no homogeneity assumptions anywhere. Approach 1 is documented
but rejected as the primary path, kept only as a time-pressure fallback of last resort.

**This will be decided when the Phase 2 baseline script is actually built**, with this document
providing the reasoning trail — not decided speculatively now, ahead of writing that code.

### `claim_type` feature — derive from already-computed data; corrected implementation note on
avoiding perfect collinearity

`load_data.py` already tags every row with `_source_file` (the origin CSV name) at concatenation
time — this is free, already-computed claim-type information that hasn't been used yet. **Decision:
map `_source_file` to a clean `claim_type` categorical (`carrier`/`outpatient`/`dme`) for Phase 2**,
rather than having the model infer claim type indirectly from which fields happen to be populated.

**Corrected implementation detail — this was wrong in the first draft and needed fixing, not just
softening.** The original text suggested keeping *both* a set of per-field-group applicability
indicators (`applicable_REV_CNTR`, `applicable_DMERC`, etc.) *and* a separate `claim_type`
main-effect dummy "for interpretability." That's not merely redundant — since each applicability
indicator is a perfect 1-to-1 proxy for one specific claim type (`applicable_REV_CNTR` is 1 if and
only if `claim_type == outpatient`), including a separate `claim_type` dummy alongside it creates
an **exactly rank-deficient design matrix**, not just high correlation. Standard unregularized
logistic regression (Newton-Raphson/IRLS) would hit a singular matrix and fail to converge;
`sklearn`'s default L2-regularized version wouldn't error, but would arbitrarily split the
coefficient between the two redundant terms in a way that's statistically meaningless.

**Corrected design: use `claim_type` dummies as the ONLY encoding of "which claim type" — do not
also create separate per-field-group applicability indicators.** Build each claim-type-specific
field's interaction directly against the relevant `claim_type` dummy (e.g.
`REV_CNTR_TOT_CHRG_AMT × claim_type_outpatient`, value zero-filled where not applicable) rather
than inventing a redundant `applicable_REV_CNTR` term alongside it — the `claim_type` dummy already
*is* the applicability indicator for that field group; no separate one is needed or valid to add.
Under Approach 3, `claim_type` is instead the literal routing key used to split the training data
into the three per-claim-type subsets — no collinearity risk there, since each subset only ever
sees its own claim type and the variable never appears as a predictor inside any single subset's
model.

### `claim_type`'s encoding scheme — nominal, not ordinal, and a specific dummy-coding variant is
required, not just any one-hot scheme

**Nominal, not ordinal.** Carrier/outpatient/DME have no real ordering, so a single integer-coded
variable (`0`/`1`/`2`) would be wrong for a linear model — it silently imposes a false
interval-scale assumption (that DME is "twice as far" from carrier as outpatient is), which is a
meaningless claim about billing-category membership. One-hot (nominal) encoding is the correct
family of choices; the question is which specific *variant* of one-hot encoding.

**Standard one-hot (`k−1` dummies + intercept, "reference-cell coding") isn't quite right either —
it silently breaks the interaction construction for whichever category is dropped.** If 3
categories are one-hot encoded but the model keeps its usual shared intercept, the 3 dummies sum to
exactly `1` for every row — a perfect linear dependency on the intercept (the "dummy variable
trap"), so standard practice drops one category as the reference. But that's specifically
incompatible with what this project needs the dummies to do: if `outpatient` is the dropped
reference category, its effect gets absorbed into the intercept, and there's no explicit
`claim_type_outpatient` dummy left to interact `REV_CNTR_*` fields against.

**The correct scheme: cell-means coding — all `k` dummies, no shared intercept.** With the
intercept dropped and all 3 dummies kept, there's no dummy-variable-trap collinearity (nothing for
the 3 dummies to be collinear against), and every claim type has its own explicit `0/1` dummy to
interact its exclusive fields against, with none singled out as an implicit reference.

**A definitional correction worth being precise about, since it's easy to conflate with a different
scheme:** cell-means coding does **not** mean "each coefficient represents that group's deviation
from the overall mean." That description is actually **effect coding** (also called deviation or
sum-to-zero contrast coding), which uses `−1/0/1`-style contrasts specifically constructed so
coefficients sum to zero and each represents a difference from the grand mean — a different,
distinct scheme. Under cell-means coding, each dummy's coefficient **is that group's own value
directly** (its own log-odds baseline), not a difference from anything — there's no subtraction
happening at all.

**Why cell-means specifically is required here, not a preference — two independent reasons, both
pointing to the same answer:**
1. **Need an explicit `0/1` dummy for every one of the 3 claim types**, not `k−1` — reference-cell
   coding drops exactly one, breaking the interaction construction for whichever claim type lost
   its dummy (the reason established above).
2. **The dummy has to be plain `0/1`, not effect coding's `−1/0/1`.** The interaction construction
   depends on `value × dummy` cleanly zeroing out for claims where a field doesn't apply — that
   only works if "not this claim type" is coded as `0`. Effect coding assigns at least one category
   `−1` for some contrasts, which would flip the sign of a zero-filled value rather than zero it
   out, corrupting the interaction term instead of correctly suppressing it.

Both constraints are satisfied only by full-`k`, `0/1`, no-intercept cell-means coding — not chosen
for interpretability preference, but because it's the only one of the three standard schemes
(reference-cell, effect, cell-means) that works at all given the interaction-based construction
these dummies are being built for. **Conclusion for the earlier question: no separate `claim_type`
variable beyond the dummy set itself is needed — each dummy's own coefficient *is* that claim
type's main effect — but the specific dummy-coding variant used matters, and must be cell-means,
not the more commonly-defaulted reference-cell scheme most one-hot encoders produce out of the
box** (e.g. `pandas.get_dummies(..., drop_first=True)` and `sklearn`'s `OneHotEncoder(drop="first")`
both default to reference-cell coding — the `drop_first`/`drop` argument needs to be left off, and
the model's own intercept term needs to be suppressed, to get cell-means coding instead).

**This entire section's decisions assume Approach 2 (or the per-variable-tested hybrid
recommended above) is what's ultimately implemented.** Under pure Approach 3 (stratified
regression), `claim_type` never appears as a predictor inside any single subset's model at all —
it's purely the routing key used to split the data beforehand, and none of the cell-means/dummy
discussion applies there.

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

**Implementation note: these decisions belong in saved, versioned pipeline code, not repeated ad
hoc terminal checks.** Diagnostic one-off checks (e.g. `.value_counts()` to confirm a hypothesis)
are appropriately run interactively and discarded. The decisions below are different — they're
permanent transformations that must apply identically to train/val/test every time the pipeline
runs. **Decision: consolidate all of them into one new script, `src/build_features.py`**, run
after `build_target_and_split.py`, rather than reconstructing each fix by hand per session.

**Implication:** the remaining items below are Phase 1 completion work for that script, not scope
creep:

- Derive `claim_type` from `_source_file`, encoded as cell-means (full `k`-dummy, `0/1`,
  no-intercept) one-hot, not reference-cell one-hot and not integer/ordinal coding
- Drop raw NPI values (all 6 fields) and `PRVDR_ZIP`; retain binary presence flags for the 5 NPI
  role-fields (`has_referring_physician`, `has_operating_physician`, etc.) as their own features
- Normalize `PRVDR_STATE_CD` through the official SSA-numeric-to-alpha crosswalk (confirmed
  necessary — the raw column mixes both conventions for the same states; the crosswalk covers 53
  jurisdictions — 50 states + DC + Puerto Rico + Virgin Islands — not 51, so don't assume a
  specific post-normalization unique-value count in advance) before using it for any
  MAC-jurisdiction or regional feature
- Drop the always-100%-null columns listed in Section 2
- Confirm (or refute) the `PRVDR_SPCLTY`/`ICD_DGNS_VRSN_CD*` zero-variance suspicion
- Decide the claim-type-specific field handling strategy for Phase 2's baseline logistic
  regression via the per-variable interaction-testing approach in Section 3, not a single blanket
  strategy chosen in advance
- If using the interaction/missing-indicator construction, use `claim_type` dummies as the sole
  "which claim type" encoding — do not also add separate per-field-group applicability indicators