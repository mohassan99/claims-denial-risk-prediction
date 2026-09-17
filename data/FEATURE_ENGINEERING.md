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

**Verified after implementation, 2026-09-13:** running the crosswalk on real data merged the
duplicate codes correctly — `CA` came back at 119,970, exactly matching `78,354 (CA) + 41,616 (05)`
from the original raw `value_counts()`, direct confirmation the fix worked as intended, not just
run without erroring.

**Checked and closed, 2026-09-13.** The full official SSA code range (00-80, not just 01-53) has
documented secondary duplicate codes for states already in the table (`55`=CA again, `67`/`74`=TX
again, `68`/`69`=FL again, `70`=KS again, `71`=LA again, `72`=OH again, `73`=PA again, `80`=MD
again — an artifact of CMS Certification Number exhaustion over time), plus non-US region codes
(`54`, `56`-`66`). Checked directly against `provider_state`'s full `value_counts()` on real data:
**51 unique values, no leftover bare numbers, no territories present in this download.** The
crosswalk is complete for this dataset as-is — confirmed, not just assumed; no extension needed.

**Status: implemented, pushed, and fully verified** — `SSA_STATE_CROSSWALK` + presence-flag/drop
logic in `src/build_features.py`. Nothing further open on this item.

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

**`PRVDR_SPCLTY` and `ICD_DGNS_VRSN_CD1`-`CD12` — confirmed zero-variance via `.value_counts()`,
and their real-world meanings looked up rather than assumed, since a statistical fact alone
doesn't tell you whether a constant is a data limitation or an expected, correct value.** Both
were flagged from `.describe()` (suspicious constant means) and confirmed via `.value_counts()`:
`PRVDR_SPCLTY` is `1.0` for every one of its 784,409 populated rows; every `ICD_DGNS_VRSN_CD`
column is `0.0` for every populated row. The two constants tell genuinely different stories once
their official CMS meanings are checked, not one story:

- **`PRVDR_SPCLTY` — a real, meaningful field, and the constant is a genuine Synthea limitation,
  now verified across its full applicable domain rather than a single claim type.** Confirmed
  against official CMS/CCW documentation: this is "CMS specialty code used for pricing the line
  item service," and `01` (which is what `1.0` represents) specifically means **General Practice**
  — not a null/placeholder code (that would be `00`, "Carrier wide"). Every carrier claim in this
  dataset being coded `01` means Synthea assigns every provider the same specialty regardless of
  what care was actually delivered.

  **First verification attempt was set up wrong, and the failure is worth keeping visible.** The
  initial check filtered by specific procedure codes plausibly requiring specialists — `90935`
  (hemodialysis), `45378` (colonoscopy), `88155` (Pap smear) — expecting either their absence from
  the data or a non-`01` specialty when present. All three came back **100% NaN** on `PRVDR_SPCLTY`
  in `train.parquet`. This looked inconclusive but was actually a test-design error: it also
  surfaced that `PRVDR_SPCLTY`'s 784,409 populated rows had been mischaracterized earlier as
  "carrier-only" — checking again against ResDAC documentation, the field is populated in **both**
  Carrier and DME RIF files (`718,074 + 66,335 = 784,409`, confirmed by direct row counts), not
  carrier alone. The three test procedure codes turned out to route entirely through *outpatient*
  claims in this dataset, where `PRVDR_SPCLTY` doesn't exist at all — the test never touched
  populated data in either direction.

  **Corrected verification: compare `PRVDR_SPCLTY` directly between the two claim types where it
  actually applies.** `carrier.csv`: 718,074 claims, 100% `1.0`. `dme.csv`: 66,335 claims, 100%
  `1.0`. Constant across the *entire* domain where the field is meant to carry information, not
  just one slice of it — a stronger, more thorough confirmation of the Synthea-limitation reading
  than the original (flawed) test would have given even if it had worked as designed. This is the
  same category of limitation as the fixed denial-status fields already documented in the
  "Decision" section of `TARGET_DEFINITION.md` (`CARR_CLM_PMT_DNL_CD`, `CLM_DISP_CD`, etc.) — worth
  a line in the Phase 5 report's limitations section.
- **`ICD_DGNS_VRSN_CD1`-`CD12` — constant, but expected and correct, not a limitation.** This field
  flags whether each diagnosis code is ICD-9 or ICD-10, with `0` meaning ICD-10 (confirmed against
  ResDAC/CMS documentation). The real ICD-9-to-ICD-10 transition occurred exactly October 1, 2015,
  and this project's beneficiary data spans 2015-2023 — almost entirely after that cutover. A
  constant `0` here simply reflects that every diagnosis in this dataset genuinely is ICD-10-coded,
  which is exactly what a modern synthetic-data generator working against a mostly-post-2015
  timeframe should produce. Nothing wrong with this field; it's telling the truth and there's
  just no variation left to learn from in this particular timeframe.

Both remain correctly droppable for modeling — zero variance means zero predictive signal either
way — but the *why* differs and is worth having straight under questioning: one is "Synthea
couldn't model this realistically," the other is "the real world genuinely doesn't vary here in
this timeframe."

**Status: implemented and pushed** — all 13 columns added to `DROP_COLUMNS` in
`src/build_features.py`.

**A 14th zero-variance field, found later (2026-09-15) while scoping the Chow-test shared-covariate
list — `PRNCPAL_DGNS_VRSN_CD`.** Same category as `ICD_DGNS_VRSN_CD1`-`CD12` above: it's the
ICD-version companion flag specifically for `PRNCPAL_DGNS_CD` (the principal diagnosis), and its
populated values are a constant `0` (ICD-10), for the same real-world reason — this dataset's
beneficiary data is almost entirely post-October-2015. Unlike the other version-code fields,
though, this one is **not** populated across all claim types: it's populated for carrier + DME
only (100% null for outpatient), the same 2-of-3 applicability pattern several other
carrier/DME-only fields show (see Section 3's shared-feature audit). Zero variance means zero
predictive signal regardless of which claim types it's populated in, so the applicability pattern
doesn't change the drop decision — it just means this field cannot double as the Chow-test's
illustrative "populated in only 2 of 3 claim types" example the way it was briefly assumed to be;
see the correction under Section 3's degrees-of-freedom discussion. **Status: implemented and
pushed 2026-09-16** — added to `DROP_COLUMNS` in `src/build_features.py` alongside
`NCH_CLM_TYPE_CD`/`NCH_NEAR_LINE_REC_IDENT_CD` (the perfectly-collinear-with-`claim_type` fields
identified in Section 3's checklist item 4).

**Three more fields dropped 2026-09-16, all EXACT duplicates confirmed via direct equality checks,
not just correlation.** `LINE_SBMTD_CHRG_AMT == LINE_ALOWD_CHRG_AMT` (100% exact, `std == 0.0`
across 602,755 interior carrier lines), `LINE_NCH_PMT_AMT == LINE_PRVDR_PMT_AMT`, and
`NCH_CLM_PRVDR_PMT_AMT == NCH_CARR_CLM_ALOWD_AMT` (both 100% exact across all 718,074 carrier
rows). All three trace to one root cause: this dataset's Synthea generation never models a
provider write-off/discount or non-assignment billing — every carrier claim behaves as if fully
assigned and billed at exactly the allowed rate, with zero exceptions found. This is meaningfully
different from the merely-high (`r=0.90`-`0.98`) correlations found in the same pairwise sweep
(`LINE_COINSRNC_AMT` vs. billed/allowed, for instance) — exact equality makes the design matrix
**singular** for any linear model (infinitely many coefficient splits produce the identical
likelihood, not just an unstable one), which is a hard blocker, not multicollinearity to note and
move past. Kept field in each pair chosen as whichever one participates in the real
payer-reconciliation identity (allowed over billed; the primary NCH payment field over its
provider-payment mirror). **Status: implemented and pushed** — all three added to `DROP_COLUMNS`.

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
uses an events-per-variable (EPV — the ratio of observed outcome events to the number of
predictors in the model, a common rule-of-thumb check for whether a model has enough data to
estimate its coefficients reliably) rule of thumb of ≥10 — with ~6,270 events, the DME subset alone
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
is not merely inefficient, it's **model misspecification** (the technical term for a model whose
assumed structure doesn't match the real data-generating process) — a real bias, potentially even
producing the wrong sign in the pooled estimate (the kind of aggregation distortion Simpson's
paradox describes, where a pattern present in each subgroup can reverse or vanish once the
subgroups are combined).

**A note on precisely what "efficient" means for a shared coefficient when the homogeneity
assumption (the claim that a variable's effect is the same across all claim-type subgroups) is
actually correct — a distinct, technical sense of the word, not the runtime/effort sense retracted
above.** When homogeneity holds, estimating a variable's effect as one pooled parameter is more
*statistically* efficient: it draws on the full training set (~1.15M rows) rather than only one
claim type's subset, so its standard error (the estimated precision of a coefficient — smaller
means a more tightly-pinned-down estimate) is smaller — a lower-variance estimate of the same true
effect, in the formal sense tied to the Cramér-Rao bound (a theoretical floor on how low an
unbiased estimator's variance can go, given the sample size). This is a real, separate benefit from
anything about human effort or code runtime, and it only applies *when the homogeneity assumption
is actually correct* for that variable — if it's wrong, there is no efficiency benefit to weigh
against anything; pooling would just be wrong (misspecified), not efficient-but-biased.

**A worked, honest caveat on the practical magnitude of this benefit, prompted directly by a
challenge that the "efficiency" argument might be overstated at this sample size — worth
addressing head-on rather than defending the point abstractly.** Standard errors shrink roughly
proportional to `1/√n`, so pooling the full ~1.15M rows versus using only DME's ~66,700-row subset
gives roughly a `√(1.15M / 66,700) ≈ 4.1×` reduction in standard error for a DME-specific estimate
— a real, calculable difference. But for the two larger claim types (carrier ~62%, outpatient
~32%), each subset is already large enough on its own that pooling likely buys only marginal
additional precision. So the efficiency argument is real but **unevenly distributed** — it matters
most for DME-specific estimates and least for carrier/outpatient ones, not a uniform justification
across the board. Worth being honest, too, that for a portfolio project (rather than a deployed
system), the practical value of chasing this rigor is mainly demonstrating correct methodology in
an interview conversation — the final PR-AUC is unlikely to meaningfully hinge on which choice is
made here.

**Combining three claim-type-specific models into one PR-AUC (Precision-Recall Area Under the
Curve, the metric this project uses instead of plain accuracy given the minority-class denial
rate) is standard, defensible evaluation practice, not a hack.** Route every held-out test claim to
its matching claim-type model (hard, deterministic routing on the known `claim_type`, not a
learned gate), pool all resulting probability scores and true labels together, and compute PR-AUC
over that pooled set exactly as for a single model. Stratified models with deterministic routing
are evaluated this way routinely in production risk models (insurance underwriting, credit
scoring) — nothing methodologically irregular about it.

### Decision: a staged testing procedure, not a single up-front choice between the three
approaches

**This is a genuine refinement over the earlier per-variable-testing recommendation, prompted
directly by a follow-up question, not just a restatement of it.** The earlier version said "test
each shared variable individually for a claim-type interaction." The sharper, more rigorous version
runs that testing in two stages rather than jumping straight to per-variable checks:

**Stage 1 — one omnibus test first.** An *omnibus test* (also called a **Chow test** in
econometrics, after economist Gregory Chow, who introduced it in 1960) checks a single global
question — "do the coefficients differ across claim-type subgroups for *any* of the shared
variables at once?" — rather than testing each variable one at a time. Practically: fit the
fully-pooled model (`claim_type` dummies + every shared covariate included once, with one shared
coefficient each) and the fully-interacted model (`claim_type` dummies + every shared covariate
replaced by its claim-type-interaction terms), then compare their fit with a **likelihood-ratio
test** (a statistical test comparing two nested models by looking at the difference in their
**log-likelihood** — a number measuring how well a model fits the observed data, with
higher/less-negative values indicating better fit — where that difference follows a known
chi-squared distribution if the simpler model is actually adequate). Note that the `claim_type`
main effects (the dummies themselves) are present in **both** models regardless — what's being
tested is only whether those dummies *interact* with the other covariates, not whether they're
included at all.

**The precise null and alternative hypotheses, stated as explicit equations.** Using
`j ∈ {carrier, outpatient, dme}` for the three claim-type cells, `D_j` for the corresponding 0/1
claim-type dummy (all 3 present, no shared intercept — see the encoding-scheme discussion below),
and `x_1, ..., x_k` for the shared covariates:

*Restricted (pooled) model:*
```
logit(P(denied)) = γ_carrier·D_carrier + γ_outpatient·D_outpatient + γ_dme·D_dme
                    + Σ_i β_i · x_i
```
Each shared covariate `x_i` gets exactly one coefficient `β_i`, used across all three claim types.

*Unrestricted (fully-interacted) model:*
```
logit(P(denied)) = γ_carrier·D_carrier + γ_outpatient·D_outpatient + γ_dme·D_dme
                    + Σ_i ( β_i,carrier · x_i·D_carrier
                          + β_i,outpatient · x_i·D_outpatient
                          + β_i,dme · x_i·D_dme )
```
Each shared covariate gets one coefficient per claim type in which it's actually populated (see
the degrees-of-freedom correction below for covariates that aren't populated in all three).

`H₀`: for every shared covariate, its claim-type-specific coefficients are all equal to each other
(`β_i,carrier = β_i,outpatient = β_i,dme` for each `i`, jointly across all shared covariates at
once — not just one variable). `H₁`: at least one covariate's coefficient differs across at least
one pair of claim types, somewhere in the set. The `γ_j` terms are identical in structure in both
models and are never restricted — only the `β` terms are being tested for homogeneity.

**Correction, 2026-09-14 — the degrees-of-freedom formula below was wrong as originally stated,
and the error is provable from logic alone, independent of any specific dataset.** The original
version of this section assumed every shared covariate is populated across all 3 claim types
uniformly, giving a flat `3k − k = 2k` degrees of freedom for `k` shared covariates. That
assumption breaks the moment a covariate is populated in only 2 of the 3 claim types.
`PRNCPAL_DGNS_CD` (populated for Carrier + DME only, structurally absent from outpatient per
`data_dictionary.md`) is a concrete example, surfaced while scoping the shared-feature audit (see
the pre-flight checklist's item 4, below). For such a covariate there is no `β_outpatient` term at
all in the unrestricted model — including one would be identically zero for every row in the
dataset, not a real, identifiable parameter — so the restricted-vs-unrestricted comparison for that
one covariate is `1` coefficient vs. `2`, not `1` vs. `3`.

**Correction to the correction, 2026-09-15 — the specific example above was itself wrong, though
the math it illustrates is not.** An empirical null-rate-by-`claim_type` audit against
`train_model.parquet` (see checklist item 4 below) shows `PRNCPAL_DGNS_CD` is **0% null in all
three claim types** — genuinely 3-of-3, not 2-of-3. `data_dictionary.md`'s "Carrier, DME" claim for
this field was wrong; it was written from general CMS/RIF schema expectations, never checked
against this dataset's actual per-column null rates. (A follow-up value-level check confirmed this
further: real ICD-10 codes like `Z733`, `N184`, `T7432X` appear across all three claim types with
`dtype=str` and no `0`/`0.0`/empty-string sentinel values anywhere — this is genuinely populated
data, not disguised missingness.) The general formula below is unaffected — it was derived from
algebra, not from this one dataset — but the illustrative example needs to change. **`CARR_NUM`**
is the corrected, verified 2-of-3 example: populated for carrier + DME, 100% null for outpatient,
confirmed via the same audit, and not zero-variance (it's a real, high-cardinality provider-number
field, unlike `PRNCPAL_DGNS_VRSN_CD`, which shows the same 2-of-3 pattern but turned out to be
constant — see Section 2). Everywhere below that referenced `PRNCPAL_DGNS_CD` as the motivating
2-of-3 case, read `CARR_NUM` instead.

**The general, correct formula.** For a covariate present in `n_i` claim types (`n_i ∈ {2, 3}` for
any covariate actually scoped into the Chow test — a covariate present in only 1 claim type is
claim-type-exclusive by definition and isn't a shared-covariate-test subject in the first place,
per checklist item 2 below), the restricted model always gives it exactly 1 shared coefficient, and
the unrestricted model gives it `n_i` coefficients — one per claim type it's actually populated in.
That contributes `n_i − 1` degrees of freedom per covariate. Total degrees of freedom for the
omnibus test:
```
df = Σ_i (n_i - 1)
```
This collapses to the original `2k` only in the special case where every shared covariate is a
full 3-of-3 field. The completed shared-feature audit (checklist item 4 below) confirms real 2-of-3
covariates exist (`CARR_NUM`, `PRVDR_NUM`), so the flat `2k` figure would in fact overstate the true
degrees of freedom for this project's actual design matrix — not just a hypothetical concern. The
test statistic itself is unaffected by this correction:
```
LR = -2 · (ℓ_pooled - ℓ_interacted)   ~   χ²(df)      where df = Σ_i (n_i - 1)
```

**Practical implication for the design matrix, restated precisely given this correction:** for a
covariate present in only 2 of 3 claim types, build interaction terms against only its 2 applicable
`claim_type` dummies — omitting the third dummy's interaction term entirely, rather than
zero-filling-and-interacting a term that would be structurally constant at zero for the whole
dataset (which would silently inflate the apparent parameter count without adding any real,
estimable information).

**A note on the ANOVA parallel drawn earlier — corrected, since the original comparison was too
loose.** Plain one-way ANOVA (Analysis of Variance) tests whether group *means* differ from a
single grand mean — a comparison concerning intercepts only, not the slopes of other covariates.
That's not quite the right analogy for testing whether *other variables' effects* differ by group.
The more precise parallel, from the ANCOVA (Analysis of Covariance) literature, is a **test of
homogeneity of regression slopes** — and the Chow test is exactly this test, generalized to more
than two groups. The **Tukey's HSD** part of the earlier parallel should also be dropped: Tukey's
HSD does pairwise comparisons of group *means* specifically, which isn't the right tool for
per-variable slope testing (Stage 2, below) — the correct framing is that Stage 1 and Stage 2 are
the *same* likelihood-ratio technique applied at two different levels of granularity (jointly
across all covariates, then one covariate at a time), not two different named techniques.

**Stage 2 — only if Stage 1 rejects.** A rejection means real evidence that *something* differs by
claim type somewhere, but not *which* specific variables carry it. That's when per-variable testing
becomes necessary: run the same likelihood-ratio comparison one covariate at a time (or use a
model-selection criterion like **AIC**, the Akaike Information Criterion — a fit-quality score that
penalizes extra parameters so added complexity has to earn its place rather than being rewarded by
default — to decide which specific interactions are worth keeping).

**A precise clarification on what a Chow-test rejection actually implies, prompted directly by a
sharp follow-up question — it's easy to over-read the rejection, and worth spelling out exactly
what it does and doesn't establish.** In the classic Chow-test construction, the two models being
compared *are* literally the fully-pooled model and the fully-stratified model — so a rejection
does mean "the fully-stratified alternative fits significantly better than the fully-pooled one."
But that comparison is between two extremes, and a rejection only tells you the true answer lies
*somewhere between them* — not that the extreme (full stratification for every shared variable) is
itself the best answer. A rejection means *at least one* shared variable has a real claim-type
interaction; it does **not** mean *every* shared variable does. Jumping straight from a Chow
rejection to full stratification would give every shared variable its own separate coefficient,
including any that are genuinely homogeneous — needlessly discarding the statistical-efficiency
benefit established above for exactly those variables, for zero gain in correctness, since pooling
was already valid for them. Per-variable testing after a Chow rejection plays the role of
localizing exactly where the heterogeneity lives, rather than assuming it's everywhere. Full
stratification remains a legitimate, defensible shortcut after a Chow rejection if the analyst-time
cost of per-variable testing isn't worth paying — it's a real time-vs-precision tradeoff, not a
question of correctness either way.

**How a mixed model (some variables shared, some claim-type-specific) is actually built — a
concrete design-matrix construction, not exotic machinery.** `claim_type` dummies (3 of them, cell-
means, no intercept) are always included, representing the baseline shift. A variable that *passes*
its homogeneity test appears once, as a single plain column, one shared coefficient. A variable
that *fails* is replaced by its interaction columns instead (one per claim type it's actually
populated in — see the degrees-of-freedom correction above for covariates not populated in all
three), with its raw column dropped entirely — keeping both would recreate the same collinearity
problem discussed under `claim_type`'s encoding scheme below. The final design matrix is simply a
mix of plain columns and interaction columns, decided per variable by its own test result.

**Why staging it this way is better than testing every variable individually from the start:** if
nothing is actually heterogeneous, the omnibus test settles that in one step instead of running a
separate test per shared variable. If everything is heterogeneous, the omnibus test also catches
that in one step, and the resulting fully-interacted pooled model is mathematically identical to
stratified regression anyway (the point already established above) — so Approaches 2 and 3 aren't
really a choice to make blind up front; they're the two endpoints of this one staged testing
procedure, discovered by the data rather than assumed in advance.

**Implementation note, decided alongside this refinement:** actually running Chow's test means
fitting two logistic regression models and comparing their log-likelihoods directly — this is
naturally done with **`statsmodels`**, not `sklearn`. `statsmodels` exposes log-likelihood,
coefficient p-values, and built-in likelihood-ratio-test support; `sklearn`'s logistic regression
is built for prediction, not this kind of inferential model comparison. This also means the Chow
test itself is arguably the first genuinely *Phase 2* step (Build Guide Step 12, the baseline
logistic regression) rather than Phase 1 — Phase 1's job ends at handing off a properly-prepared
feature set; deciding how to fit the baseline model on it is Phase 2's.

**This will be decided when the Phase 2 baseline script is actually built**, with this document
providing the reasoning trail — not decided speculatively now, ahead of writing that code.

### Pre-flight checklist — read this before writing any Chow-test or baseline fitting code

**Added 2026-09-13 after a handoff to a new chat missed one of these points — the fix isn't to
retype this reasoning into every handoff message, it's to name the specific traps explicitly, once,
here, and have handoffs point at this checklist by name.** Four distinct risks, each capable of
silently invalidating the test or crashing the fit if skipped:

1. **The `claim_type` dummies must appear identically (cell-means, all 3, no intercept) in both
   the pooled and interacted models.** This isn't optional styling — it's what makes the two
   models properly *nested*, which the likelihood-ratio test's validity depends on. If the pooled
   model uses a normal intercept instead and omits the `claim_type` dummies entirely, the test
   conflates "does claim type shift the baseline" with "do the interactions matter" into one
   number, rather than isolating the interaction question. See the null/alternative hypotheses
   above — both are stated assuming this shared structure.
2. **Claim-type-*exclusive* fields (the structural-missingness columns from Section 2) are not
   Chow-test subjects — don't let them leak into the "shared feature set."** The Chow test is
   scoped to covariates present across more than one claim type. A field like
   `REV_CNTR_TOT_CHRG_AMT` (outpatient-only) has no coherent "shared coefficient" to test in the
   first place — it must appear identically, zero-filled and interacted against its one applicable
   `claim_type` dummy, in *both* models, entirely outside the hypothesis being tested. Conflating
   this category with the shared-covariate list breaks the nesting the same way point 1 does.
   **Correction, 2026-09-16 — this zero-fill is scoped to the design-matrix-building step itself,
   not to `train_model.parquet`.** An earlier version of `build_features.py` zero-filled these
   columns' raw values in the shared parquet file directly — that was wrong (see point 3's
   correction below) and has been reverted. The zero-fill described here happens on a copy, inside
   whichever script actually builds the Chow-test/logistic-regression design matrix, never in the
   file XGBoost also reads from.
3. **`train_model.parquet` deliberately still has raw `NaN`s for NUMERIC claim-type-exclusive
   fields — this is now a permanent design decision, not open work.** A first implementation
   (2026-09-16) zero-filled these columns directly in `build_features.py`, on the reasoning that
   the interaction construction in point 2 needs a zero-filled value. That was wrong and has been
   reverted: `0` is a legitimate real value for these fields (a genuine `$0` charge, a genuine `0`
   count), so zero-filling the raw column conflates "genuinely `$0`" with "doesn't apply to this
   claim type" — the same distinct-value-vs-missingness conflation already caught for the NPI
   presence flags and `PRNCPAL_DGNS_VRSN_CD` (Section 2), recurring in a new place. It also
   discards information XGBoost could use natively, since XGBoost branches on real `NaN` and can
   treat "this field is absent" as its own signal — a zero-fill silently removes that. **General
   principle: `0` is a value, `NaN` is the absence of one — never use the former to represent the
   latter, for any field type.**

   `build_features.py`'s `fill_claim_type_exclusive_fields()` now only sentinel-fills
   **categorical/string** claim-type-exclusive columns (`"NOT_APPLICABLE"`, an explicit new
   category rather than an overloaded existing value — this doesn't have the 0-vs-missing problem
   and is correctly implemented). **Numeric claim-type-exclusive columns keep their real `NaN` in
   `train_model.parquet` — this is intentional, not unfinished.** The zero-fill-for-interaction
   trick from point 2 is still correct and still needed, but belongs ONLY inside Phase 2's actual
   Chow-test/design-matrix-building script (not yet written), applied there on a **copy** of the
   relevant columns, scoped to that one multiplication — never baked upstream into the shared file
   both the linear model and XGBoost read from. Fitting `statsmodels` directly against
   `train_model.parquet`'s raw `NaN`s will still error or silently drop rows exactly as originally
   warned here — that risk is unchanged; only *where* the fill happens has moved.
4. **The shared-vs-claim-type-exclusive column split — now fully enumerated via an empirical
   audit, 2026-09-15, superseding the 2026-09-14 documentation-only pass.** Two rounds of checks
   against `train_model.parquet` (a null-rate-by-`claim_type` sweep, then a sentinel-aware
   follow-up, then two targeted value-level/crosstab checks) settled every open item:

   - **Confirmed genuinely shared (3-of-3, 0% null in all three claim types):** `PRNCPAL_DGNS_CD`
     (data dictionary's "Carrier, DME" claim was wrong — see the correction above), `provider_state`,
     and all 5 NPI-presence flags (`has_referring_physician`, `has_performing_physician`,
     `has_attending_physician`, `has_operating_physician`, `has_rendering_physician`) — resolving
     the item left open on 2026-09-14. **`HCPCS_CD`** is also 3-of-3 by presence, with within-carrier
     missingness now resolved — see Section 5.
   - **Confirmed 2-of-3, needing the omit-third-interaction treatment above:** `CARR_NUM`
     (carrier + DME, absent outpatient — the corrected illustrative example) and `PRVDR_NUM`
     (outpatient + DME, absent **carrier** — `data_dictionary.md`'s "All claim files" claim was
     also wrong, in the opposite direction from its `PRNCPAL_DGNS_CD` error). Both now have their
     cardinality treatment implemented — see Section 5.
   - **Excluded — perfectly collinear with `claim_type`, a new finding, not previously
     considered.** `NCH_CLM_TYPE_CD` and `NCH_NEAR_LINE_REC_IDENT_CD` are native CMS fields, not
     ones this project engineered, that happen to be CMS's own claim-classification codes. A
     crosstab confirmed each claim type maps to exactly one distinct value with zero overlap
     (carrier→`71`/`O`, outpatient→`40`/`W`, DME→`82`/`M`) — including either would recreate the
     `claim_type` dummy set under a different label and reproduce the exact rank-deficiency bug
     already documented once for the applicability-indicator case. Excluded from the shared list
     entirely.
   - **A methodological note on how these were verified, worth keeping for the same reason as the
     rest of this document's error trail.** An initial sentinel-detection pass (treating `0`,
     `0.0`, `"0"`, and empty string as possible disguised-missingness placeholders, prompted by
     the question "is `0.0` a valid diagnosis code?") produced a batch of false positives on the
     5 NPI-presence flags and on `PRNCPAL_DGNS_VRSN_CD` — because `0` is a legitimate, meaningful
     value for a binary indicator (role not populated) and for `PRNCPAL_DGNS_VRSN_CD`'s own
     constant ICD-10 flag, not a placeholder for missing data in either case. The lesson: a
     blanket zero-as-sentinel rule is only valid for genuine code/ID-type columns where `0` isn't
     a valid domain value (which is exactly why it correctly resolved the `PRNCPAL_DGNS_CD`
     question) — it must not be applied to engineered binary flags or, by the same logic, to
     dollar-amount/count fields where `$0` or a `0` count is a real, common outcome. **This same
     lesson recurred once more, 2026-09-16, in `build_features.py`'s numeric zero-fill — see point
     3's correction above.**

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
an **exactly rank-deficient design matrix** (a technical way of saying the columns of predictor
data aren't all independent — some information is exactly duplicated, which breaks the math the
model needs to solve), not just high correlation. Standard unregularized logistic regression
(Newton-Raphson/IRLS, the standard numerical algorithms used to fit a logistic regression) would
hit a singular matrix and fail to converge; `sklearn`'s default L2-regularized version wouldn't
error, but would arbitrarily split the coefficient between the two redundant terms in a way that's
statistically meaningless.

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
meaningless claim about billing-category membership. One-hot encoding (representing a categorical
variable as a set of separate 0/1 columns, one per category) is the correct family of choices for a
nominal variable; the question is which specific *variant* of one-hot encoding.

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
directly** (its own log-odds baseline — log-odds being the natural-log-scale quantity a logistic
regression's linear predictor actually models, since probability itself isn't linear in the
predictors but its log-odds transform is), not a difference from anything — there's no subtraction
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

**This entire section's decisions assume Approach 2 (or the staged-testing hybrid recommended
above) is what's ultimately implemented.** Under pure Approach 3 (stratified regression),
`claim_type` never appears as a predictor inside any single subset's model at all — it's purely the
routing key used to split the data beforehand, and none of the cell-means/dummy discussion applies
there.

**Status: implemented and pushed** — `claim_type_carrier`/`claim_type_outpatient`/`claim_type_dme`
cell-means dummies built in `src/build_features.py`.

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

**Status of the consolidated script:** `src/build_features.py` implemented and pushed 2026-09-13,
covering the state crosswalk, NPI presence flags + drops, `claim_type` cell-means encoding, and all
confirmed-droppable columns (always-null + zero-variance) from Sections 1-2 above. Produces
`train_model.parquet`/`val_model.parquet`/`test_model.parquet` from `train`/`val`/`test.parquet`.
Phase 1 is complete — the state-crosswalk verification is fully closed (Section 1), and nothing
remains open from Phase 1 itself.

**Remaining, genuinely open — this is Phase 2's starting point, not Phase 1's:** see the pre-flight
checklist in Section 3 above. **2026-09-15: item 4 (the shared/exclusive column split) is now fully
resolved** — see the checklist's updated item 4 for the final shared (3-of-3), 2-of-3, and
excluded-as-collinear lists. **2026-09-16: (a) and most of (c) are now done** —
`PRNCPAL_DGNS_VRSN_CD`, `NCH_CLM_TYPE_CD`, and `NCH_NEAR_LINE_REC_IDENT_CD` are added to
`DROP_COLUMNS`, and `fill_claim_type_exclusive_fields()` sentinel-fills categorical
claim-type-exclusive columns. **See Section 5 for the cardinality-encoding step that resolves (b)
(`HCPCS_CD`'s within-carrier missingness) and `PRVDR_NUM`/`PRNCPAL_DGNS_CD`/`provider_state`'s
cardinality treatment.** What's genuinely still open: (c') the zero-fill-for-interaction step and
the 2-of-3 omit-third-interaction construction for `CARR_NUM`/`PRVDR_NUM`, both of which belong in
`src/build_chow_design_matrix.py` (Section 5) but aren't written yet. The degrees-of-freedom
formula (`df = Σ(n_i − 1)`) remains correct regardless of these remaining items — it was derived
from algebra, not from any specific column.

**Also, separately: a real target-construction fix, not a feature-engineering one, worth flagging
here because it changes what every prior audit in this document was computed against.**
`denial_reasons.py`/`denial_rules.py` gained a 6th risk factor, `missing_hcpcs`, on 2026-09-16 —
every prior rule taking `hcpcs_col` was structurally unable to fire when `HCPCS_CD` was missing,
which mechanically forced `is_denied` toward 0% for the 62.5% of carrier claims missing that field
(confirmed empirically at exactly 0/448,567 denials), backwards from a real payer system where a
missing procedure code is itself a denial trigger (CARC 16). This requires rerunning
`build_target_and_split.py` and very likely lowering `CALIBRATION_SCALE` from 1.4 — not yet done as
of this note. Full writeup in `data/TARGET_DEFINITION.md` and this project's decisions-and-learnings
notes. Once `is_denied` is regenerated and recalibrated, the EDA figures, `data_dictionary.md`'s
null-rate numbers, and this document's shared-feature audit above should all be treated as computed
against the *old* label until spot-checked against the new one — the underlying column-applicability
facts (which claim types a field is populated in) won't change, since that's a property of the raw
CMS data, not of `is_denied`, but any is_denied-dependent number quoted anywhere in this repo's docs
predates this fix.

---

## 5. Cardinality encoding for the Chow-test design matrix (`src/build_chow_design_matrix.py`)

**Added 2026-09-17.** Four confirmed shared/2-of-3 covariates — `HCPCS_CD`, `PRNCPAL_DGNS_CD`,
`PRVDR_NUM`, `provider_state` — have been flagged since the original pre-flight checklist as
needing "documented cardinality treatment" before entering any regression, but that treatment was
never actually implemented until now. This section documents the choices, and the boundary that
governs where this code lives.

**The boundary, restated once more because it's easy to get backwards:** every encoding below
runs in `src/build_chow_design_matrix.py`, on a **copy** of `train_model.parquet` loaded at
script start — never inside `build_features.py`, and `train_model.parquet` itself is never
written to by this script. This is the same boundary established in Section 3's pre-flight
checklist item 3 for the zero-fill step, extended to cover encoding generally: anything built
specifically for the Chow-test/logistic-regression design matrix, and not useful (or actively
harmful, in the zero-fill case) to XGBoost reading the same shared file, belongs here, not
upstream.

**`provider_state` (51 categories, no natural ordering) — plain reference-cell one-hot
(`drop_first=True`).** Unlike `claim_type`, this field is never used in an interaction
construction, so none of Section 3's cell-means requirement applies — that requirement was
specific to needing an explicit dummy for every claim type to interact against, not a general
rule for every categorical in the model. Ordinary reference-cell coding is the right default
here.

**`HCPCS_CD` and `PRNCPAL_DGNS_CD` (144 and hundreds of distinct codes respectively) — top-20 +
`__OTHER__` + `__MISSING__`.** Full one-hot on either would blow up the design matrix (144+
columns for a single covariate); a fixed top-N bucket keeps the matrix tractable while preserving
the individual identity of the codes that actually carry volume. Two buckets beyond the top-N,
deliberately kept separate rather than merged into one:
- `__OTHER__` — a real, populated code that didn't make the top-N cutoff.
- `__MISSING__` — genuine `NaN`. Merging this into `__OTHER__` would silently claim "this claim
had *some* uncommon procedure code" when the truth is closer to "this claim's procedure code is
unknown or absent" — a materially different fact, and conflating the two would undo the same
distinct-value-vs-missingness discipline this document has enforced everywhere else (Section 2's
`PRNCPAL_DGNS_VRSN_CD` case, Section 3 point 3's zero-fill correction).

**This is also the resolution to `HCPCS_CD`'s within-carrier-missingness question, left open since
2026-09-15 (Section 3's pre-flight checklist item 4).** At the time, the anticipated fix was "most
likely a genuine missing-indicator." The `__MISSING__` dummy produced by this general-purpose
encoding scheme *is* that missing-indicator — it didn't need its own bespoke treatment once every
other high-cardinality shared covariate needed the same top-N-plus-buckets scheme anyway.

**`PRVDR_NUM` (8,460 unique values, real repeat structure already used by `provider_outlier`/
`dx_procedure_mismatch`) — frequency encoding, not one-hot.** One-hot at this cardinality is
infeasible outright (8,460 columns for a single covariate); frequency encoding (each provider
mapped to its own share of non-null `PRVDR_NUM` rows) is a standard, defensible choice for a
high-cardinality categorical with meaningful repeat structure, and requires no target information
(unlike target encoding, which would risk its own leakage concerns worth avoiding for a covariate
that's also used elsewhere in this project as a fraud/outlier proxy).

**Critical implementation detail: `PRVDR_NUM`'s real `NaN` (its 2-of-3 gap, absent for carrier)
must survive the frequency encoding untouched, not get silently filled.** `frequency_encode()`
computes its lookup table via `value_counts(dropna=True)` and maps via `.map()`, which correctly
leaves unmapped/`NaN` inputs as `NaN` in the output. This is not incidental: `PRVDR_NUM` is a
confirmed 2-of-3 shared covariate, and per Section 3's degrees-of-freedom correction, its NaN for
the one absent claim type must stay NaN so the (not yet written) interaction-term construction can
correctly omit that claim type's term entirely. Fabricating a frequency value there — even
something as seemingly neutral as `0` — would repeat the exact 0-vs-missing mistake the
2026-09-16 numeric zero-fill correction was written to prevent, just for a different encoding
scheme than the one that mistake originally occurred in.

**Correction, 2026-09-17 — the script's own sanity check for this initially asserted a stronger
claim than the data supports, and the failure it caught is itself a useful finding.** The first
version of this check expected `prvdr_num_freq`'s NaN count to equal carrier's row count exactly.
Running it produced a 136-row mismatch — investigated directly rather than dismissed: `PRVDR_NUM`
turns out to have a small, genuine *within-outpatient* gap (136 null rows out of 367,542, ~0.037%)
in addition to its 100%-null carrier gap. This is the same category of thing as `HCPCS_CD`'s
within-carrier missingness described above — ordinary, non-structural missingness sitting on top
of a structural one — just three orders of magnitude smaller, which is why it never surfaced as
its own line item until this check caught it. The sanity check now compares against the sum of
both real components (carrier's structural count plus outpatient's small residual) rather than
carrier's count alone, and confirms clean.

**Status: implemented and pushed** — `top_n_encode()`, `frequency_encode()`, and
`build_chow_design_matrix()` in `src/build_chow_design_matrix.py`. **Not yet implemented**, and
the next piece of this script: the claim-type interaction-term construction itself (Section 3's
zero-fill-on-a-copy step for numeric claim-type-exclusive covariates, and the omit-third-
interaction logic for `CARR_NUM`/`PRVDR_NUM`) — this encoding step is a prerequisite for that
work, not a substitute for it.