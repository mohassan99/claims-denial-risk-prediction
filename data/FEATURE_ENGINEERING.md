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
carrier/DME-only fields show (see Section 6's full presence audit). Zero variance means zero
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

**Twenty more always-100%-null columns found 2026-09-21, via the pre-sentinel-fill full presence
audit (Section 6) rather than a targeted check.** The original 2026-09-13 always-null sweep (the
10 columns above) turned out to be incomplete, not stale — confirmed directly against
`train_model.parquet`: each of these 20 (`CARR_LINE_MTUS_CD`, `CARR_LINE_RX_NUM`,
`CLM_CLNCL_TRIL_NUM`, `FI_NUM`, `HCPCS_1ST_MDFR_CD`-`4TH_MDFR_CD`, `LINE_NDC_CD`,
`LINE_PMT_80_100_CD`, `REV_CNTR_1ST_ANSI_CD`-`4TH_ANSI_CD`, `REV_CNTR_APC_HIPPS_CD`,
`REV_CNTR_DSCNT_IND_CD`, `REV_CNTR_IDE_NDC_UPC_NUM`, `REV_CNTR_NDC_QTY_QLFR_CD`,
`REV_CNTR_OTAF_PMT_CD`, `REV_CNTR_PACKG_IND_CD`) sat as a dataset-wide CONSTANT string
(`"NOT_APPLICABLE"`) rather than real `NaN`, since `fill_claim_type_exclusive_fields()` correctly
(but silently) sentinel-fills a column that's 100%-null in all three claim types the same way it
fills one that's null in only one or two — zero variance, zero information, either way. Does not
change the corrected 2-of-3 (13) or 3-of-3 counts from Section 6 — a separate, 0-of-3 bucket
entirely. **Status: implemented and pushed** — all 20 added to `DROP_COLUMNS`.

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
full 3-of-3 field. **Section 6's complete presence audit confirms 13 real 2-of-3 covariates exist
in this dataset, not just the 2 (`CARR_NUM`, `PRVDR_NUM`) documented here originally** — and a
2026-09-22 follow-up (also Section 6) found several of the nominally-3-of-3 covariates are
EMPIRICALLY 1-of-3 or 2-of-3 once actual per-category variance (not just null-rate) is checked —
so the flat `2k` figure meaningfully overstates the true degrees of freedom for this project's
actual design matrix, for two independent reasons now, not one. The test statistic itself is
unaffected by this correction:
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
   trick from point 2 is still correct and still needed, and is now implemented in
   `src/build_chow_design_matrix.py` (Sections 5-6), applied there on a **copy** of the relevant
   columns, scoped to that one multiplication — never baked upstream into the shared file both the
   linear model and XGBoost read from.
4. **The shared-vs-claim-type-exclusive column split — SUPERSEDED, see Section 6 for the final,
   complete, empirically-verified accounting.** The version of this item that stood from
   2026-09-15 through 2026-09-16 confirmed only 2 shared 2-of-3 covariates (`CARR_NUM`,
   `PRVDR_NUM`) — that count was itself incomplete, caught only once the actual design-matrix code
   produced a column-count mismatch against a hand-derived expectation. Section 6 documents the
   full, one-time enumeration that should have been the starting artifact rather than something
   assembled incrementally across a dozen individual-column surprises. **A second, deeper layer of
   this same lesson recurred 2026-09-22:** even the "3-of-3, genuinely shared" bucket from that
   audit turned out to include covariates that are only nominally shared (never `NaN`) but
   EMPIRICALLY zero-variance within a claim type — see Section 6's 2026-09-22 addendum for the
   specific NPI-flag corrections this produced.

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

**Status as of 2026-09-22: the full Chow-test design-matrix pipeline (`src/build_chow_design_matrix.py`)
is written and verified against real data, and the Section-4-flagged scope gap is now CLOSED.**
Cardinality encoding (Section 5), the unrestricted (fully-interacted) design matrix, and the
restricted (pooled) design matrix (Section 6) all exist and pass their own sanity checks — and, as
of 2026-09-22, the unrestricted matrix now interacts ALL genuinely-shared covariates (2-of-3 AND
3-of-3), not just the 2-of-3 ones — see Section 6's 2026-09-22 addendum for the memory-optimization
work and the 39 additional empirical zero-variance findings that surfaced once this ran against
real data. **What's genuinely still open:** the actual Chow-test fitting code itself (two
`statsmodels.Logit` calls + the likelihood-ratio statistic) — not yet written. The
degrees-of-freedom formula (`df = Σ(n_i − 1)`) remains correct regardless — it was derived from
algebra — and the covariate universe it sums over is now the fully-verified one from Section 6,
not a partial scope.

**Also, separately: a real target-construction fix, not a feature-engineering one.**
`denial_reasons.py`/`denial_rules.py` gained a 6th risk factor, `missing_hcpcs`, on 2026-09-16 —
every prior rule taking `hcpcs_col` was structurally unable to fire when `HCPCS_CD` was missing,
which mechanically forced `is_denied` toward 0% for the 62.5% of carrier claims missing that field
(confirmed empirically at exactly 0/448,567 denials), backwards from a real payer system where a
missing procedure code is itself a denial trigger (CARC 16). Recalibrated 2026-09-17 (decoupled
`missing_hcpcs` from `CALIBRATION_SCALE`, base_prob fixed at 0.08) — the resulting overall
`is_denied` rate is 12.1%, confirmed in range. **All of this project's EDA figures,
`data_dictionary.md`'s null-rate numbers, and the shared-feature/presence audits (Section 6) remain
valid** — the underlying column-applicability facts (which claim types a field is populated in) are
a property of the raw CMS data, not of `is_denied`, and were re-verified against
`train_model.parquet` after recalibration.

---

## 5. Cardinality encoding for the Chow-test design matrix (`src/build_chow_design_matrix.py`)

**Added 2026-09-17.** Five confirmed shared/2-of-3 covariates — `HCPCS_CD`, `PRNCPAL_DGNS_CD`,
`PRVDR_NUM`, `CARR_NUM`, `provider_state` — needed cardinality treatment before entering any
regression. `CARR_NUM` was initially missed (see the correction note in
`src/build_chow_design_matrix.py`'s docstring) — caught when the interaction-term construction
(Section 6) hit its raw string column directly, not silently.

**The boundary, restated once more because it's easy to get backwards:** every encoding below
runs in `src/build_chow_design_matrix.py`, on a **copy** of `train_model.parquet` loaded at
script start — never inside `build_features.py`, and `train_model.parquet` itself is never
written to by this script.

**`provider_state` (51 categories, no natural ordering) — plain reference-cell one-hot
(`drop_first=True`).** Unlike `claim_type`, this field is never used in an interaction
construction, so none of Section 3's cell-means requirement applies.

**`HCPCS_CD` and `PRNCPAL_DGNS_CD` (144 and hundreds of distinct codes respectively) — top-20 +
`__OTHER__` + `__MISSING__`.** `__MISSING__` kept deliberately separate from `__OTHER__` (a real
populated code that didn't make the top-N cutoff) — conflating them would claim "some uncommon
code" when the truth is "code unknown or absent," undoing the distinct-value-vs-missingness
discipline enforced everywhere else in this document. This is also the resolution to `HCPCS_CD`'s
within-carrier-missingness question, left open since 2026-09-15: the `__MISSING__` dummy IS the
missing-indicator anticipated there.

**`PRVDR_NUM`/`CARR_NUM` (8,460 and similarly high cardinality) — frequency encoding, not
one-hot.** Real `NaN` preserved for each covariate's one absent claim type, never filled — required
so the interaction-term construction (Section 6) can correctly build only the applicable terms.
`PRVDR_NUM` additionally has a small genuine within-outpatient residual gap (136 rows, ~0.037%),
caught by the script's own sanity check comparing against the sum of both real components (its
structural carrier-null count plus this residual) rather than the structural count alone.

**Status: implemented and pushed, fully verified.**

---

## 6. The claim-type interaction-term construction, the restricted design matrix, and the full
presence audit that corrected both

**Added 2026-09-17.** This section covers three pieces of work that turned out to be tightly
coupled: building the unrestricted (fully-interacted) design matrix, building its restricted
(pooled) counterpart, and — triggered by a real bug those two matrices' cross-checks surfaced — a
complete, one-time enumeration of every column's claim-type presence pattern, replacing the
incremental, one-column-at-a-time discovery process this document had been running on since
Section 3 was first written.

### The unrestricted (fully-interacted) design matrix

`add_claim_type_interactions()` in `src/build_chow_design_matrix.py` builds `value ×
claim_type_dummy` interaction terms for every remaining numeric claim-type-exclusive or 2-of-3
covariate, zero-filling ONLY on this in-memory copy (never `train_model.parquet` — Section 3
checklist item 3). Two real bugs surfaced and fixed while building this, both caught by the
script's own sanity checks rather than assumed correct:

- **An initial version raised `TypeError` unconditionally on any remaining non-numeric column**,
  on the theory that a leftover string column was always a bug. Checked directly against real
  data before trusting that theory: of 84 non-numeric columns still carrying real `NaN`, only
  `CARR_NUM`/`PRVDR_NUM` were ever genuinely claim-type-exclusive (and both are already consumed
  into numeric `carr_num_freq`/`prvdr_num_freq` before this function runs) — the other 82 are
  ordinary, legitimate partial real-world missingness in non-claim-type-exclusive fields
  (`ICD_DGNS_CD2`-`25`, `ICD_PRCDR_CD1`-`24`, `HCPCS_CD`, `BETOS_CD`, etc.), never in scope for
  this treatment. The unconditional raise would have crashed on all 82 of them. Corrected to skip
  non-numeric columns unconditionally instead.
- **`prvdr_num_freq`/`carr_num_freq` were wrongly grouped into an "already encoded, skip" set**
  alongside the genuinely-finished one-hot dummies (`state_*`, `hcpcs_*`, `dgns_*`). That was
  backwards: those two were frequency-encoded SPECIFICALLY so their real `NaN` could survive into
  this step and get interaction terms built — skipping them silently produced zero interaction
  terms for both, caught by the script's own sanity check (empty lists where exactly 2 each were
  expected).

### The restricted (pooled) design matrix

`build_restricted_design_matrix()` builds the actual counterpart the Chow test compares against:
every genuinely shared covariate collapses to ONE plain column (a single shared coefficient)
instead of separate per-claim-type interaction terms — this is the literal restriction under test.
Claim-type-exclusive fields (n_i=1) get IDENTICAL treatment to the unrestricted model — one
interaction term, no restriction possible with only one claim type to begin with, matching the
algebra directly (n_i=1 contributes `n_i-1=0` degrees of freedom either way). Both design-matrix
functions share one `_null_pattern()` helper for classifying each column, specifically so they can
never silently disagree about which claim types a covariate applies to.

### The bug that triggered the full audit

Cross-checking the two matrices' column counts (`unrestricted - restricted` should equal exactly 1
extra column per genuine 2-of-3 covariate) came out to **13**, not the expected **2** — the number
that was actually wrong was the expectation, not the code. `_null_pattern()` classifies every
column dynamically at runtime; it had already been correctly finding and interacting every real
2-of-3 covariate, by name, regardless of whether this document had documented that name yet.

### The full presence audit — the artifact that should have existed from the start

Rather than chase the 13 one column at a time (the pattern this document had followed since
`PRNCPAL_DGNS_CD` first turned out wrong on 2026-09-15), a complete enumeration was run once:
every column's null rate computed for each of the 3 claim types, classified by `n_present` (how
many claim types it's populated in), saved as `data/full_claim_type_presence_audit.csv`.

**Results: `n_present` distribution across all 206 non-identifier/non-label columns —
1-of-3 (claim-type-exclusive): 32. 2-of-3 (shared, needs omit-third treatment): 13.
3-of-3 (genuinely shared): 161. 0-of-3 (populated nowhere): 0** — confirming `DROP_COLUMNS`
already has no gaps; every column populated in zero claim types is already gone.
`32 + 13 + 161 = 206`, plus the 6 excluded identifier/label/dummy columns (`BENE_ID`, `CLM_ID`,
`_row_id`, `is_denied`, and the 3 `claim_type_*` dummies) = 212, matching
`train_model.parquet`'s actual column count exactly.

**The complete list of 13 confirmed 2-of-3 covariates, corrected from the 2 previously
documented (`CARR_NUM`, `PRVDR_NUM`) in Section 3:**

| Column | Absent claim type |
|---|---|
| `CARR_CLM_CASH_DDCTBL_APLD_AMT` | outpatient |
| `CARR_CLM_PRMRY_PYR_PD_AMT` | outpatient |
| `CARR_NUM` | outpatient |
| `LINE_ALOWD_CHRG_AMT` | outpatient |
| `LINE_BENE_PMT_AMT` | outpatient |
| `LINE_BENE_PRMRY_PYR_PD_AMT` | outpatient |
| `LINE_BENE_PTB_DDCTBL_AMT` | outpatient |
| `LINE_NCH_PMT_AMT` | outpatient |
| `LINE_SRVC_CNT` | outpatient |
| `NCH_CARR_CLM_ALOWD_AMT` | outpatient |
| `NCH_CARR_CLM_SBMTD_CHRG_AMT` | outpatient |
| `NCH_CLM_BENE_PMT_AMT` | outpatient |
| `PRVDR_NUM` | **carrier** |

**Why this pattern isn't arbitrary — a real structural explanation, not a coincidence.** 12 of the
13 share `absent_type: outpatient`, splitting into two families, both explained by real Medicare
RIF architecture: (1) `LINE_*` fields — carrier and DME are both Part B **non-institutional** claim
types sharing the same line-item RIF schema; outpatient is **institutional** and uses a parallel
`REV_CNTR_*` schema for the equivalent concepts instead, by design of the RIF layout itself, not a
data quality issue. (2) `CARR_CLM_*`/`NCH_CARR_CLM_*` fields — named for "carrier claim"
specifically, but populated in DME too, consistent with DME's RIF layout closely mirroring
carrier's (the same carrier+DME pairing already seen for `PRVDR_SPCLTY` and `CARR_NUM` itself).
`PRVDR_NUM` is the one genuine outlier, absent from **carrier** rather than outpatient — the
opposite direction from every other field in the list, worth keeping visible rather than letting it
blend into "the usual pattern."

**Status: implemented and pushed, fully verified** —
`add_claim_type_interactions()`/`build_restricted_design_matrix()` in
`src/build_chow_design_matrix.py`; `data/full_claim_type_presence_audit.csv` committed as the
permanent, canonical source for every future claim-type-applicability question, superseding
Section 3's incomplete two-covariate list. **See the 2026-09-22 addendum below** for the follow-up
work that closed the remaining "~161 covariates not yet interacted" gap this section originally
left open, and the further corrections that surfaced doing it.

### Interacting the genuinely-3-of-3 covariates, a memory-optimization saga, and 39 more empirical
zero-variance findings (2026-09-22)

**Closing the scope gap.** This section's original "Not yet done" note flagged that the ~161
genuinely-shared 3-of-3 covariates (`provider_state`/`HCPCS_CD`/`PRNCPAL_DGNS_CD`'s dummies, the 5
NPI flags) were sitting as untested plain columns in both design matrices — meaning the omnibus
Chow test, as originally scoped, only ever covered the claim-type-exclusive/2-of-3 covariates, not
the full `k` the original equations describe. `add_claim_type_interactions()`'s Pass 2 now
interacts all of these too, via a new `_interact_with_zero_variance_guard()` helper.

**Why a guard is needed at all, and why it isn't hypothetical.** A one-hot category or NPI flag
crossed with a claim type it never co-occurs with produces an interaction column that's constant
at `0` across the entire dataset — not a small-sample precision problem, a non-identifiable
parameter (the same failure mode already documented for claim-type-exclusive fields, just
discovered empirically here instead of known in advance from the schema). Manually checking
`HCPCS_CD × DME` first (66,335 DME rows, the smallest claim type) confirmed this was real, not
theoretical: 13 of 21 HCPCS top-N categories have zero DME activations. The guard was written to
catch this generally, for every genuinely-3-of-3 group against all three claim types, not just
the one case checked by hand.

**Three real memory crashes fixing THIS one gap, each smaller than the last, each with a distinct
root cause worth keeping visible rather than collapsing into "fixed a bug":**
1. **888 MiB allocation failure** — pandas fragments its internal per-dtype blocks when new
   columns are assigned one at a time inside a loop (`df[new_col] = ...`), and periodically has to
   consolidate them, needing a temporary array the size of the whole block. Fixed by building new
   columns into a dict and doing exactly one `pd.concat()` per function instead.
2. **Same class of error, next line down** — a leading `df = df.copy()` at the top of a function
   independently triggers the same block-consolidation cost, regardless of the fix above, if the
   input it's copying already arrives fragmented. Fixed by removing `add_claim_type_interactions()`'s
   leading copy entirely, once traced through to confirm the function never mutates its input in
   place (every step reassigns the local name via `drop()`/`concat()`) — `build_restricted_design_matrix()`'s
   copy stays, since that function genuinely does mutate columns in place and both functions share
   the same input object in `__main__`.
3. **43.9 MiB allocation failure** — small enough to indicate the real, underlying problem: every
   one-hot dummy and interaction term was stored as `int64` (8 bytes/cell) to hold a value that's
   always exactly `0` or `1`. Fixed by downcasting every genuinely-binary source column
   (`claim_type_*`, the 5 NPI flags, every `get_dummies()` output) to `int8` (1 byte/cell) at
   creation — an 8x memory reduction for this entire class of column, with every interaction term
   built from them automatically inheriting `int8` (numpy keeps `int8 * int8 -> int8` for values
   this small, no overflow risk). `int8`, not `bool`: the two are the *same* size in numpy (1
   byte/cell either way — memory is byte-, not bit-addressable), chosen for being the more
   conventional dtype heading into a numeric design matrix.

Final result after all three fixes: 446 unrestricted columns (307 interaction terms), 280
restricted columns, ~3.3 GB / ~3.0 GB memory footprint, completed without error.

**The 39 zero-variance terms actually found — far more than the 13 anticipated from the manual
`HCPCS × DME` check, and each one traced to a real, defensible structural cause, not noise:**

*10 more HCPCS drops, all DME-supply codes (`A4253`, `A4259`, `A4604`, `A7034`, `A7035`, `A7037`,
`A7038` — carrier AND outpatient both zero, DME survives; `H2001`, `99241` — carrier and DME both
zero, outpatient survives) plus `__MISSING__ × {outpatient, dme}`.* These are consistent with,
not contradictory to, findings already established elsewhere in this document: the `A`-series
codes are catheter/CPAP-supply DME codes (the same family flagged in the original
`dx_procedure_mismatch` mapping work); `99241`'s carrier-zero result was checked directly against
`train.parquet` and confirmed (61,858 outpatient rows, 0 carrier rows) — consistent with, not
contradicting, the original `deprecated_code` discovery (a real federal-policy fact about the
code's payment status, independent of which claim type happens to carry it in this synthetic
data). `HCPCS_CD`'s `__MISSING__` bucket being zero in DME/outpatient confirms the
already-documented finding that missing-`HCPCS_CD` is a carrier-specific phenomenon in this
dataset (the same 62.5%-within-carrier gap `missing_hcpcs` was built around) — DME and outpatient
claims apparently never leave this field blank.

*A real correction to this section's own Section-3-cited claim that "5 NPI-presence flags —
confirmed genuinely shared (3-of-3, 0% null in all three claim types)."* That claim checked only
whether the flag is ever `NaN` (true — it's a derived `.notna().astype(int)`, always exactly `0`
or `1`) — never whether it's *constant* within a claim type, which is a different question the
null-rate check can't answer. Checked precisely against which interaction terms this guard
actually kept versus dropped for each flag:

| NPI flag | Terms kept | True status |
|---|---|---|
| `has_referring_physician` | carrier, dme | genuinely 2-of-3 (outpatient dropped) |
| `has_performing_physician` | carrier only | **empirically 1-of-3**, not 3-of-3 |
| `has_attending_physician` | outpatient only | **empirically 1-of-3**, not 3-of-3 |
| `has_operating_physician` | outpatient only | **empirically 1-of-3**, not 3-of-3 |
| `has_rendering_physician` | outpatient only | **empirically 1-of-3**, not 3-of-3 |

Four of the five flags turn out to be empirically claim-type-exclusive, not shared — a materially
different, and cleaner, real-world story than "shared everywhere" once actual CMS RIF semantics
are checked: `PRF_PHYSN_NPI` (performing) is a Part B professional/carrier billing concept, while
`AT`/`OP`/`RNDRNG_PHYSN_NPI` (attending/operating/rendering) are institutional/outpatient
concepts — `RFR_PHYSN_NPI` (referring) is the one flag that genuinely spans both professional
contexts (carrier and DME), consistent with referrals driving both physician services and DME
orders. This is the same category of finding as `PRVDR_SPCLTY`/`CARR_NUM`'s carrier-DME pairing
documented earlier in this file — a real structural fact about the RIF schema, checked and
confirmed, not assumed.

**Degrees-of-freedom consequence, precisely.** A dummy losing 1 of its 3 potential interaction
terms this way now contributes `n_i-1=1` degree of freedom to the omnibus test (same as a genuine
2-of-3 covariate); one losing 2 of 3 (the four NPI flags above, `99241`, `H2001`, and the
`__MISSING__` bucket) contributes `n_i-1=0` — the same as a genuine 1-of-3 field, and correctly
excluded from the hypothesis being tested for the same reason. This is the exact same formula
established earlier in this document, now confirmed to apply to empirically-discovered exclusivity
exactly as it does to schema-known exclusivity — no new rule needed, just a wider set of covariates
it turns out to govern.

**Status: implemented and pushed, fully verified.** All ~161 genuinely-shared covariates are now
correctly represented in both design matrices — interacted (minus the 39 empirically-zero terms)
in the unrestricted matrix, single plain columns in the restricted one. This section's original
"Not yet done" note about this scope gap is now closed. What remains open: the actual Chow-test
fitting code (two `statsmodels.Logit` calls + the likelihood-ratio statistic) — not yet written.

### Both design matrices were rank-deficient: every dependency named, and removed (2026-09-23 to 2026-09-24)

**How it surfaced.** The first `--method firth` smoke test refused to fit: `firthmodels` raised
"Weighted design matrix is rank deficient." `src/check_rank.py` then showed the *raw* matrices were
rank-deficient, not just the IRLS-weighted one: **restricted 144 columns / rank 119 (deficiency
25), unrestricted 310 / rank 275 (deficiency 35)** at `--sample-frac 0.2`. Two theories were ruled
out by direct checks rather than argued away:

- **Separation-driven weight collapse** (the 6 separating HCPCS codes pushing IRLS weights to zero):
  ruled out, since the deficiency was identical with and without `--exclude-separating-codes`.
- **A small-sample artifact of computing the top-20 encoding on a subsample**: ruled out, since the
  deficiency was identical at `--sample-frac 0.02` and `0.2` (10x the data).

This had been present in every earlier fit. LBFGS optimizes without inverting anything, so it
pushed through silently, and the `HessianInversionWarning` on every earlier run was this issue, not
separation. Firth's Newton-Raphson has to invert at every iteration, which is why it failed loudly.

**Correction to `check_rank.py`'s own NPI test, kept visible.** An intermediate check reported
`has_attending + has_operating + has_rendering == claim_type_outpatient` for only 67.8% of rows and
called the hypothesis "refuted." The test was wrongly designed. If each of the three flags equals
`claim_type_outpatient`, their *sum* is 3 on outpatient rows, so the equality can only hold on
non-outpatient rows; 67.8% was simply the non-outpatient share. That was evidence *for* the
hypothesis. `src/explain_dependencies.py` replaced ad hoc tests with explicit equations: first
columns constant everywhere, then columns constant *within* each claim type (exact combinations of
the claim-type dummies), then orthogonal matching pursuit for the sparsest exact equation behind
each remaining dependency. It accounts for **all 25 restricted and all 35 unrestricted**
dependencies (11+9+5 and 14+10+11), each verified to a relative residual of ~1e-15.

| Group | Equation(s) | What it means |
|---|---|---|
| Zero-everywhere fields (11 restricted / 14 unrestricted) | MSP 1st/2nd paid, blood deductible, MCO-paid switch, bene-payment amounts, DME screening savings, reduced-payment physician assistant `= 0` | Synthea never populates these. |
| All 5 NPI flags | `has_performing = carrier`; `has_referring = carrier + dme`; `has_attending = has_operating = has_rendering = outpatient` | No information beyond claim type (see correction below). |
| Constant within claim type | `CLAIM_QUERY_CODE = 3·outpatient`; `NCH_PROFNL_CMPNT_CHRG_AMT = $4·outpatient`; `REV_CNTR_UNIT_CNT = 1·outpatient`; `LINE_BENE_PRMRY_PYR_PD_AMT = $1·dme` ($0 in carrier) | Fixed values on every claim of that type, most likely Synthea constants. |
| Outpatient payment = charge | `CLM_OP_PRVDR_PMT_AMT = REV_CNTR_PRVDR_PMT_AMT = CLM_TOT_CHRG_AMT` | Synthea pays the full charge with no contractual adjustment. Same pattern as the carrier `SBMTD = ALOWD` duplicates (Section 2, 2026-09-16). |
| Outpatient deductible duplicate | `REV_CNTR_CASH_DDCTBL_AMT = NCH_BENE_PTB_DDCTBL_AMT` | The same amount recorded at line level and claim level. |
| Outpatient cost-sharing identity | `COINSRNC_WGE_ADJSTD = RDCD_COINSRNC = PTNT_RSPNSBLTY − PTB_DDCTBL` | A genuine accounting identity: patient responsibility = deductible + coinsurance. |
| DME duplicates (unrestricted only) | `LINE_PRMRY_ALOWD_CHRG = LINE_ALOWD_CHRG`; `CARR_CLM_PRMRY_PYR_PD = NCH_CARR_CLM_ALOWD` (within DME) | Duplicates within DME only. They differ in carrier, which is why the pooled matrix doesn't show them. |
| HCPCS dummy trap within claim type (unrestricted only) | `Σ hcpcs_*__x__carrier = claim_type_carrier`, and the same for DME | `drop_first` chose a single global reference code that never occurs in carrier or DME. It protects the pooled matrix but not the per-claim-type blocks. |
| `carr_num_freq` = state frequency (unrestricted only) | coefficients ≈ each state's own share (CA 0.099, FL 0.094, NY 0.061…) | Each state maps to one carrier number, so the encoding is a function of state, exactly in the span of the ~50 state dummies within each claim type. It looked "not sparse" only because the solver stopped at 25 terms. It escaped the restricted matrix because the outpatient zero-fill breaks the identity there. |

**A correction to this section's own 2026-09-22 NPI table, kept visible above rather than
edited.** That table classified 4 of the 5 NPI flags as "empirically 1-of-3" and
`has_referring_physician` as "genuinely 2-of-3." Both undersold it. The flags are not merely
claim-type-*exclusive*; each is *identically equal* to a claim-type dummy (or to the sum of two)
on every row. The 2026-09-22 check looked at which interaction terms were non-zero, not at whether
the kept terms varied, so it couldn't see this. The RIF-semantics explanation given there
(performing = professional/carrier; attending/operating/rendering = institutional/outpatient;
referring spans carrier and DME) still holds. It just means that, in this synthetic data, every
claim of a type has exactly the role fields that type uses, with no variation left to learn from.

**Consequence for the Chow test itself.** With singular matrices, the LR degrees of freedom are
`rank(unrestricted) − rank(restricted)`, not the column-count difference. On the matrices above
that is 275 − 119 = **156**, not the 166 `fit_chow_test.py` reported. Its name-based
`Σ(n_i − 1)` cross-check was internally consistent but measured the wrong thing whenever either
matrix was rank-deficient. No Chow-test result produced before 2026-09-24 should be used.

**Fixes, all in `src/build_chow_design_matrix.py` (baseline/Chow only; `train_model.parquet`
untouched, since XGBoost doesn't need any of this):**

- **A. Claim-type-determined columns: one general rule, not a name list.** A column is dropped when
  it takes the same value on every row of each claim type it's present in. **"Present" means
  non-null, and a 0 is a present value** (the standing "0 is a value, NaN is the absence of one"
  rule, restated explicitly when this fix was approved). So a field that is non-null and always
  exactly 0 within a claim type counts as constant there, while a claim type where the field is
  100% NaN is absent and not judged. This one rule removes the zero-everywhere fields, the 5 NPI
  flags, and the four constant-within-claim-type fields. Two edge cases:
  - *Partial presence with a nonzero constant*: kept, because the zero-filled term is then
    `c × presence`, which varies.
  - *Partial presence where every present value is 0*: dropped (it becomes an all-zero column
    after the existing zero-fill) but printed as a warning. The zero-fill is conflating a present 0
    with absence in that case, a pre-existing issue outside this fix.
- **A (per claim type, unrestricted only).** A column constant in *some* of its claim types keeps
  its other terms; only the constant claim type's `__x__` term is skipped. Pass 2's zero-variance
  guard is generalized the same way: a term is skipped when constant within the claim type (all 0
  *or* all 1), not only when all 0.
- **B. Named exact identities, verified on the actual data every run** (the build raises if any
  stop holding), then the redundant side dropped. Kept representatives are `CLM_TOT_CHRG_AMT`,
  `NCH_BENE_PTB_DDCTBL_AMT`, `REV_CNTR_RDCD_COINSRNC_AMT`, `LINE_ALOWD_CHRG_AMT` and
  `NCH_CARR_CLM_ALOWD_AMT`. Only the `__x__dme` term is dropped for the two DME duplicates, so the
  distinct carrier information survives.
- **C. Within-claim-type dummy trap.** When a dummy group's terms in one claim-type block sum to
  exactly that claim-type dummy, one term is dropped as the within-type reference. It is applied
  uniformly to `state_`/`hcpcs_`/`dgns_`.
- **D. `carr_num_freq` removed**, as redundant with `provider_state`.

**Two further changes in `src/fit_chow_test.py`:**

- **Separating-code exclusion moved upstream.** It now drops the 6 codes' dummies from the shared
  intermediate before either matrix is built, instead of post hoc per matrix. The end result for
  the codes is the same (they merge into the reference category in both), but it has to come first
  now. Otherwise excluding a code *after* fix C chose that block's reference could remove a second
  term from the same block, leaving the restricted model's pooled reference column outside the
  unrestricted span, a silent nesting violation.
- **Full rank is enforced before fitting.** Rank is computed with chunked tall-skinny QR, so no
  full float64 copy is needed on the 8 GB machine. The script refuses to fit a rank-deficient
  matrix and requires name-based df = column-count difference = `rank(U) − rank(R)` before the
  unrestricted fit starts. Restricted-fit summaries saved before this change are rejected as stale.

**Tested on synthetic data before touching real data, and the test caught a real problem.** A
synthetic frame planted every dependency type above. The first run showed the name-based df
breaking: fix C had picked the block's most frequent term as reference, and in DME that was a
DME-only code (like the real `A4253`). That left the category with zero unrestricted terms but a
pooled restricted column. Nesting still held, but the df bookkeeping didn't. The fix:

- The within-type reference now prefers a category that also has a term in another claim type.
  The real carrier block has codes shared with outpatient, and the real DME block has `__OTHER__`.
- The df accounting counts the unavoidable fallback case as n_i = 0 (−1 df), which is exactly
  consistent with the column counts.

After the fix, with and without the exclusion, both matrices were full rank, the restricted
columns lay in the unrestricted span (checked directly), all three df computations agreed, and
LR ≥ 0. Broken identities and mismatched NaN patterns were confirmed to raise rather than drop.

**Status: implemented and pushed; not yet run against real data.** Next: `check_rank.py` on real
data (expect full rank on both matrices), then the `--method firth` vs `--method standard
--exclude-separating-codes` comparison. This will be the first Chow-test result produced on
correctly-specified matrices.

### Accounting for the restricted matrix's 117 columns (not the predicted 118) (2026-09-24)

**The gap.** The explained dependencies predicted 118 restricted columns: 144 original columns,
minus the 25 dependencies `explain_dependencies.py` named, minus `carr_num_freq` (fix D). On real
data (`--sample-frac 0.2`) the rebuilt restricted matrix has **117**. The rebuilt matrix is full
rank, so the extra drop didn't break anything. It still had to be accounted for rather than
assumed.

**Method.** I printed `build_chow_design_matrix()`'s `df.attrs["rank_fix_report"]` on the 0.2
sample. Then I diffed the rebuilt restricted column set against two things: the 144-column list
saved in the pre-fix `chow_restricted_fit_summary.json`, and the 25 left-hand sides written out
in `reports/rank_deficiency_explained__restricted.txt`.

**Result: two differences, and only one of them is a real extra drop.**

1. **A representative swap, not an extra drop.** For the outpatient cost-sharing identity,
   `explain_dependencies.py` (OMP) named `REV_CNTR_RDCD_COINSRNC_AMT` and
   `REV_CNTR_COINSRNC_WGE_ADJSTD_C` as the dependent columns and kept
   `REV_CNTR_PTNT_RSPNSBLTY_PMT`. Fix B does the opposite: it keeps `REV_CNTR_RDCD_COINSRNC_AMT`
   and drops `REV_CNTR_PTNT_RSPNSBLTY_PMT` along with the wage-adjusted duplicate. Four columns
   linked by two identities lose two columns either way, and both choices span the same space. The
   count is the same.
2. **The real extra column is `LINE_PRMRY_ALOWD_CHRG_AMT__x__dme`.** This is the within-DME
   duplicate from fix B (`LINE_PRMRY_ALOWD_CHRG_AMT == LINE_ALOWD_CHRG_AMT` within DME).
   `build_restricted_design_matrix()` skips it deliberately, "so both matrices stay
   name-consistent." It was *not* a rank dependency in the restricted matrix, which is why
   `explain_dependencies.py` never listed it and the prediction missed it.

**It was not a `zero_with_gaps` case.** The report's `zero_with_gaps_warning` list is empty. I
checked on the full population (1,151,951 rows): `LINE_PRMRY_ALOWD_CHRG_AMT` is 100% present in
DME, equals `LINE_ALOWD_CHRG_AMT` on every DME row, and is 100% NaN (absent, not zero) in carrier
and outpatient. No present 0 is being conflated with absence here.

**Why the extra drop is correct, and what it reveals about the pre-fix test.** Because
`LINE_PRMRY_ALOWD_CHRG_AMT` exists only in DME, the restricted builder treated it as
claim-type-exclusive (n_i = 1) and gave it a `__x__dme` column. Numerically, though, that column
*is* `LINE_ALOWD_CHRG_AMT × claim_type_dme`. So the old restricted matrix held both the pooled
`LINE_ALOWD_CHRG_AMT` and its DME-specific slope, which together span the carrier and DME slopes
separately. **The pooling restriction on `LINE_ALOWD_CHRG_AMT` was silently vacuous in every
pre-fix Chow run.** The name-based df counted it as one restriction, but the restricted model
could fit that difference freely. Dropping the duplicate from the restricted matrix is what
actually imposes the restriction. Keeping it would also break the name-based df check: it would
be a restricted-only column with no unrestricted counterpart, and `_compute_df_and_table()` would
correctly raise.

**Correction to the 2026-09-24 table above, kept rather than edited.** The "DME duplicates" row
says the two pairs "differ in carrier." That is true for `CARR_CLM_PRMRY_PYR_PD_AMT` vs.
`NCH_CARR_CLM_ALOWD_AMT`: both are present in carrier, and they are equal on only 18.3% of carrier
rows. It is **not** true for `LINE_PRMRY_ALOWD_CHRG_AMT`, which is *absent* in carrier (100% NaN),
not different there. That difference is why the two duplicates behave differently in the
restricted matrix. `CARR_CLM_PRMRY_PYR_PD_AMT` stays a pooled 2-of-3 column, while
`LINE_PRMRY_ALOWD_CHRG_AMT` is DME-exclusive and gets dropped.

**Full reconciliation:** 144 − 25 explained − 1 (`carr_num_freq`) − 1
(`LINE_PRMRY_ALOWD_CHRG_AMT__x__dme`) = **117**. This matches the real data.

### First Chow-test results on the full-rank matrices, and two fitting problems found on the way (2026-09-24)

**Result.** On the full train set (1,151,951 claims), Firth's penalized likelihood-ratio test
**rejects H0**. The pooled model is not adequate for every shared covariate: **LR = 4,732.5 on
158 df, p ≈ 0** (below float precision). df = 158 is confirmed four independent ways: the
name-based Σ(n_i − 1), the column-count difference (275 − 117), the rank difference (both
matrices full rank), and the number of tested coefficients. The unpenalized log-likelihoods at the
same estimates (−273,281.4 restricted vs. −270,897.8 unrestricted) differ by a similar 4,767.1, so
the rejection does not come from the penalty. The **standard-MLE-with-exclusion comparison could
not produce a valid test**. See "Separation is much wider than 6 codes" below.

**Problem 1: every earlier standard-method fit was an optimizer failure, not a result.**
statsmodels' `lbfgs` never moved off its starting point, β = 0. At β = 0 every predicted
probability is 0.5 and the log-likelihood is exactly n·ln(0.5). Every "converged" standard fit
reported exactly that value: −15,969.42 at `--sample-frac 0.02` (n = 23,039) and **−159,694.87 at
0.2 (n = 230,391), the same number as the pre-fix `chow_test_results.txt`**. It is far worse than
simply predicting the 12.1% base rate for everyone (about −85,170 at 0.2). The likely cause is
dollar columns up to ~$410,000 sitting next to 0/1 dummies: the first step overflowed `exp()` (the
RuntimeWarnings on every run), and the optimizer's own convergence flag didn't catch it. **The
earlier "identical log-likelihoods, LR = 0" finding, which `diagnose_separation.py` was written to
explain, was this failure.** The separation it found is real (below), but separation is not what
made the two log-likelihoods identical.

**Problem 2: two separately-penalized Firth fits are not Firth's penalized LR test.** Each Firth
fit adds 0.5·log|X′WX| for its *own* design. The two models have different dimensions and
parameterizations, so the difference of two separately-penalized log-likelihoods mixes a
penalty-difference term into the statistic. On the same 0.02 data that approach gave **339.1**, vs.
**213.2** for the proper test. The standard penalized LR test (Heinze & Schemper 2002; what
`logistf` and `firthmodels`' own `.lrt()` do) fits the restricted model *as the full model with the
tested coefficients held at 0*, still penalized by the full model's information.

**The fix: put the Chow test in "full model, some coefficients = 0" form.** For a shared covariate
`x`, {x_pooled, x·D_2, x·D_3} spans the same space as {x·D_1, x·D_2, x·D_3}. So
`U′ = [every restricted column, Z]`, where Z is the unrestricted-only interaction terms minus one
per pooled covariate, *is* the unrestricted model. H0 is then exactly "Z's coefficients = 0". This
is verified, not assumed, on every run:
- U′ is full rank.
- U′ has as many columns as the unrestricted matrix.
- Every unrestricted term left out of Z lies in span(U′).

On the 0.02 data, U′'s penalized log-likelihood equals the original unrestricted matrix's to
1e-12.

**Problem 3: memory.** `firthmodels` allocates three extra k × n float64 buffers besides X, about
10 GB at full size. New `src/chunked_logit.py` accumulates everything Newton-Raphson needs (X′WX,
the Firth hat diagonals, the modified score) over row chunks, on column-scaled data. It is
validated against `firthmodels` (penalized LL equal to ~1e-12, and Stage 2 p-values equal to
~1e-11) and against statsmodels' Newton fits (~1e-8). Peak RSS for a full-size fit is about 4.9 GB.
That came after one OOM kill, fixed by not loading the ~130 raw columns the design matrices never
use (verified to give identical matrices). `fit_chow_test.py` now refuses to compute an LR from any
fit that reports non-convergence.

**Separation is much wider than 6 codes, and `--exclude-separating-codes` cannot remove it.** On
the full train set:

| Within carrier (718,074 claims) | Claims | Denials |
|---|---|---|
| HCPCS null (`__MISSING__`) | 448,567 | 31,250 |
| G0444, 96127, G0442 (the only codes with any denial) | 108,977 | 9,046 |
| **The other 34 codes**, including the known 6 plus **G8839** (6,036) and every `__OTHER__` code (4,906) | **160,530** | **0** |

So 13.9% of the whole train set sits in carrier HCPCS cells with an exact 0.000 denial rate, not
just the 6 codes `diagnose_separation.py` flagged. Excluding the 6 codes' dummies also doesn't
remove their claims. It merges them into carrier's reference cell, which then has zero denials
itself. Run on the full data, the standard exclusion fit behaves accordingly:
- The restricted (pooled) fit converged (LL −274,120.89, 80 iterations).
- The unrestricted fit showed the textbook separation signature: log-likelihood flat at
  −271,242.34, score → 1e-10, Newton steps stuck at ~1 per iteration, then a singular information
  matrix.

The MLE does not exist, so there is no valid χ² test from it. The supremum-based "LR" (≈ 5,757,
153 df) points the same way but is not reported as a result. **Firth is the primary (and only
valid) Stage 1 result.** Its caveat stands (`_fit_logit_firth` docstring): under true separation,
the χ² reference distribution for the penalized LR is simulation-validated, not proven. With
LR = 4,732.5 on 158 df, no plausible reference distribution changes the conclusion.

**The likely root cause is in the label, not the features. It is an open decision, not changed
here.** `rule_provider_outlier` and `rule_duplicate_claim` (`denial_rules.py`) both group on
`PRVDR_NUM`, which is **100% null for carrier claims** (verified on the full train set: null share
1.0 carrier, 0.0004 outpatient, 0.0 DME). pandas' `groupby(dropna=True)` drops those rows, so
neither risk factor can ever fire for a carrier claim. That leaves carrier denials driven only by
`missing_hcpcs` (the `__MISSING__` cell) and code-specific rules that apply to just three codes.
The carrier denial rate is 5.6%, vs. 24.1% outpatient and 16.3% DME. Any fix changes `is_denied`
and invalidates every result above; see `reports/SESSION_LOG.md`.

### Stage 2: which shared covariates actually differ by claim type (2026-09-24)

Stage 1 rejected H0, so Section 3's Stage 2 ran (`src/fit_chow_stage2.py`, Firth, full train set).
**Each variable is tested on its own.** Only that variable's tested coefficients are held at 0,
with the full model's penalty kept, which is the same penalized LR as Stage 1 applied to one block.
The three categorical groups are tested as whole variables. The 11 variables' df sum to exactly
Stage 1's 158. Every constrained fit converged. p-values are Holm-adjusted across the 11 tests.
The machinery was checked on the 0.02 data against `firthmodels`' own `.lrt()`, and the p-values
match to ~1e-11.

| Variable | df | LR | Holm p | Verdict |
|---|---|---|---|---|
| HCPCS_CD | 10 | 3,260.4 | ≈ 0 | interact |
| PRNCPAL_DGNS_CD | 40 | 825.4 | 2.5e-146 | interact |
| prvdr_num_freq | 1 | 158.7 | 1.9e-35 | interact |
| provider_state | 100 | 327.9 | 3.8e-25 | interact |
| LINE_SRVC_CNT | 1 | 21.0 | 3.2e-05 | interact |
| CARR_CLM_CASH_DDCTBL_APLD_AMT | 1 | 8.6 | 0.020 | interact |
| NCH_CARR_CLM_SBMTD_CHRG_AMT | 1 | 6.2 | 0.065 | pool |
| LINE_BENE_PTB_DDCTBL_AMT | 1 | 4.2 | 0.16 | pool |
| LINE_ALOWD_CHRG_AMT | 1 | 0.8 | 1 | pool |
| LINE_NCH_PMT_AMT | 1 | 0.4 | 1 | pool |
| NCH_CARR_CLM_ALOWD_AMT | 1 | 0.3 | 1 | pool |

**How to read this before building the mixed model.**
- **HCPCS_CD's huge LR is at least partly the carrier separation above, not a pricing or
  clinical effect.** 34 of 37 carrier codes can't be denied at all under the current label. So
  any HCPCS code shared with outpatient necessarily "behaves differently" in carrier.
- **`prvdr_num_freq` compares outpatient and DME only**, because `PRVDR_NUM` is absent in carrier.
- **The five "pool" verdicts are all carrier/DME dollar fields.** Their DME slopes are not
  distinguishable from their carrier slopes.

**Not acted on yet.** Section 3's mixed model is: interact the 6 rejected variables, and pool the
5 others. It should wait for the `provider_outlier` / `duplicate_claim` decision
(`reports/SESSION_LOG.md`). If carrier's label changes, the Stage 1 and Stage 2 results change
with it.
