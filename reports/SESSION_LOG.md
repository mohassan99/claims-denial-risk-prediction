# Session log

Plain-language record of each working session: what ran, what it means, what changed, and what
needs a decision. Newest entry at the bottom.

---

## 2026-09-24: Chow test, first valid result; the label decision is now blocking

### Results

**The Chow test rejects pooling.** On the full train set (1,151,951 claims), Firth's penalized
likelihood-ratio test gives **LR = 4,732.5 on 158 degrees of freedom, p ≈ 0**. At least some
shared covariates affect denial risk differently by claim type.

df = 158 was confirmed four independent ways: the name-based formula, the column counts
(275 − 117), the matrix ranks (both full rank), and the number of tested coefficients. The
unpenalized log-likelihoods differ by a similar amount (4,767), so the conclusion is not an
artifact of the Firth penalty.

**Stage 2: 6 of 11 shared variables differ by claim type.** Each variable was tested on its own,
with Holm correction. These need claim-type-specific effects:
- HCPCS code
- principal diagnosis
- provider frequency
- state
- line service count
- carrier cash deductible

The other 5 can share one coefficient across claim types. All five are carrier/DME dollar fields.
The full table is in `data/FEATURE_ENGINEERING.md` Section 6, and
`reports/chow_stage2_results__firth.txt` has the raw output.

**The standard-MLE-with-exclusion comparison produced no valid test, and it can't as currently
built.** Its unrestricted fit shows textbook separation on the full data: the log-likelihood goes
flat, the coefficients run off to infinity, and the information matrix becomes singular. So Firth
is the primary result. This is not a Firth-vs-exclusion disagreement: the exclusion run has no
test statistic to disagree with. See "Why" below.

### What I found along the way (all fixed or documented)

1. **Task 1: restricted matrix has 117 columns, not 118. Explained.**
   - The extra drop is `LINE_PRMRY_ALOWD_CHRG_AMT__x__dme`. It is a DME-only exact duplicate of
     `LINE_ALOWD_CHRG_AMT`, skipped on purpose by the restricted builder. It was **not** a
     zero_with_gaps case: the field is 100% present in DME and absent (NaN, not 0) elsewhere.
   - Side finding: in every earlier run, this duplicate made the pooling restriction on
     `LINE_ALOWD_CHRG_AMT` do nothing.
   - The other difference in the diff is just a different (equivalent) choice of which
     cost-sharing column to keep.

2. **Every earlier standard-method result was an optimizer failure.** statsmodels' lbfgs never
   left its starting point.
   - The old "log-likelihood −159,694.87" is exactly 230,391 × ln(0.5): every claim given a
     50% denial probability.
   - The earlier "restricted and unrestricted log-likelihoods identical, LR = 0" finding came
     from that failure, not from the data.

3. **The Firth comparison as coded wasn't the right test.** Two separately-penalized fits give
   339 vs. 213 for the proper penalized LR test on the same data. firthmodels also needs ~10 GB
   at full size, which is more than the 8 GB machine has.
   - **Fix:** new `src/chunked_logit.py`, a Newton-Raphson solver that works in row chunks.
     Peak memory is ~4.9 GB at full size. It matches firthmodels to 1e-12 and statsmodels to
     1e-8.
   - `fit_chow_test.py` now tests "full model with the tested coefficients held at 0," which is
     the standard form. It also refuses to report an LR from any fit that didn't converge.

4. **Separation is far wider than the 6 known codes.** Within carrier, **34 of 37 HCPCS codes
   have zero denials** across the whole train set: 160,530 claims, 13.9% of train. That includes
   G8839 (6,036 claims) and every "other" code. Only G0444, 96127 and G0442 (plus claims with no
   HCPCS) ever get denied in carrier. Excluding the 6 codes' dummies merges those claims into
   carrier's reference group, which then also has zero denials. That's why the exclusion method
   can't work.

5. **One OOM crash on the first full run.** Fixed by not loading ~130 raw columns the design
   matrices never use. The resulting matrices were verified identical.

### Decision needed from you (blocking everything downstream)

**The label bug is confirmed, and it's bigger than suspected.** Both `rule_provider_outlier`
*and* `rule_duplicate_claim` in `denial_rules.py` group on `PRVDR_NUM`.
- `PRVDR_NUM` is **100% empty for carrier claims**. On the full train set the empty share is
  100% for carrier, 0.04% for outpatient, and 0% for DME.
- pandas drops empty keys when grouping, so **neither rule can ever fire for a carrier claim.**
  That's ~62% of the data.
- Carrier's denial rate is 5.6%, vs. 24.1% outpatient and 16.3% DME.
- This is the most likely reason 34 carrier codes have zero denials.

**Options:**
- **A. Recommended: fix both rules for carrier, then rerun everything.** Carrier claims would
  group on a carrier-side provider identifier instead.
  - I checked the candidate fields on carrier claims in the train set.
    `CARR_CLM_BLG_NPI_NUM` (billing provider NPI) is the strongest: 0% empty, 5,106 distinct
    providers, a median of 75 claims each. That's close to how `PRVDR_NUM` behaves for
    outpatient.
  - `CARR_NUM` is a poor choice: only 51 distinct values (one Medicare contractor per state),
    not a provider.
  - `ORG_NPI_NUM` shows the same 5,106 distinct values; not yet checked whether it's a duplicate
    of the billing NPI.
  - Cost: `is_denied` changes; the label's calibration (the 12.1% rate) must be redone; every
    result above has to be rerun (~1 hour of compute).
  - Benefit: the label no longer has a structural hole in 62% of the data, and most of the
    separation problem likely goes away.
- **B. Keep the label and document the limitation.** Proceed with the current results.
  - Cost: HCPCS and carrier effects will partly reflect the label's blind spot rather than
    denial risk. An interviewer who spots it will ask.
- **C. Fix `duplicate_claim` only, or `provider_outlier` only.** Possible, but both have the
  same cause, so a partial fix is hard to defend.

**Two smaller decisions, unchanged from before:**
- **The 3 fixed-constant fields** (`CARR_CLM_PMT_DNL_CD`, `CLM_DISP_CD`,
  `CLM_MDCR_NON_PMT_RSN_CD`). My recommendation: wait until the label decision. If you choose A,
  fold them into the same `build_features.py` rerun, since it has to happen anyway.
- **~130 unencoded raw columns.** No change. Secondary diagnosis codes are still the group most
  worth encoding.

### What I did not do, and why

- **I did not build the mixed baseline model, PR-AUC metrics, XGBoost, or SHAP (task 6).** All
  of it sits on `is_denied`, and option A would change it. That would throw away the work.
- **I did not change the exclusion flag's behavior.** It's kept as is and documented as unable to
  remove separation. The fix that would work drops the zero-denial carrier claims from the
  estimation sample, which is a Chow-test scope change and yours to make. After option A it may
  not be needed at all.

### Housekeeping

- I ran from a cloud workspace this session: same code, same data, same pinned packages, Linux
  Python 3.11. The local shell on your machine wouldn't start.
- Result files are in `reports/`, uncommitted, per the earlier convention: `chow_test_results__firth.txt`,
  `chow_stage2_results__firth.txt/.csv`, and the fit summaries.
- **To reproduce on your machine** (each step fits in 8 GB; peak ~4.9 GB):
  - `python src/fit_chow_test.py --method firth --restricted-only`
  - then `python src/fit_chow_test.py --method firth --skip-restricted`
  - then `python src/fit_chow_stage2.py --method firth`

---

## 2026-09-24 (continued): label fixed (your decision: option A), everything downstream rebuilt

### Results

**The carrier label bug is fixed.** `provider_outlier` and `duplicate_claim` now identify carrier
providers by their billing NPI instead of the facility provider number, which is empty on
every carrier claim.

| | Before | After |
|---|---|---|
| Overall denial rate | 12.1% | **14.9%** (inside the 10–15% target) |
| Carrier denial rate | 5.6% | **10.0%** |
| Outpatient / DME denial rate | 24.1% / 16.2% | **unchanged, claim for claim** |
| Big claim-type × code cells with zero denials | 9 | **0** |

Before touching anything, I re-ran the old pipeline from scratch and confirmed it reproduces your
existing files exactly. So every change in the table comes from the fix.

**The Chow test still rejects pooling, and now every method agrees.** Firth gives LR = 2,642 on
158 df, down from 4,732: a large share of the old statistic was the bug. Standard MLE gives 2,644,
within 0.1%. Standard MLE is technically non-convergent because of a single claim (diagnosis N186
on one carrier claim, not denied), and that claim barely moves the number. The separation
problem is gone, so the "exclude separating codes" workaround is no longer needed.

**Stage 2: 5 of 11 shared variables need claim-type-specific effects:** state, HCPCS, principal
diagnosis, provider frequency, and carrier cash deductible. The other 6 can be pooled.
- HCPCS's statistic fell 79%, because it had mostly been measuring the bug.
- Line service count flipped from "interact" to "pool" for the same reason.

### How this happened and what now prevents it

- **Cause.** The two rules were written against the facility claim layout. pandas silently drops
  empty group keys, and the label was only ever validated overall. The 5.6% vs 24% gap between
  claim types was visible in Phase 1, but nothing required it to be explained.
- **Guardrails now built in; each one fails the build rather than printing a warning nobody
  reads:**
  1. **Key-coverage check.** Before any rule groups on ID columns, the check confirms those
     columns are filled in for every claim type.
  2. **Per-claim-type firing table.** For every risk factor and claim type, the code must declare
     either "fires" or "zero, because …", and every build checks it. The old label fails this
     check on exactly the two cells the bug zeroed out.
  3. **`reports/label_audit.txt`.** Written on every build, it lists any large claim-type × code
     group with zero denials. The old label had 9; the new one has 0.
  4. **Label fingerprint on cached fits.** A model fit on an old label can't be silently reused.
- **Standing rule added to CLAUDE.md:** audit everything per claim type, never only overall.

### Rebuilt downstream

`train/val/test.parquet`, the `*_model.parquet` files, `labeled_claims_for_eda.parquet`, both Chow
stages, and the Phase 1 EDA figures (`run_eda.py`) were all rebuilt. The figures aren't tracked in
git, so I copied the regenerated ones straight into your `reports/` folder.

**Your local data files are still the old label.** The data folder is gitignored, so a pull won't
update it. To rebuild locally (about 5 minutes; Chow reruns are optional):

```
python src/build_target_and_split.py
python src/build_features.py
python src/run_eda.py
```

### Small decisions for you (nothing is blocked)

- **CALIBRATION_SCALE.** I kept it at 1.4, which gives 14.9%. That way only carrier labels
  changed, and the before/after is clean. Retuning to about 1.0 would bring the rate back near
  12.4%, but it would also change outpatient and DME labels. **My recommendation: keep 1.4.**
- **Minimum support for interaction terms.** One interaction term (N186 × carrier) rests on a
  single claim. A rule like "skip interaction terms with fewer than N claims" would make standard
  MLE converge cleanly, but it changes the Chow test's tested set and df. **My recommendation:
  leave it.** Firth handles it and the conclusion doesn't change.
- **The 3 fixed-constant fields and the ~130 unencoded columns:** unchanged from before.

### Next

Build the baseline mixed model: interact the 5 variables above and pool the other 6. Then
PR-AUC on validation, broken out by claim type, then XGBoost and SHAP.

---

## 2026-09-25: Phase 2 baseline mixed model, fit and evaluated on val

### Results

**Built the model Chow Stage 2 actually selected.** `src/build_chow_design_matrix.py` gained
`build_mixed_design_matrix()`: the 5 variables Stage 2 found claim-type-specific (provider_state,
HCPCS_CD, PRNCPAL_DGNS_CD, prvdr_num_freq, CARR_CLM_CASH_DDCTBL_APLD_AMT) stay as claim-type
interaction terms; the other 6 shared variables are pooled to one coefficient each; every
claim-type-exclusive covariate is unaffected either way. It reuses the existing, already-verified
unrestricted/restricted builders rather than re-deriving the same claim-type logic a third time.

**Fit with Firth's penalized MLE** (`src/fit_baseline_model.py`, `chunked_logit.ChunkedLogit`) on
the full train set (1,151,951 rows, 269 columns). Converged in 17 iterations. Standard MLE was
tried first and stalled (a rare-category near-separation, score near 0 but step stuck) --
switched straight to Firth after confirming the same pattern at `--sample-frac 0.2`, where Firth
converged cleanly in 12 iterations.

**Val-set metrics, never only overall:**

| | n | positive rate | PR-AUC | ROC-AUC | Brier | mean predicted |
|---|---|---|---|---|---|---|
| Overall | 287,988 | 14.88% | 0.557 | 0.772 | 0.088 | 0.148 |
| Carrier | 179,028 | 10.07% | 0.141 | 0.612 | 0.089 | 0.100 |
| Outpatient | 92,321 | 24.01% | 0.815 | 0.893 | 0.079 | 0.238 |
| DME | 16,639 | 16.04% | 0.262 | 0.706 | 0.125 | 0.163 |

Calibration is tight everywhere (mean predicted probability within ~0.2 points of the actual rate
in every claim type) -- expected, since the claim-type dummies act as per-claim-type intercepts.
Outpatient is the easiest claim type by a wide margin (its `deprecated_code` rule is close to
deterministic); carrier is the hardest, consistent with carrier's risk factors
(`missing_hcpcs`, `provider_outlier`) being the weakest-grounded ones in `denial_reasons.py`, not
a sign of a modeling problem.

### Train/val encoding consistency (new problem, not faced by the Chow test)

The Chow test only ever needed train. This is the first script needing two splits in the same
feature space, which raised two real risks, both fixed:
1. **Categorical vocabulary drift.** `top_n_encode`/`frequency_encode` in
   `build_chow_design_matrix.py` gained optional `categories=`/`freq_map=` params. Train fits them
   once (`return_encoders=True`); val reuses train's exact categories and provider frequencies,
   not its own -- a code just inside val's own top-20 but outside train's would otherwise silently
   get its own dummy instead of falling into `__OTHER__`, against coefficients that were never fit
   for it.
2. **Per-split constancy decisions.** The rank-deficiency fix that drops claim-type-determined
   columns could, in principle, disagree between splits (a field constant in train's DME rows by
   chance, not val's). Train's decision is now carried in the `encoders` dict and reused verbatim
   on val, with a printed note (not a build failure) if val's own data would have disagreed.
3. **Belt and suspenders:** val's final design matrix is reindexed to train's exact column list
   (fill 0) before prediction, regardless of the above. In this run: 1 column
   (`dgns_N186__x__carrier`, the single-claim interaction from the open minimum-support decision)
   existed in train and not val -- correctly zero-filled.

### One fixed bug: an out-of-memory kill on the first full-size run

The first full-run attempt built val's design matrix (train_intermediate + train_mixed + X_train
DataFrame + X_val DataFrame) all before fitting, then converted X_train to a fresh float64 numpy
array for `ChunkedLogit` -- at that moment X_train, X_val, and the new float64 copy were all alive
together, and the container's memory cgroup killed the process at 6.1 GB anon-rss. Fixed by
reordering: build train's matrix, convert to array, fit, discard the array -- THEN build val's
matrix, only once training's large arrays are gone. This follows the same discipline
`fit_chow_test.py` already used for its own two design matrices; this script just hadn't needed it
before doing a train+val combination for the first time. Full run after the fix: no OOM, ~4
minutes.

### Housekeeping

- Result files: `reports/baseline_model_results__firth.txt` (readable report),
  `reports/baseline_model_fit__firth.json` (coefficients, encoders, metrics -- for reuse when
  writing up the XGBoost comparison later).
- `--sample-frac 0.02` and `0.2` smoke tests both run and checked before the full run, per
  standing practice.

### Next

XGBoost on `train_model.parquet`/`val_model.parquet`, same metrics (PR-AUC primary, per claim
type), then SHAP.

---

## 2026-09-25 (continued): finished pushing the baseline model, then built and evaluated XGBoost

### Housekeeping: finished the interrupted push

The baseline-model commit from earlier this session (`09299dd` locally) needed pushing via the
GitHub MCP connector (the cloud session's own `git push` still hits the proxy 403). Split across
two `push_files` calls by mistake -- the first covered `CLAUDE.md`, `README.md`,
`reports/SESSION_LOG.md`, `reports/baseline_model_results__firth.txt` (commit `a2a5fa2`) but
missed `src/build_chow_design_matrix.py`, `src/fit_baseline_model.py`, and
`reports/baseline_model_fit__firth.json`; caught immediately via `get_commit`'s file list and
pushed those three in a follow-up commit (`d0936eb`) that says so in its own message. Verified both
landed with the right file counts, then `git fetch` + `git reset --hard origin/main` on the local
clone (GitHub is source of truth, same discipline as this session's two earlier duplicate-commit
fixes; local `09299dd` duplicated content now split across the two pushed commits under different
hashes).

### XGBoost (task queue item 3)

**Confirmed, not assumed, that XGBoost needs none of the baseline's encoding machinery** --
CLAUDE.md's task explicitly asked this be checked rather than skipped past. Reasoning (full detail
in `src/fit_xgboost.py`'s docstring):
- Top-n/frequency encoding (`build_chow_design_matrix.py`) exists only to keep a linear model's
  dummy-column count bounded. A tree splits on a raw categorical value directly, so `provider_state`,
  `HCPCS_CD`, `PRNCPAL_DGNS_CD`, `PRVDR_NUM`, `CARR_NUM` all go in at full cardinality via pandas
  `category` dtype + xgboost's native categorical split support (`tree_method="hist",
  enable_categorical=True`).
- The rank-deficiency fixes (constant-within-claim-type columns, exact linear identities) are also
  linear-model-only -- they cost a tree nothing, so none of that removal is reused.
- No `__x__claim_type` interaction terms either: a tree splits on `claim_type_*` directly and then
  splits differently per branch, which already gives every covariate an implicit claim-type-specific
  effect.
- Missing values are passed through as NaN, not zero-filled -- xgboost's hist method learns a
  default split direction per split, a real advantage over the baseline's forced zero-fill.
- **New capability the baseline didn't have:** the ~130 columns `fit_chow_test.py`'s `_PENDING_*`
  groups exclude (secondary/tertiary diagnosis & procedure codes, legacy provider ID strings, other
  CMS categorical codes, line sequence numbers) were excluded only because no cardinality-reduction
  scheme had been picked for them yet -- irrelevant for a tree, so they're included here. Raw dates
  stay excluded, same as the baseline (no Phase 2 date transform decided for either model).

Train/val category consistency handled the same way as the baseline's encoders: train's category
list per column is fit once and reused verbatim on val (a val-only value becomes NaN, not an error)
-- `_apply_categories()` in `fit_xgboost.py`.

**Fit:** `XGBClassifier(tree_method="hist", enable_categorical=True, max_depth=6,
learning_rate=0.1)`, early-stopped on val PR-AUC (`eval_metric="aucpr"`, patience 30) --
stopped at 89 trees of 500 allowed. Peak memory ~5.8 GB on the full 1,151,951-row train set (159
columns, 108 categorical), no OOM; ran in well under the full 500-tree budget.

**Val-set metrics, never only overall:**

| | n | positive rate | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|
| Overall | 287,988 | 14.88% | 0.5832 | 0.8161 | 0.0858 |
| Carrier | 179,028 | 10.07% | 0.1881 | 0.7061 | 0.0862 |
| Outpatient | 92,321 | 24.01% | 0.8200 | 0.9025 | 0.0781 |
| DME | 16,639 | 16.04% | 0.2603 | 0.7149 | 0.1238 |

**Beats the baseline logistic model in every claim type** (baseline: overall 0.5574/0.7724,
carrier 0.1407/0.6124, outpatient 0.8154/0.8927, DME 0.2616/0.7059 PR-AUC/ROC-AUC). The biggest
gain is on carrier -- still the hardest claim type for either model, but XGBoost closes a real
share of the gap (PR-AUC +33% relative, ROC-AUC +0.09). Outpatient and DME move less, consistent
with the baseline already capturing most of the signal there (`deprecated_code` is close to
deterministic in outpatient regardless of model).

**Feature importance, two things worth a closer look with SHAP (not blocking, not acted on here):**
1. `HCPCS_CD` dominates by a wide margin (0.377 of total gain), consistent with it being the
   single most Chow-significant shared variable and the baseline's largest interaction block.
2. Several raw provider-identifier columns rank highly: `ORG_NPI_NUM`, `TAX_NUM`,
   `CARR_CLM_BLG_NPI_NUM`, `PRF_PHYSN_UPIN`, `PRVDR_NUM`. Plausibly a legitimate "this provider is
   denied more often" signal -- the same kind of thing the baseline's `prvdr_num_freq` already
   captured, just pre-aggregated into one number there instead of split on directly here. But raw
   ID splitting can also memorize individual providers' small-sample noise rather than a
   generalizable pattern, and that's not distinguishable from gain-based importance alone. Left as
   an open question for the SHAP pass (task queue item 4), not something to act on now.

**Checked, and NOT new leakage:** `LINE_PRMRY_ALOWD_CHRG_AMT` ranks #2 by gain (0.199). Verified
it's an exact DME-only duplicate of `LINE_ALOWD_CHRG_AMT` (already documented in
`build_chow_design_matrix.py`'s fix B, and already part of the baseline's pooled features) --
confirmed numerically (DME: denial rate 1.3% when >0, 20.0% when both fields are 0, identical for
both columns). So this isn't new information the baseline didn't already have access to, just a
raw duplicate the tree happened to weight instead of its twin -- `build_chow_design_matrix.py`'s
identity-removal (fix B) is linear-model-only and correctly not applied to the XGBoost feature set
by design (see `fit_xgboost.py`'s docstring), so both copies are present and equally informative.

### Housekeeping

- Result files: `reports/xgboost_results.txt` (readable report incl. top-20 feature importance),
  `reports/xgboost_fit.json` (hyperparameters, metrics, top-20 importances).
- `--sample-frac 0.02` and `0.2` smoke tests both run before the full run, per standing practice;
  0.2 already beat the baseline overall (PR-AUC 0.5716 vs 0.5574) using only 20% of train.

### Next

SHAP on the XGBoost model -- in particular the provider-identifier question above -- then Phase 3
(Azure ML deploy).

## 2026-09-25 (continued): SHAP on the XGBoost model -- Phase 2 complete

### What I did

1. **Model persistence.** Added model-saving to `fit_xgboost.py`: right after fitting,
   `model.save_model(reports/xgboost_model.json)` (native XGBoost booster format) plus a metadata
   JSON (`reports/xgboost_model_meta.json` -- column order, which columns are categorical, train's
   exact category list per categorical column, best iteration). The booster file alone doesn't
   carry the pandas category-dtype mapping needed to reconstruct feature vectors identically, so
   both are needed together. Re-ran the full fit (no `--sample-frac`) to produce these -- it
   reproduced identical results to the earlier full run (89 trees, same metrics to displayed
   precision), confirming the save addition didn't change fitting behavior.
2. **Synthetic-data validation first** (standing practice for new modeling-adjacent code, before
   touching real data): built a small synthetic XGBoost model with a categorical column, a numeric
   column with injected NaN, and planted signal + label noise, then confirmed
   `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")`'s SHAP values sum to
   (predicted margin - expected_value) to ~1.67e-6 -- i.e. the explainer correctly handles
   XGBoost's native categorical splits and missing-value routing.
3. **`src/fit_shap.py` (new).** Loads the saved model + metadata -- does NOT refit -- then, before
   anything else, re-verifies the loaded model reproduces `reports/xgboost_fit.json`'s already-
   reported val metrics (PR-AUC, ROC-AUC, Brier, mean predicted) to 1e-6. This closes a real risk:
   without it, SHAP could silently explain a subtly different model than the one already written
   up (stale file, category-list mismatch, etc.) and nobody would notice. It passed cleanly.
   Then draws a 20,000-row sample of val, stratified by label (same idea as
   `fit_chow_test.py`'s streaming stratified sampler, but done in memory since val already fits),
   and runs `TreeExplainer` on it. Sanity-checked per run: `shap_values.sum(axis=1) +
   expected_value` reconstructs `model.predict(X, output_margin=True)` to 5.2e-6 max error on the
   full sample -- the pipeline is explaining the right model correctly.
4. Reports mean |SHAP| importance overall and per claim type (never only overall, per this
   project's standing audit rule), and answers the provider-ID question quantitatively: for each
   of the 5 flagged raw ID columns (ORG_NPI_NUM, TAX_NUM, CARR_CLM_BLG_NPI_NUM, PRF_PHYSN_UPIN,
   PRVDR_NUM), correlates each category's mean |SHAP| in the sample against log(its claim count in
   train).
5. Hit one bug along the way: the JSON writer crashed on `recon_err` (a numpy float32 scalar,
   `TypeError: Object of type float32 is not JSON serializable`) -- fixed with an explicit
   `float()` cast, re-ran the smoke test (`--sample-size 2000`) to confirm the fix, then ran the
   full 20,000-row pass.

### Results: SHAP overall and per claim type

Loaded model verified against reported val metrics before explaining (match to 1e-6). SHAP
reconstruction error 5.245e-6 (max, over the full sample).

Top features by mean |SHAP| (overall): HCPCS_CD 0.781, LINE_PLACE_OF_SRVC_CD 0.166,
CARR_CLM_BLG_NPI_NUM 0.118, PRNCPAL_DGNS_CD 0.093, PRVDR_NUM 0.087, LINE_NUM 0.075, ORG_NPI_NUM
0.070, TAX_NUM 0.068, CARR_LINE_PRCNG_LCLTY_CD 0.034, CARR_LINE_CLIA_LAB_NUM 0.022. Full top-20 and
per-claim-type top-10 tables (carrier/outpatient/dme) in `reports/shap_results.txt` /
`reports/shap_fit.json`.

**Finding 1 -- gain and SHAP disagree for one feature.** `LINE_PRMRY_ALOWD_CHRG_AMT` was gain-based
importance's #2 feature (0.199, previous session) but doesn't appear anywhere in SHAP's overall
top 20; it only shows up at #10 within DME specifically (0.028). This is not a leakage retraction
-- already verified last session to be an exact DME-only duplicate of a feature the baseline
already used -- but it is a genuine methodological finding worth documenting: gain sums total
split-quality improvement, which a feature can dominate via a few very effective splits even if
its typical per-prediction contribution is small; SHAP reflects actual average per-prediction
impact and is the more trustworthy "what does the model actually rely on" answer of the two. Going
forward, this project's top-line feature-importance claims should cite SHAP, not gain, when they
disagree -- both are kept in their respective report files for anyone who wants to see the
divergence directly.

**Finding 2 -- the provider-ID question, answered per column (not one verdict).** Correlation of
|mean SHAP| with log(train claim count), per flagged column:

| Column | corr | low-count-half mean\|SHAP\| | high-count-half mean\|SHAP\| | n categories | overall mean\|SHAP\| |
|---|---|---|---|---|---|
| PRVDR_NUM | +0.234 | 0.102 | 0.274 | 2,440 | 0.087 |
| CARR_CLM_BLG_NPI_NUM | +0.065 | 0.161 | 0.159 | 3,152 | 0.118 |
| ORG_NPI_NUM | +0.017 | 0.082 | 0.082 | 4,683 | 0.070 |
| PRF_PHYSN_UPIN | -0.150 | 0.042 | 0.032 | 145 | 0.009 |
| TAX_NUM | -0.186 | 0.139 | 0.090 | 1,314 | 0.068 |

Reading: `PRVDR_NUM` is the clearest case of a genuine, volume-supported signal -- providers with
more claims in train get *more* SHAP weight (0.274 vs 0.102), not less, which is what you'd expect
if the model is picking up a real, statistically-supported "this provider's claims get denied more
often" pattern. It corroborates the baseline model's large `prvdr_num_freq` coefficient -- same
underlying signal, just pre-aggregated there vs. raw-split here. `CARR_CLM_BLG_NPI_NUM` and
`ORG_NPI_NUM` are essentially flat -- no evidence of memorization, but no strong evidence of a
volume-driven signal either; call these unresolved rather than cleared. `TAX_NUM` and
`PRF_PHYSN_UPIN` both show the negative correlation the memorization concern predicted -- lower-
volume categories carrying disproportionately more SHAP weight -- but both are minor overall
contributors (mean |SHAP| 0.068 and 0.009 respectively, well below HCPCS_CD's 0.781 or even
PRVDR_NUM's 0.087), so this is a real but small-magnitude finding: worth naming honestly in the
writeup, not a reason to distrust the model's headline PR-AUC/ROC-AUC numbers, since these two
features aren't carrying much of the model's overall weight to begin with.

**Bottom line on task queue item 4:** the raw provider-ID features are not uniformly one thing.
Recommend keeping all 5 in the model (none is large enough to meaningfully skew results even in
the worst case, and PRVDR_NUM in particular is a genuine asset), but flagging TAX_NUM and
PRF_PHYSN_UPIN's memorization signature explicitly in the final written report/interpretation
deliverable (Phase 5) rather than treating XGBoost's feature list as self-evidently trustworthy.

### Housekeeping

- New files: `src/fit_shap.py`, `reports/shap_results.txt`, `reports/shap_fit.json`. Modified:
  `src/fit_xgboost.py` (model-saving addition; re-run confirmed identical `xgboost_results.txt`/
  `xgboost_fit.json` content to before, so those two files are unchanged).
- `reports/xgboost_model.json` (~21.8 MB) and `reports/xgboost_model_meta.json` (~605 KB) are
  real, needed on disk (`fit_shap.py` requires them), but are gitignored rather than pushed: this
  cloud session's GitHub-push path requires reading a file's exact content into the tool call, and
  a file this size doesn't fit that discipline. Both are fully deterministic
  (`random_state=42`) from `train_model.parquet` -- `python src/fit_xgboost.py` regenerates them
  byte-for-byte identically on any machine with the pinned deps. Not a data-loss risk, same
  reasoning as `data/processed/` already being gitignored for regenerable large files.
- This closes out Phase 2 (baseline + Chow test + XGBoost + SHAP), per the top-of-file phase list.

### Next

Phase 3: Azure ML deploy. No open decision blocking it as of this entry.
