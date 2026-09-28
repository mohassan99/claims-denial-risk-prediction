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

## 2026-09-25 (Phase 3 session): XGBoost model deployed to Azure ML and verified live

### Results

**The Phase 2 XGBoost model is live** as an Azure ML managed online endpoint:
`claims-denial-xgb` in workspace `mlw-claims-denial` (resource group `rg-claims-denial`,
Canada Central), deployment `blue`, 1 x `Standard_DS2_v2`, key auth. It takes raw claim rows as
JSON (same column names as `*_model.parquet`) and returns P(denied) per row.

**It returns the same predictions as the local model.** `deploy/verify_endpoint.py` scored 3,000
label-stratified val rows over HTTPS and compared each one against the local model:

| | n | PR-AUC local | PR-AUC endpoint |
|---|---|---|---|
| all | 3,000 | 0.5550 | 0.5550 |
| carrier | 1,857 | 0.2042 | 0.2042 |
| outpatient | 954 | 0.7813 | 0.7813 |
| dme | 189 | 0.2644 | 0.2644 |

Max per-row difference 5e-7, which is just the endpoint rounding to 6 decimals. (The sample PR-AUCs
differ from the full-val 0.583 only because it's a 3,000-row sample; the local scoring code
reproduces the full-val 0.5832 / 0.8161 exactly.) Output: `reports/endpoint_verification.txt`.

**The model was rebuilt from scratch first and reproduced Phase 2 byte-for-byte.** The fitted
model files are gitignored and this cloud session started without them. I copied
`combined_claims_raw.parquet` from your machine and re-ran `build_target_and_split.py` →
`build_features.py` → `fit_xgboost.py`. `label_audit.txt`, `xgboost_results.txt` and
`xgboost_fit.json` all came out identical to the committed versions (`git diff` empty): same
14.87% label, same 89 trees, same metrics. The model was deployed from that rebuild.

### Decisions made (and why)

- **Region is `canadacentral`, not the planned `eastus`.** The Azure for Students subscription has
  an "Allowed resource deployment regions" policy: swedencentral, mexicocentral, francecentral,
  denmarkeast, canadacentral. Azure ML isn't offered in mexicocentral, so Canada Central was the
  closest supported region. You confirmed it, along with the `rg-claims-denial` /
  `mlw-claims-denial` names.
- **Instance size `Standard_DS2_v2` x 1** (2 vCPU / 7 GB). The subscription allows 4 vCPUs per VM
  family and 6 per region, and managed endpoints reserve an extra 20% for rolling upgrades. A 2-vCPU
  instance is the largest that fits. Azure warns that DS3_v2 is its recommended minimum, but the
  model is 22 MB and scores 500 rows per request without trouble.
- **Scoring applies exactly the val-time transform.** `deploy/score.py` casts each categorical
  column using train's category list from `xgboost_model_meta.json`. A value never seen in training
  becomes NaN (xgboost's missing branch), never a wrong integer code, which is the same rule as
  `fit_xgboost._apply_categories`. Missing columns → NaN (same as a structurally-absent claim-type
  field). Label, IDs and raw dates are ignored if sent. Each response carries
  `unseen_category_counts`, so a caller can tell when a prediction rests on inputs the model never
  saw. In the 5,000-row val check, 73 values across all rows were val-only categories, as expected.
- **The Azure ML model registry is now the durable copy of the model.** It's registered as
  `claims-denial-xgboost` v1, tagged with a content hash plus val PR-AUC/ROC-AUC.
  `deploy/deploy.sh download-model` pulls it back into `reports/`, so no future session has to
  refit it (refitting needs ~7 GB RAM plus swap). `deploy.sh` only registers a new version when the
  local files' hash changes.
- **Everything Azure-side is one idempotent script** (`deploy/deploy.sh`). Each step checks what
  exists first, so re-running it from a new session or from your laptop never duplicates anything.
  Sub-commands: `status`, `test`, `download-model`, `stop`, `teardown`.

### What went wrong on the way, and what now prevents it

1. **Label rebuild OOM-killed twice.** This sandbox has the same ~8 GB as your laptop, and loading
   `combined_claims_raw.parquet` alone peaks at 5.7 GB. Fixed by adding a 12 GB swap file in the
   sandbox. Nothing in the pipeline changed. Noted in CLAUDE.md for future cloud sessions.
2. **`ImageBuildFailure: Identity(object id: ) does not have permissions for .../environments/read`**
   on every *first* build of a new environment version. The same deployment succeeded when re-run a
   few minutes later with no permission change. My working explanation (a hypothesis, not
   confirmed): the image build itself succeeds in the background, and the deployment's own
   status check fails. That check seems to run under the caller's identity, and a personal Microsoft
   account has no directory object ID here (`az role assignment list --assignee <your email>`
   can't resolve you in the graph, which fits). Workaround: re-run `deploy/deploy.sh`, which is safe
   because it's idempotent. If this becomes a nuisance, the proper fix is to run deployments as a
   service principal.
3. **Stale managed identity.** I deleted and recreated the endpoint without waiting for the delete
   to finish. The new deployment's VM then tried to pull its image with the *old* identity's client
   ID (`UserAssignedIdentityNotFound` / MSI token 404). Fixed by fully deleting and waiting until
   `show` returned not-found before recreating. `deploy.sh` now also waits 120 s after creating an
   endpoint so its AcrPull/Storage role assignments propagate before the deployment starts. (On a
   brand-new workspace the container registry only exists after the first image build, so the very
   first endpoint couldn't get AcrPull at all. That's another reason the first attempt failed.)
4. **Container crashed on start: `sklearn needs to be installed`.** `XGBClassifier` imports
   scikit-learn, which my local venv had and the inference image didn't. It was caught from the
   deployment logs and fixed in `deploy/conda.yml`. To stop this class of bug recurring, `score.py`
   was then re-tested in a **clean venv containing only `conda.yml`'s packages**, and through the
   real Azure ML inference server (`azmlinfsrv`) locally, before redeploying. Lesson: test the
   scoring script in an environment built from the deployment's own dependency list, not the dev
   venv.
5. `az ml online-deployment update` doesn't route traffic, so the deployment that succeeded via
   `update` got 0% traffic ("No valid deployments to route to"). `deploy.sh` now always sets
   `blue=100` after create/update.

### Files

New: `deploy/score.py`, `deploy/conda.yml`, `deploy/endpoint.yml`, `deploy/deployment.yml`,
`deploy/deploy.sh`, `deploy/verify_endpoint.py`, `deploy/sample_request.json` (6 val rows, 2 per
claim type, non-null fields only), `reports/endpoint_verification.txt`. Modified: `.env.example`
(adds `AZURE_TENANT_ID`, `AZURE_LOCATION`, and the chosen names), README (Setup → Azure section,
status line, Phase 3 Progress entry), CLAUDE.md. Your local `.env` isn't in git. This session's
`.env` has the tenant/subscription IDs plus RG, workspace and location.

### Decision for you

- **The endpoint bills while it's up, whether or not anyone calls it.** A DS2_v2 costs roughly
  $0.15/hour (about $100+/month), which is about the size of the Azure for Students credit.
  Recommendation: keep it up while you record the Phase 5 demo or show it to someone, then run
  `deploy/deploy.sh stop`. That deletes only the VM-backed deployment. The endpoint, the registered
  model and the workspace all stay, and `deploy/deploy.sh` brings the deployment back in about
  15 minutes. The workspace itself costs pennies at idle (storage, key vault, registry).

### Next

Phase 4: the GenAI layer, which calls this endpoint and explains a score with SHAP plus the denial
reason codes (`labeled_claims_for_eda.parquet` has `denial_reason_carc_1/2`).

## 2026-09-26: Databricks port (PySpark medallion pipeline, MLflow, Unity Catalog) + a split finding

### Results

**The Phase 1 pipeline now also runs on Databricks, and the Spark version reproduces the pandas one
exactly, from the label onward.** Bronze → silver → gold Delta tables in `workspace.claims_denial`:

| Check (`databricks/pipeline/parity_check.py`) | Result |
|---|---|
| `label_audit` built by Spark vs `reports/label_audit.txt` | byte-identical |
| `gold_labeled_claims_for_eda` vs `labeled_claims_for_eda.parquet` (1,799,924 rows) | every cell equal |
| `gold_{train,val,test}_model` vs `*_model.parquet` | every cell equal, same columns/types, same row order |
| Negative control: 5 planted differences in a copy | all 5 caught (3 cells, 1 label, 1 row-order) |
| Bronze DME vs the raw `dme.csv` text | every cell equal (empty field → null is the only difference) |
| Bronze → silver for all three files vs `combined_claims_raw.parquet` | **pending: carrier.csv not uploaded yet** |

**Model: same model, now tracked in MLflow and registered in Unity Catalog.** `databricks/train_mlflow.py`
fits XGBoost from the gold tables using `src/fit_xgboost.py`'s own functions. The fitted booster and its
metadata are **byte-identical** to `reports/xgboost_model.json` / `xgboost_model_meta.json` (val PR-AUC
0.5832, ROC-AUC 0.8161, 89 trees); the script refuses to log anything if that check fails. Logged to the
MLflow experiment `/Users/mohassan99@gmail.com/claims-denial-risk-prediction`, registered as
`workspace.claims_denial.claims_denial_xgboost` v2. v2 is a pyfunc wrapper around `deploy/score.py`, so
the registered model takes raw claim rows and returns exactly the Azure endpoint's numbers (checked: the
sample request gives the same 6 scores; 2,000 val rows match the local model with max diff 0.0). v1 (plain
xgboost flavor, needs pre-encoded categoricals) is superseded.

**Orchestration.** `databricks/deploy_job.py` publishes the code to the workspace and creates the Job
`claims-denial-medallion-pipeline` (bronze → silver → silver_label → gold → parity, serverless). The
parity task has passed as a real Job run; the full run waits on carrier.csv.

### Decisions made (and why)

- **Databricks Free Edition, not Azure Databricks.** Tried Azure first (your choice). What failed, in
  order: the free 14-day trial tier is refused outright for Azure for Students
  (`TrialSkuNotSupportedForSubscriptionType`); a Premium workspace was created in Canada Central but its
  first cluster failed (VM size not available to the subscription); every standard x86 size Databricks
  supports is `NotAvailableForSubscription` in Canada Central and France Central; Sweden Central offers
  them but with 0 quota; Denmark East has usable VMs but no Azure Databricks; the self-service quota
  increase returned `ResourceNotAvailableForOffer`. The Premium workspace was deleted the same hour (its
  NAT gateway bills even idle); cost was cents. Free Edition: serverless only, non-commercial, us-east-2.
- **Business logic stays in `src/`.** The Spark modules import every rule parameter and feature decision
  (REASON_CATALOG, the HCPCS→dx table, provider-key fields, DROP_COLUMNS, the state crosswalk, ...) and
  only re-express the mechanics. So there is one definition of the label and features, not two.
- **Same random draws, not a re-draw.** `is_denied` is a Bernoulli draw from numpy's PCG64 (seed 42) in
  row order, followed by the 30% second-reason draws on denied multi-factor rows, and a seed-43 payment
  noise draw. Spark can't reproduce that stream, and re-drawing would change thousands of labels. The
  draws are generated by the same numpy calls on the driver (one float per row -- seconds, not memory)
  and joined back by `_row_id`. Stated plainly in `silver_label.py`.
- **pandas quirks reproduced deliberately, and documented where they live:** pandas' default NA strings
  (`"NA"`, `"NULL"`, ...) → null; pandas' dtype inference (numeric iff every value parses, int64 only if
  never null); a stable (bene, date, row) order for the prior-auth lookback; `sum` over all-null = 0;
  exact linear-interpolation percentiles (Spark `percentile`, not the approximate one).
- **Row order is part of parity.** Gold tables carry `_split_row` so a model fit from Delta sees rows in
  the parquet's order (tree histograms can depend on order) -- which is why the booster came out
  byte-identical rather than merely close.

### Finding: the "time-based" split is a day-of-month split (decision needed)

`build_target_and_split.py` sorts by the raw `CLM_FROM_DT` string before cutting at 64%/80%. The dates
look like `28-Sep-2015`, so the sort orders by **day of month first**. Checked on the actual files:
train = days 1-20, val = days 20-25, test = days 25-31, each spanning 2015-2023. Consequences:
- **No leakage**: all lines of a claim share a date, so a claim never straddles splits; the Phase 2
  metrics are valid for what is effectively a random-like split.
- **The documented "time-based split" is false** (code comment, TARGET_DEFINITION.md). An interviewer
  who checks would find it.
- **Fix:** parse the date before sorting (a one-line change in `build_target_and_split.py` and in
  `databricks/pipeline/gold.py`). It changes `train_model.parquet`, so every Phase 2 result, the
  deployed model and the UC model would be rebuilt; expect metrics to drop somewhat (the future is
  harder to predict). **Recommendation: fix it** -- a real temporal holdout is the realistic setup and a
  stronger story. Not done: it changes the shared training data, which is yours to approve.

Smaller note: your local `data/processed/combined_claims_raw.parquet` predates `_row_id` (added to
`load_data.py` on 2026-09-16). Nothing downstream depended on it for modeling, which is why every result
still reproduced; this session added `_row_id` to its own copy (identical to what `load_data.py` does)
so the Databricks tables could be compared row by row.

### Files

New: `databricks/` (`run_pipeline.py`, `train_mlflow.py`, `deploy_job.py`, `requirements.txt`,
`pipeline/{common,bronze_silver_claims,silver_label,gold,parity_check,pyfunc_model}.py`),
`reports/label_audit_databricks.txt`, `reports/databricks_mlflow_run.json`. Modified: README (Databricks
setup + Phase 3b entry), CLAUDE.md (Databricks section, finding, open decision), `.env.example`
(`DATABRICKS_HOST`, `DATABRICKS_TOKEN`).

### Next

1. You: upload `carrier.csv` to the `raw` volume (Catalog → workspace → claims_denial → Volumes → raw →
   Upload). Then I run the full Job and the bronze → silver parity check.
2. You: decide on the split fix.
3. Phase 4 (GenAI layer).

## 2026-09-27: true time-based split (your decision), everything refit; Databricks port complete

### Results

**The split is now time-based.** `build_target_and_split.py` sorts by the parsed claim date (stable
sort, so lines sharing a date keep their order), and `databricks/pipeline/gold.py` does the same.

| split | dates | rows | denial rate |
|---|---|---|---|
| train | 2015-01-08 to 2020-10-29 | 1,151,951 | 14.4% |
| val | 2020-10-29 to 2021-11-23 | 287,988 | 17.8% |
| test | 2021-11-23 to 2023-03-02 | 359,985 | 14.2% |

The label did not change (`label_audit.txt` is byte-identical); only which claims land in which split.

**Everything downstream was refit, val set:**

| | Before (day-of-month split) | After (time-based) |
|---|---|---|
| Chow Stage 1 (Firth LR, 158 df) | 2,642 | 2,527, still rejects pooling |
| Chow Stage 2 selection | 5 interact / 6 pool | same 5 / same 6 |
| Baseline PR-AUC overall / carrier / outpatient / DME | 0.557 / 0.141 / 0.815 / 0.262 | 0.671 / 0.141 / 0.877 / 0.264 |
| XGBoost PR-AUC overall / carrier / outpatient / DME | 0.583 / 0.188 / 0.820 / 0.260 | 0.691 / 0.191 / 0.878 / 0.266 |
| XGBoost ROC-AUC overall | 0.816 | 0.849 |
| XGBoost trees (early stopping) | 89 | 73 |

**I predicted the metrics would drop, and they went up. The prediction was wrong, and here's why.**
The val period (Oct 2020 to Nov 2021) has a different claim mix: outpatient is 35.5% of lines (30.6%
in train), and 24.9% of outpatient lines carry a Medicare-non-payable consultation code (15.1% in
train, 14.5% in test). Those claims are near-deterministic denials, so the easy positives are more
common, which raises overall PR-AUC. Within each claim type, carrier and DME barely moved. The model
isn't better; the val period is easier. That shift is also why val's denial rate is 17.8%. A random
split would have hidden it. Test's mix is back near train's, which matters for the final one-time
test evaluation.

**SHAP, provider-ID question re-checked:** `PRF_PHYSN_UPIN`'s memorization signature is gone
(correlation now +0.196, was -0.150). `TAX_NUM` is still slightly negative (-0.149) and still a minor
contributor. `PRVDR_NUM` stays positive (+0.198). `HCPCS_CD` still dominates (mean |SHAP| 0.870).

**Azure endpoint redeployed:** model v2 registered, deployment `blue` updated, and
`verify_endpoint.py` passed again on 3,000 val rows (max difference 5e-7, identical PR-AUC by claim type).

**Databricks port complete:** with `carrier.csv` in the `raw` volume, the Job ran all five tasks from the
raw CSVs, and every table matched pandas cell for cell, including `silver_claims` (1,799,924 rows)
built straight from the files. UC model v3 (trained from the Delta tables) is byte-identical to the
pandas fit.

### What went wrong on the way, and what now prevents it

1. **Chow Stage 2 refused to reuse its cache**: "fit on different columns or a different label
   vector". That's the label-fingerprint guardrail working as designed, since train's rows changed.
   I deleted the stale cache and refit.
2. **Job task `silver` failed on serverless**: `reduce(DataFrame.unionByName, parts)` reaches for the
   JVM (`_jdf`), which serverless forbids. It had worked through Databricks Connect. Fixed with
   `reduce(lambda a, b: a.unionByName(b), parts)`.
3. **Azure login expired, and the device-code login is now blocked** by Entra security defaults
   (AADSTS530035: sign-in "deemed unsafe", because the code was approved on your computer but used
   from the cloud sandbox). Worse, `deploy.sh` didn't notice: `az group exists` failed inside a
   command substitution (which `set -e` ignores) and the script tried to *create* the resource group.
   The create failed too, so nothing happened. Fixes:
   - `deploy.sh` now checks the login first and stops if it's expired.
   - You created a **service principal** `claims-denial-deployer` with Contributor on `rg-claims-denial`
     only (verified from here: that one role at that one scope, nothing at the subscription level).
     Its client ID and secret are in the gitignored `.env`; `deploy.sh` logs in with it automatically.
     The secret expires about 90 days from today.
4. The sandbox restarted once mid-run (swap disappeared, the MLflow job died). I re-enabled swap and
   reran it; nothing was lost.

### Docs

- `docs/DATABRICKS.md` (new): study guide covering the vocabulary, architecture, design decisions,
  results, how to run it, gotchas, and interview talking points.
- README: em dashes removed; Databricks status and a UI pointer added; appended a 2026-09-27 Progress
  entry.
- `data/data_dictionary.md`: em dashes removed; new section mapping every Delta table to its pandas
  file.
- `data/TARGET_DEFINITION.md`: appended the split-correction addendum; the earlier "confirmed
  correct" note is kept as written, per the append-only rule.
- The Project's handoff doc was updated for Databricks.

### Decision for you

- **Test set.** It has never been touched. The standard practice is one final evaluation on it, once
  modeling is frozen (before the Phase 5 report). Recommend doing it at the start of Phase 5.

### Next

Phase 4: the GenAI layer.

## 2026-09-27 (later): GitHub sync, Azure deployment stopped, demo runbook

- **Pushed** all of today's work to GitHub (split fix + refit, Databricks fixes, docs). The
  `push_files` route re-sends every file's full text, so from now on the cloud session hands you a
  `.patch` file instead (your choice): `git pull origin main`, `git am <file>.patch`,
  `git push origin main`. Written into CLAUDE.md's Git section.
- **Azure deployment stopped** (your decision) to save credit. The first `deploy.sh stop` failed:
  Azure refuses to delete a deployment that still has 100% traffic. Fixed: `stop` now sets traffic
  to 0 first. Status after: no deployments, traffic `{}`; endpoint, model v2 and workspace kept.
- **`deploy.sh` no longer tags new model versions with hardcoded (stale) metrics.** v2 had been
  registered with the old "89 trees, PR-AUC 0.583" description; the metrics now live only in
  `reports/xgboost_fit.json`.
- **New `docs/DEMO.md`:** start the endpoint, what to show in Azure and Databricks, stop it, and
  what costs money. Includes a warning found while writing it: old local
  `reports/xgboost_model*.json` files (pre-2026-09-27) would be registered and deployed by
  `deploy.sh`, silently restoring the old model. Run `deploy.sh download-model` first (expect hash
  `11f409f72fc4c45e`).
- Databricks Free Edition costs nothing; no action needed there.
