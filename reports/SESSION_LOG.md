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
