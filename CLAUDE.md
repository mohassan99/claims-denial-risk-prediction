# Claims Denial Risk Prediction — Claude Code instructions

Portfolio project: predict claim-denial risk on CMS Synthetic Medicare Claims (carrier,
outpatient, DME), with an engineered noisy-OR label (`is_denied`, 14.9% since the 2026-09-24 carrier fix; was 12.1%). Phases: 0 setup,
1 data/EDA (done), **2 baseline logistic + Chow test (in progress)** then XGBoost + SHAP,
3 Azure ML deploy, 4 GenAI layer, 5 report/video, 6 README/portfolio.

## How the user wants to work (read first)

- **You run, read, fix, and re-run yourself.** The user is not watching the terminal. Do not
  stop to ask them to paste output. Work the loop end to end, then report.
- **At the end of every session**, append a plain-language entry to `reports/SESSION_LOG.md`
  (create it if missing): what you ran, what the results mean, what you changed and why, what's
  next, and any decision you need from the user. Write it for someone reading after the fact who
  did not follow the code. Lead with results and decisions, not process.
- **Stop and ask only for real decisions**: anything that changes the target (`is_denied`), the
  shared `train_model.parquet`, the scope of the Chow test, or deletes/rewrites prior work. Put the
  question in `SESSION_LOG.md` and in chat with your recommendation and the tradeoff.
- Never suggest "start over" as a fix. Fix forward.

## Environment (Windows, Git Bash)

- Repo root: `D:\Project_Claims_Denial\claims-denial-risk-prediction`
- Activate venv: `source venv/Scripts/activate` (Windows layout: `Scripts`, not `bin`).
- Terminal Python: use heredocs (`python << 'EOF' ... EOF`), never `python -c "..."` (breaks on
  pasted leading whitespace).
- **Machine has 8 GB RAM and has hit memory ceilings repeatedly.** Full design matrices are ~3 GB
  each. Rules:
  - Test with `--sample-frac 0.02` first, then `0.2`, then full.
  - Full Chow fits run staged: `--restricted-only`, then `--skip-restricted` (same flags both times).
  - Don't run `python src/build_chow_design_matrix.py` directly on the full file (its `__main__`
    holds both matrices at once). Use `fit_chow_test.py` / `check_rank.py` instead.
  - Never hold both design matrices in memory at once; `del` + `gc.collect()` between.
- Dependencies are pinned in `requirements.txt` (includes `statsmodels==0.15.0`,
  `firthmodels>=0.8.2`). `firthlogist` is dead (pulled from PyPI) — do not use it.

## Git

- You commit and push from the local repo now (the old "Claude pushes via GitHub API, user pulls"
  pattern no longer applies). **Always `git pull origin main` before pushing.**
- If working from a cloud workspace that can't push: commit there, deliver a `git bundle` to the
  repo folder, and have the user run `git pull origin main && git pull <bundle> main && git push`.
- Commit incrementally, one verified change per commit, with messages that explain *why*.
- After each phase step, append to README's Progress section (append only, never rewrite it).
- Never commit secrets; `.env` stays gitignored. No coursework references anywhere in the repo.

## Engineering standards (hard-won, don't relearn)

- **`0` is a value, `NaN` is the absence of one.** Never use one to represent the other, for any
  field type. "Present" means non-null; a 0 is a present value.
- Verify every claim against real data before asserting it. This project has repeatedly found its
  own docs wrong by checking (NPI flags, presence counts, df formula, rank). Say "not yet
  verified" when it isn't.
- Build new DataFrame columns into a dict, then one `pd.concat()`. Never assign columns one at a
  time in a loop (fragmentation → memory crashes).
- `.copy()` only when a function mutates its input in place.
- Binary 0/1 columns are `int8`, not `int64`.
- Never build an interaction term that is constant within a claim type (all 0 or all 1).
- `train_model.parquet` is shared with XGBoost: baseline-only exclusions/zero-fills happen in
  `build_chow_design_matrix.py` / `fit_chow_test.py`, never upstream in `build_features.py`,
  unless the user approves.
- Before writing code against a third-party package, verify its actual installed API
  (`inspect.signature`, docstring). Before relying on a package name, confirm it exists on PyPI.
- Test new design-matrix logic on synthetic data with planted cases before running on real data.
- **Audit every rule, feature and check per claim type, never only overall** (added 2026-09-24).
  An overall rate can look right while one claim type is silently exempt: that is how
  `provider_outlier`/`duplicate_claim` never fired on carrier for two weeks. Concretely:
  - before any `groupby` on ID/key columns, confirm each key is populated in every claim type
    (`denial_rules.require_key_coverage`);
  - a new risk factor must get a `denial_reasons.RISK_FACTOR_EXPECTATION` row for every claim
    type ("fires" or "zero: <verified reason>"), or the label build fails;
  - any large cell with an exact 0% or 100% rate (claim type × code, etc.) must be explained in
    writing before moving on (`reports/label_audit.txt` lists them every build).
- A gap between claim types that nobody can explain (e.g. carrier 5.6% vs outpatient 24%) is a
  finding to chase down, not background.
- Any cached fit/intermediate must be keyed to the exact label it was built from (the Chow scripts
  fingerprint the label vector); never reuse a cache across a label rebuild.

## Documentation conventions

- `data/FEATURE_ENGINEERING.md` is the source of truth for every feature decision. Section 6's
  2026-09-23/24 addendum covers the rank-deficiency work. **Append corrections; never edit away
  wrong earlier claims** — the visible correction trail is deliberate (interview prep).
- `data/TARGET_DEFINITION.md` covers the label; `data/data_dictionary.md` is a column reference.
- Read the relevant doc section before re-deriving anything.

## Current state (as of 2026-09-24, end of session)

**Label:** fixed 2026-09-24. `provider_outlier` / `duplicate_claim` now key on the billing NPI for
carrier claims (`denial_rules.build_provider_key`). Rate 14.9% overall (carrier 10.0%, outpatient
24.1%, DME 16.2%), `CALIBRATION_SCALE` still 1.4. `build_target_and_split.py` writes
`reports/label_audit.txt` and fails if any risk factor x claim type cell breaks
`RISK_FACTOR_EXPECTATION`. See TARGET_DEFINITION.md's 2026-09-24 addendum.

Key scripts in `src/`:
- `build_chow_design_matrix.py` — restricted/unrestricted design matrices (rank fixes A–D).
- `chunked_logit.py` — memory-bounded Newton-Raphson logistic (ordinary or Firth), with
  coefficients optionally held at 0 under the full model's penalty. Validated against
  firthmodels and statsmodels. Peak ~4.9 GB at full size.
- `fit_chow_test.py` — Stage 1. Reparameterizes the unrestricted design as [restricted, Z] and
  tests Z = 0 (Firth = Heinze-Schemper penalized LR). Enforces full rank, span equality, df
  agreement (4 ways), convergence, and a label fingerprint on the cached restricted fit.
- `fit_chow_stage2.py` — per-variable penalized LR tests, Holm-adjusted, resumable.
- statsmodels' lbfgs is NOT used for fitting anymore: it never left beta = 0 on these matrices.

**Results on the corrected label (full train set):** Firth LR 2,642.1 on 158 df, H0 rejected.
Standard MLE agrees within 0.1% but is formally non-convergent because of one claim (dx N186 x
carrier). Stage 2: interact provider_state, HCPCS_CD, PRNCPAL_DGNS_CD, prvdr_num_freq,
CARR_CLM_CASH_DDCTBL_APLD_AMT; pool the other 6. `--exclude-separating-codes` is obsolete (no
separation left). Details in FEATURE_ENGINEERING.md Section 6's last addendum.

## Task queue (do in order; log each in SESSION_LOG.md)

1. **Build the Phase 2 baseline mixed model** per Section 3: claim_type dummies + the 5
   interacted variables as interaction terms + the 6 pooled variables as single columns.
   Fit with `chunked_logit` (Firth or standard), train set.
2. **Baseline metrics on val**: PR-AUC (primary, 14.9% positives), ROC-AUC, calibration, and
   per-claim-type breakdown (never only overall).
3. **XGBoost** on `train_model.parquet`, same metrics, per claim type.
4. **SHAP**, then Phase 3.

## Open decisions for the user (don't act on these alone)

- **CALIBRATION_SCALE**: kept at 1.4 (rate 14.9%, inside the 10–15% target) so the fix changed
  carrier labels only. Retuning to ~1.0 (~12.4%) would also move every outpatient/DME label.
- **Minimum support for interaction terms**: dx N186 x carrier has one claim. A rule like "skip
  interaction terms with < N nonzero rows" would make standard MLE converge, but changes the
  Chow test's tested set and df.
- 3 fixed-constant fields (`CARR_CLM_PMT_DNL_CD`, `CLM_DISP_CD`, `CLM_MDCR_NON_PMT_RSN_CD`):
  move to `build_features.py` `DROP_COLUMNS`?
- ~130 raw columns excluded from the baseline pending an encoding decision (see `_PENDING_*` in
  `fit_chow_test.py`); secondary diagnosis/procedure codes are the one group likely worth encoding.
