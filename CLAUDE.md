# Claims Denial Risk Prediction — Claude Code instructions

Portfolio project: predict claim-denial risk on CMS Synthetic Medicare Claims (carrier,
outpatient, DME), with an engineered noisy-OR label (`is_denied`, ~12.1%). Phases: 0 setup,
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

## Documentation conventions

- `data/FEATURE_ENGINEERING.md` is the source of truth for every feature decision. Section 6's
  2026-09-23/24 addendum covers the rank-deficiency work. **Append corrections; never edit away
  wrong earlier claims** — the visible correction trail is deliberate (interview prep).
- `data/TARGET_DEFINITION.md` covers the label; `data/data_dictionary.md` is a column reference.
- Read the relevant doc section before re-deriving anything.

## Current state (as of 2026-09-24)

Key scripts in `src/`:
- `build_chow_design_matrix.py` — builds restricted (pooled) and unrestricted (claim-type-
  interacted) design matrices; now removes every rank-deficiency dependency (fixes A–D, see its
  docstring and FEATURE_ENGINEERING Section 6).
- `fit_chow_test.py` — fits both, LR test. `--method {standard,firth}`,
  `--exclude-separating-codes`, `--sample-frac`, `--restricted-only` / `--skip-restricted`.
  Refuses to fit unless both matrices are full rank and name-based df == column-count diff ==
  rank diff. Rejects restricted-fit summaries saved before 2026-09-24 (no `rank` field).
- `check_rank.py`, `explain_dependencies.py` — rank diagnostics; the latter writes each
  dependency as an exact equation.
- `diagnose_separation.py`, `verify_provider_outlier_mechanism.py` — separation diagnostics.

Verified on real data (`--sample-frac 0.2`): all four matrices full rank — restricted 117
cols (111 with exclusion), unrestricted 275 (263 with exclusion). Expected df = 275 − 117 = 158.

Separation (confirmed): 6 HCPCS codes (94010, 96156, 99401, 99408, 99495, M1069) have an exact
0.000 denial rate within carrier, full population. Two fixes are implemented, meant to be compared:
Firth (`--method firth`) vs standard MLE with `--exclude-separating-codes`.

## Task queue (do in order; log each in SESSION_LOG.md)

1. **Account for the restricted matrix's extra dropped column.** Explained dependencies predict 118
   restricted columns; real data gives 117. Print `build_chow_design_matrix()`'s
   `df.attrs["rank_fix_report"]` on a 0.2 sample and diff the dropped lists against
   FEATURE_ENGINEERING Section 6's table. Identify the extra column and whether it was a
   `zero_with_gaps` case (a present 0 conflated with absence by the zero-fill). Document it.
2. **Smoke tests** (`--sample-frac 0.02`): `--method firth`, then `--method standard
   --exclude-separating-codes`. Check convergence, finite distinct log-likelihoods, LR ≥ 0.
3. **Full staged runs** for both methods (restricted-only, then skip-restricted).
4. **Compare** the two: reject/not-reject agreement, LR, p, df. Report exclusion as primary if they
   disagree (Firth's chi-square under true separation is simulation-validated, not proven; see
   `_fit_logit_firth` docstring). Write results into FEATURE_ENGINEERING Section 6 and README.
5. If H0 is rejected: Stage 2 per-variable LR tests (FEATURE_ENGINEERING Section 3).
6. Then Phase 2 continues: baseline metrics (PR-AUC), XGBoost, SHAP (see project Build Guide).

## Open decisions for the user (don't act on these alone)

- **Possible target-construction bug in `provider_outlier`.** The presence audit shows
  `PRVDR_NUM` is 100% null for all carrier claims. If `rule_provider_outlier` in
  `denial_rules.py` groups on `PRVDR_NUM`, pandas' `groupby(dropna=True)` means the factor can
  never fire for any carrier claim (~62% of data). First confirm which column the rule groups on
  (read the function). If confirmed, write it up with options (e.g. group carrier claims on
  `CARR_NUM` or a carrier provider field) and wait for the user; changing it changes `is_denied`
  and invalidates downstream results. Also correct `fit_chow_test.py`'s docstring, which says
  PRVDR_NUM is NaN "for these specific claims" — it's NaN for all carrier claims.
- 3 fixed-constant fields (`CARR_CLM_PMT_DNL_CD`, `CLM_DISP_CD`, `CLM_MDCR_NON_PMT_RSN_CD`):
  move to `build_features.py` `DROP_COLUMNS` now, or wait until XGBoost?
- ~130 raw columns excluded from the baseline pending an encoding decision (see `_PENDING_*` in
  `fit_chow_test.py`); secondary diagnosis/procedure codes are the one group likely worth encoding.
