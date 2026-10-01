# Phase 5: evaluation for decisions, report and video

Phase 5 turns the model into a decision: which claim lines a scarce reviewer should look at, and what that is
worth. This file covers step 1, the one-time test evaluation. Figures, the report, the video and the Tableau
companion follow.

## Terms

- **p**: the model's predicted probability that a line is denied.
- **Calibrated**: scores match outcomes (of lines scored about 0.30, about 30% are denied).
- **ECE (expected calibration error)**: count-weighted average gap between mean score and observed rate over 10
  equal-width score bins. 0 is perfect.
- **Brier score**: mean of (p - y)^2, y = 1 if denied. Lower is better.
- **PR-AUC**: area under the precision-recall curve; its floor is the denial rate, not 0.5.
- **precision@k / lift@k**: share denied among the top k lines; lift = that share divided by the denial rate.
- **r, L, A**: review cost, loss if a denial is missed, line payment in dollars (see below).

## Cost assumptions (all hypothetical)

```
r   = w * b * m / 60  = 37.51 * 1.3 * 20 / 60 = $16.25 per review
L_i = A_i * (1 - rho) + c * rho * A_i
flag line i when p_i * L_i > r      (equivalently p_i > r / L_i)
```

- w = $37.51/h: median wage of claims adjusters, examiners and investigators, May 2025 (BLS Occupational Outlook
  Handbook). b = 1.3 benefits load (BLS ECEC puts benefits at 30.0% of total compensation, June 2026, which is
  1.43 times wages; 1.3 is the conservative end and 1.43 is in the sweep). m = 20 minutes per desk review: an
  assumption, no source found.
- rho, the recovery rate after an overpayment: reported at both 0.24 and 0.55. For overpayments OIG audits identified
  (Oct 2014 to Dec 2016), CMS reported collecting 55% and OIG verified 24% (HHS OIG A-04-18-03085).
- c = 0.125 recovery cost per dollar recovered: top of the 9.0% to 12.5% RAC contingency fees in 2008 (CRS R40592).
- A_i is the native Synthea line payment: `LINE_NCH_PMT_AMT` (carrier, DME), `REV_CNTR_PMT_AMT_AMT` (outpatient).
  Checked on val: 100% identical to `combined_claims_raw.parquet` by `_row_id`; the label build never touches them.
- Sweep: b in {1.3, 1.43}, m in {10, 20, 30}, c in {0.09, 0.125, 0.20}, rho in {0.24, 0.55} (36 settings), saved in
  the results JSON for every claim type.

The reasoning and sources are also in the Claude Docs note "Phase 5 Cost Assumptions".

How this relates to `docs/COST_SENSITIVE_THRESHOLDS.md` (written earlier the same day): that guide derives
t* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP)) and suggests sweeping t on validation. With C_TP = C_FP = r,
C_TN = 0 and C_FN = L_i, t* reduces to r / L_i, a different cutoff for every line because L_i scales with its dollars.
Because validation showed the scores are calibrated, the cutoff follows from the costs and no sweep over t was needed
to set it. The decision curve over a single t is still reported.

## Protocol: decide on validation, report test once

`src/evaluate_test.py` runs in two stages.

1. `--stage val`: checks the model hash (`11f409f72fc4c45e`), reproduces the recorded val PR-AUC, ROC-AUC and Brier
   overall and per claim type to 1e-6, then applies a rule written before looking at val calibration: recalibrate a
   claim type with isotonic regression only if a version fit on the first half of val (by date) cuts the second
   half's Brier score by more than 1% relative. Gains were 0.37% (carrier), 0.23% (outpatient), -0.11% (DME), so raw
   scores are kept everywhere. Writes `reports/phase5/protocol.json` and `val_results.json`. Committed before test.
2. `--stage test`: reads the protocol, changes nothing in it, scores test once, writes
   `reports/phase5/test_results.json` with the time used (2026-10-01T02:43:10Z), model hash and test data hash
   (`aac1a228d33d0cfa`). It refuses to run again while that file exists.

The cost rule needs no threshold search: the cutoff for each line follows from the costs, provided the scores are
calibrated, which validation confirmed.

## Results on test (2021-11-23 to 2023-03-02, 359,985 lines)

| Group | Lines | Denial rate | PR-AUC (val) | ROC-AUC (val) | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 359,985 | 14.2% | 0.570 (0.691) | 0.813 (0.849) | 0.083 | 0.007 |
| Carrier | 219,887 | 9.9% | 0.189 (0.191) | 0.713 (0.710) | 0.085 | 0.010 |
| Outpatient | 120,547 | 21.5% | 0.807 (0.878) | 0.898 (0.923) | 0.073 | 0.018 |
| DME | 19,551 | 16.2% | 0.259 (0.266) | 0.710 (0.716) | 0.125 | 0.012 |

**Why outpatient fell, checked.** The share of outpatient lines carrying the deprecated consultation code (the
near-deterministic `deprecated_code` rule) fell from 24.9% in val to 14.5% in test. On outpatient lines without it,
PR-AUC is flat: 0.200 (val) and 0.206 (test), ROC-AUC 0.718 and 0.730. The model did not get worse; the easy cases
became rarer. It also shows how much of the headline skill is that one rule: outside it, PR-AUC is 1.6 to 2.3 times
the denial rate (carrier 0.189 vs 9.9%, outpatient without the code 0.206 vs 9.0%, DME 0.259 vs 16.2%).

**Ranking for scarce reviewers (overall, rho = 0.24).**

| Top k of lines | By score: precision, lift, share of denied dollars | By value p*L - r: precision, share of denied dollars |
| --- | --- | --- |
| 1% | 0.95, 6.7x, 23% | 0.80, 80% ($31.7M of $39.9M) |
| 5% | 0.93, 6.6x, 84% | 0.71, 89% |
| 20% | 0.40, 2.8x, 85% | 0.29, 97% |

Ranking by score finds the most denials; ranking by expected value finds the most dollars. Which to use depends
on whether the reviewer's goal is error counts or dollars.

**Cost rule p*L > r, no capacity limit.** Net savings versus reviewing nothing: $29.6M (rho 0.24) and $19.0M
(rho 0.55), 95% and 94% of what a perfect model would net. By claim type:

- Outpatient: the rule flags 60% of lines and nets $28.9M, but reviewing every outpatient line would net $28.2M.
  Line payments are large, so almost any outpatient denial is worth a $16.25 review; the model adds little here
  unless review capacity is limited, which in practice it is (60% of 120,547 lines is about 24,000 reviewer hours).
- Carrier: the rule flags 5.8% and nets $0.73M; reviewing everything would lose $2.2M. This is where the model's
  selection pays for itself.
- DME: Synthea's DME line payments are tiny (median $0, max about $110 in val), so almost no line is worth a review.
  This is a property of the synthetic data, not of real DME (CPAP rentals cost more); stated as a limit.

## Limits

- The label is synthetic (noisy-OR from assumed probabilities, then a random draw), and so are the dollars. These
  results show the method and the discipline, not real payer behavior or real savings.
- One row is a claim line. A review is usually per claim, so line counts overstate the number of reviews.
- Costs are hypothetical; the sweep shows how much the conclusions move.
- Test data were rebuilt in the cloud session from `combined_claims_raw.parquet` (with `_row_id` added in the same
  order `load_data.py` uses); `reports/label_audit.txt` came out byte-identical and the split rates matched (14.4%,
  17.8%, 14.2%). The user's local `val_model.parquet` / `test_model.parquet` predate the 2026-09-27 split fix and
  must be rebuilt (`python src/load_data.py`, `python src/build_target_and_split.py`, `python src/build_features.py`).

**Correction (2026-09-30, same day).** The line above named only the local val and test files. The local
`train_model.parquet` is stale too, and older: built 2026-09-22, it predates both the 2026-09-24 label fix (12.1% denied
instead of 14.4%) and the split fix (its dates run 2015 to 2023, days 1 to 20 of each month). All three local split files
need the rebuild. Nothing in this evaluation used them: the model came from the registry and val/test were rebuilt.

## Figures (step 2)

`python scripts/make_phase5_figures.py` writes six PNGs to `reports/phase5/figures/`. It reads only saved results
(`test_results.json`, `test_predictions.parquet`, `reports/shap_fit.json`, the Phase 4 run summaries) and never scores the
test set. Before drawing, it recomputes every plotted number it can (precision and dollar share at each k, net savings per
setting, review-all net) and fails if any differs from the saved JSON; the passes are in `figure_checks.txt`.

| File | What it shows | What it says |
| --- | --- | --- |
| `fig1_ranking_precision_and_dollars.png` | Precision and share of denied dollars covered as more lines are reviewed, ranked by score and by value p*L - r (rho 0.24) | Score ranking finds denials (about 95% precision in the top 5%); value ranking finds dollars (80% of denied dollars in the top 1%, 97% in the top 20% versus 85% by score) |
| `fig2_calibration.png` | Reliability curve per claim type with ECE, and how many lines fall in each score bin | 87% of lines score below 0.3, where the curve hugs the diagonal. Only 4.9% of lines score 0.6 or higher, and those bins run above the diagonal (0.66 scored, 0.86 denied), so the model under-states risk there. ECE stays 0.007 overall because those bins are small |
| `fig3_cost_vs_cutoff.png` | Net savings versus one cutoff t for every line, for review time m = 10, 20, 30 minutes (r = $8.13, $16.25, $24.38), at rho 0.24 and 0.55 | The per-line rule p*L > r beats the best single cutoff chosen with hindsight on test by $1.4M to $2.9M in all six settings (for example $29.6M versus $26.8M at m = 20, rho 0.24). The ratio of review cost to loss moves the optimum, which is why one cutoff cannot be right for every line |
| `fig4_decision_curve.png` | Net benefit (TP/n - FP/n * t/(1-t)) versus threshold, overall and by claim type | The model beats review-all and review-none at every threshold overall and in outpatient. In carrier and DME it adds value only below about t = 0.25 and 0.3; above that net benefit is about zero (DME slightly negative) |
| `fig5_shap_importance.png` | Mean absolute SHAP, top 15 features | HCPCS procedure code is 0.87, next is 0.15. SHAP comes from the 20,000-row validation sample, not test |
| `fig6_grounding_by_failure_type.png` | Ungrounded tokens by kind for each Phase 4 run | The 9 ungrounded tokens in v1 were all numbers the model computed itself; the 3 in the next run were code descriptions from memory; none after prompt v2 |

Notes and limits:

- The dots in figure 3 are the best single cutoff found on test, with hindsight. They are an upper bound for any single
  cutoff and are not a tuned setting. The cost rule itself was frozen on validation.
- Figure 4 reads the saved decision curve values (t = 0.02 to 0.60). Figure 6 reports grounding, which means traceable to a
  tool output, not correct. Policy over-statements were found by reading and are not in the counts.
- Figure 6 uses the five saved run summaries. The 516 of 516 re-run after the CPT additions has no saved run folder of its own, so it is not drawn.
- Palette: first three slots of the validated default (blue, orange, aqua), the set that passes the all-pairs colorblind checks.
