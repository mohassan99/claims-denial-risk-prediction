# Cost-sensitive thresholds: how the cost of a false positive and a false negative sets the cutoff

Written 2026-09-30 as a study guide and as the design basis for the Phase 5 evaluation. Nothing in this document has been
computed on the test set. Every dollar figure below is a hypothetical used to show the arithmetic, not a measured cost.

## Why this matters for the project

The model gives each claim a probability p of denial. A payer's claims quality team does not act on a probability. It acts
on a list: which claims get reviewed, in what order, with how many reviewers. Turning p into a list needs a cutoff
(threshold) t, and the right t depends on what each kind of mistake costs. A model can have a good PR-AUC and still be
operated at a bad threshold, so the threshold is a business decision as much as a modeling one.

## Terms

For each claim, flag it if p is at least t. Every claim then falls into one of four cells:

| | Actually denied | Actually not denied |
|---|---|---|
| Flagged | TP (true positive) | FP (false positive) |
| Not flagged | FN (false negative) | TN (true negative) |

Let P = TP + FN (all truly denied claims), N = FP + TN (all truly not-denied claims), n = P + N, and prevalence
pi = P / n. In this project pi is about 0.14 to 0.18 depending on the split (train 14.4%, val 17.8%, test 14.2%).

| Metric | Formula | Question it answers |
|---|---|---|
| Recall (sensitivity, true positive rate) | TP / (TP + FN) | Of the claims that were denied, what share did we catch? |
| Precision (positive predictive value) | TP / (TP + FP) | Of the claims we flagged, what share were denied? |
| Specificity | TN / (TN + FP) | Of the clean claims, what share did we leave alone? |
| False positive rate | FP / (FP + TN) = 1 - specificity | How much do we bother clean claims? |
| F1 | 2 x precision x recall / (precision + recall) | One balanced number; it assumes FN and FP cost the same |
| Lift | precision / pi | How much better than picking claims at random? |
| Precision@k | precision among the k highest scores | What if we can only review k claims? |

Accuracy is not useful here. With about 15% denied, flagging nothing is about 85% accurate.

## The cost model

Let C_FP be the cost of a false positive (a wasted review, provider friction), C_FN the cost of a false negative (a missed
overpayment, or a wrongful denial that accrues interest), and C_TP and C_TN the costs of the two correct outcomes. These
are usually 0, or the cost of the review itself for C_TP.

If p is a calibrated probability, then for one claim:

- Expected cost of flagging it = (1 - p) x C_FP + p x C_TP
- Expected cost of not flagging it = p x C_FN + (1 - p) x C_TN

Flag when flagging is cheaper:

    p x (C_FN - C_TP) > (1 - p) x (C_FP - C_TN)

Solving for p gives the cost-optimal threshold:

    t* = (C_FP - C_TN) / [ (C_FP - C_TN) + (C_FN - C_TP) ]

With C_TP = C_TN = 0 this reduces to t* = C_FP / (C_FP + C_FN).

How to read it:

- If a miss costs more than a false alarm (C_FN > C_FP), t* falls below 0.5 and you flag more. Recall rises, precision falls.
- If a false alarm costs more than a miss (C_FP > C_FN), t* rises above 0.5 and you flag fewer. Precision rises, recall falls.
- Equal costs give t* = 0.5, which is the default most software uses. It is right only when the costs really are equal.

Hypothetical example, to show the arithmetic: a review costs $15 (C_FP) and a missed overpayment costs $120 (C_FN), so
t* = 15 / 135 = 0.111. That is below the base rate of about 0.15, so the cost-optimal policy would flag a large share of
claims.

The same formula describes both sides of the lead list. The positive class changes with the lead type. For likely
overpayments the costs are review effort versus unrecovered dollars. For likely wrongful denials the costs are review
effort versus interest, appeal cost and regulatory exposure. Each side gets its own costs and its own t*.

## When capacity binds

In the example above the policy flags more claims than a team can review. Then t* is not the operating point. The team
reviews the top k claims, where k is set by reviewer hours, and the effective threshold is the score of the k-th claim. What
matters is how well the model ranks, so the metrics are precision@k, lift and dollars captured at k.

Dollars should enter the ranking. Rank claim i by

    expected value_i = p_i x amount_i x chance it can be recovered - review cost

and not by p_i alone. A 40% claim worth $5,000 can outrank a 70% claim worth $50.

## How to optimize the cutoff

1. **Check calibration first.** t* is only valid if p is a real probability. Use a reliability curve and the Brier score,
   per claim type. Recalibrate if needed. Prevalence moves between splits in this project, so calibration must be checked
   on the data used to choose t. Verified so far (from `reports/xgboost_results.txt`, validation set): the average predicted probability matches the actual
   denial rate overall and in each claim type (overall 0.1805 vs 0.1779, carrier 0.1006 vs 0.1004, outpatient 0.3168 vs 0.3099,
   DME 0.1596 vs 0.1589). Not yet verified: calibration across the score range, which is what the high-score end of a lead list
   depends on. That needs a reliability curve (bin by score, compare each bin's mean prediction to its actual rate).
2. **Sweep t on the validation set.** At each t compute the confusion matrix and the total cost
   C(t) = C_FP x FP(t) + C_FN x FN(t) (plus review costs). Take the minimizer.
3. **Sweep the cost ratio too.** The costs are assumptions. Show how t* and the chosen operating point move as C_FN / C_FP
   varies. A flat region means the choice is robust to what we do not know.
4. **Draw a decision curve.** Net benefit(t) = TP/n - (FP/n) x t/(1 - t), plotted against t, next to "flag everything" and
   "flag nothing". It is the same cost logic in one picture, and the ratio t/(1 - t) is the odds that encode the cost trade.
5. **Choose t on validation, report once on test.** Never tune on the test set.
6. **Do it per claim type.** Carrier, outpatient and DME have very different base rates and precision (val PR-AUC 0.191,
   0.878, 0.266 for XGBoost), so one global threshold would hide that.

## Which metrics matter most for the lead-list use case

- Precision@k and lift, because reviewer capacity is the binding constraint.
- Dollars captured at k (recall weighted by claim amount).
- Calibration, because both the cost threshold and the expected-value ranking depend on it.
- PR-AUC as the threshold-free summary. ROC-AUC barely moves with prevalence, so it flatters a rare-event problem.
- Expected cost at the chosen operating point, reported with its sensitivity range.

## Audit design that goes with it

If scores decide which paid claims get audited, reviewers only ever see high-scoring claims, and the outcomes they record are
biased toward the model's own beliefs. Two guards: keep a random slice of claims in the sample so the plan's accuracy rate
and the model's calibration can still be estimated without bias, and select the rest with known probabilities that grow with
the score, then reweight each audited claim by one over its selection probability when estimating error rates. The output is
a lead for a human decision, not a decision.

## What this project can and cannot show

The label is synthetic: a noisy-OR probability built from assumed base probabilities, then a Bernoulli draw. A claim where
the model and the label disagree is therefore model error, not a finding about a real payer. With real adjudication data the
same disagreement would be a lead. What Phase 5 can show is the method on the held-out test set: precision@k and lift per
claim type, a calibration check, the cost curve with its sensitivity range, a decision curve, and the explanation attached to
each lead. The report should say plainly that the operational claim needs real remittance data to validate.

## Phase 5 checklist drawn from this guide

- Choose t on the validation set per claim type and per lead type, then evaluate the untouched test set once.
- Figures: precision@k and lift, calibration, cost versus threshold across a range of cost ratios, decision curve.
- State every cost as a labelled hypothetical.
