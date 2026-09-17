"""
Probabilistic, multi-reason denial label engine.

Supersedes the pure boolean-OR combination in denial_rules.py (the individual
risk-factor detector functions there -- rule_missing_prior_auth,
rule_dx_procedure_mismatch, rule_provider_outlier, rule_duplicate_claim,
rule_deprecated_code, rule_missing_hcpcs -- are still used here as inputs;
only the combination logic changes).

WHY PROBABILISTIC, NOT DETERMINISTIC (see chat for full discussion, keep the
short version here since it's the answer to an obvious review question):
1. A deterministic function of the same features used for modeling makes the
   ML step meaningless -- the model would just be re-deriving the rule, and
   an unrealistically perfect PR-AUC is itself a leakage red flag.
2. Real payers don't deny 100% of claims that trip a risk factor -- reviewer
   discretion, retroactive justification, and minor-gap tolerance mean it's
   inherently a graded risk, not a hard trigger.

Note this probabilistic treatment applies to WHETHER a claim is denied
(is_denied). WHICH reason gets recorded for a denied claim is deliberately
NOT randomized (see REASON_PRIORITY below, added 2026-09-11) -- that column
is excluded from the Phase 2 feature set entirely, so its only consumers are
Phase 4/5 demo material, where a reproducible "same active factors -> same
recorded reason" property matters more than statistical realism. See
data/TARGET_DEFINITION.md Addendum #3 for the full design discussion.

WHY NO REAL JOINT/CONDITIONAL PROBABILITY TABLE IS USED: searched specifically
for one. Public sources (Experian State of Claims, Kodiak/HFMA, MGMA,
Aptarro) publish *marginal* survey-share stats ("35% of revenue-cycle leaders
cite prior auth as a top driver") -- self-reported frequency-of-mention, not
measured claim-level incidence, and they don't sum to 100% since respondents
cite multiple reasons. True CARC/RARC co-occurrence data lives inside payer
adjudication systems and isn't published, for the same reason no public CMS
PUF has a real denial field. Most per-rule base rates below are therefore
documented, reasoned ASSUMPTIONS anchored loosely to those marginal
benchmarks, not measured facts -- tunable knobs, disclosed as such in the
README/report, not presented as ground truth. `deprecated_code` is one
exception -- see its entry below and data/TARGET_DEFINITION.md Addendum #6.
`missing_hcpcs` is a second exception, for the opposite reason: no published
benchmark exists for it AT ALL (it's not a scenario survey data covers), so
its base_prob is a disclosed structural-reasoning judgment, not even a
scaled-down survey anchor -- see its entry below.

NOTE ON NOISY-OR INDEPENDENCE ASSUMPTION: the noisy-OR combination below
treats all active risk factors as statistically independent given they're
active. This is a disclosed simplification, not a measured fact -- e.g. a
provider outlier is plausibly correlated with dx/procedure mismatches (same
underlying sloppy-billing cause), which independence doesn't capture. See
data/TARGET_DEFINITION.md Addendum #1. `missing_hcpcs` makes this assumption
mostly moot for the claims it fires on, though: by construction (see
denial_rules.py Rule 8's docstring), none of the other four HCPCS-dependent
rules can be simultaneously active on the same claim, so there's no
independence violation to worry about between missing_hcpcs and those four
specifically -- only provider_outlier (which doesn't use HCPCS_CD) can
co-occur with it.

Sources for the anchoring (see full citations in chat / Phase 5 report):
- Kodiak Solutions / HFMA 2024: ~11.8% industry-wide initial denial rate
  (Medicare Advantage ~15.7%, commercial ~13.9%)
- Experian Health 2025 State of Claims: ~35% of revenue-cycle leaders cite
  prior-authorization issues as a top denial driver
- Aptarro / industry compilations: coding/documentation issues implicated in
  up to ~49% of denials in some analyses
- Louisiana Medicaid transparency report: duplicate claims ~31% of all
  denials, second-largest single category (added 2026-09-11, backs
  duplicate_claim below)
- CMS Transmittal 1875 / MLN Matters MM6740 (effective 2010-01-01): Medicare
  Part B no longer recognizes CPT consultation codes 99241-99245/99251-99255
  for payment (added 2026-09-12, backs deprecated_code below -- this is an
  exact federal policy citation, not a survey estimate)
- CARC 16 ("Claim/service lacks information or has submission/billing
  error(s) which is needed for adjudication") -- standard X12/WPC
  Claim Adjustment Reason Code, backs missing_hcpcs below (added
  2026-09-16). No marginal survey/frequency benchmark exists for this
  specific scenario -- the base_prob is structural reasoning, not a scaled
  survey anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from denial_rules import (
    rule_deprecated_code,
    rule_dx_procedure_mismatch,
    rule_duplicate_claim,
    rule_missing_hcpcs,
    rule_missing_prior_auth,
    rule_provider_outlier,
)

# ---------------------------------------------------------------------------
# Reason catalog: risk factor key -> (CARC code, label, base conditional
# probability of denial GIVEN the risk factor is active, source note)
# ---------------------------------------------------------------------------
REASON_CATALOG = {
    "missing_hcpcs": {
        "carc_code": "16",
        "label": "Claim/service lacks information (no procedure code submitted)",
        "base_prob": 0.08,
        "source_note": "Added 2026-09-16, discovered via a real-data anomaly during "
        "Phase 2 feature engineering, not planned in advance -- see "
        "data/FEATURE_ENGINEERING.md and denial_rules.py Rule 8's docstring for the full "
        "investigation. No published marginal-survey benchmark exists for this exact "
        "scenario (unlike most other factors here) -- base_prob is a disclosed structural "
        "judgment, not a measured rate.\n\n"
        "REVISED 2026-09-17, and DECOUPLED from CALIBRATION_SCALE (see "
        "UNCALIBRATED_FACTORS below) -- both changes forced by the same real-data finding, "
        "not a preference. The original 0.80 value assumed missing_hcpcs would activate on "
        "a small minority of claims, the way every other factor in this file does. Real "
        "data broke that assumption badly: even after excluding carrier's first/last-line "
        "synthetic-total artifact (denial_rules.py Rule 8), missing_hcpcs still activates "
        "on ~33.9% of ALL claims -- a population share no other factor in this file comes "
        "close to. Scaling a factor that large by the same CALIBRATION_SCALE tuned for "
        "every other (much rarer) factor is mathematically incoherent: at the original "
        "0.80 x 1.4 (clamped to 0.95), this single factor alone pushed overall is_denied "
        "to 46.4%, then 41.6% after the line exclusion -- far outside the 10-15% target "
        "no matter how CALIBRATION_SCALE is retuned, since fixing it for this factor would "
        "break the other five, which were already correctly calibrated on their own. "
        "0.08 is the reasoned value that lets missing_hcpcs contribute a modest, plausible "
        "~2-3 percentage points to the population rate given its ~34% activation share -- "
        "arithmetic: target_addition ~= 12.5% - (other five factors' ~9.6% combined "
        "contribution) ~= 2.9%; own_prob ~= 2.9% / 33.9% ~= 0.086, rounded to 0.08. This is "
        "a real retreat from the original 'fundamental adjudication blocker' framing -- "
        "worth stating plainly rather than hiding the reversal: the factor is real and "
        "correctly modeled as INCREASING risk (fixing the original backwards-from-reality "
        "circularity, still the main point of adding this rule), but its true weight in "
        "this dataset is closer to a moderate risk factor than a near-certain one, once its "
        "actual population share is taken into account. This CARC (16, 'claim/service "
        "lacks information') deliberately overlaps with provider_outlier's CARC below -- "
        "both are legitimate, independent real-world reasons a payer might cite the same "
        "generic code; real EOBs routinely reuse CARC 16 across genuinely distinct root "
        "causes. Carrier's first/last line (confirmed synthetic claim-total artifacts, not "
        "real missing-code claims) are excluded from this rule's detection entirely -- see "
        "rule_missing_hcpcs in denial_rules.py for exactly what's confirmed and what isn't "
        "about that pattern.",
    },
    "deprecated_code": {
        "carc_code": "181",
        "label": "Procedure code was invalid on the date of service",
        "base_prob": 0.85,
        "source_note": "Added 2026-09-12. HIGHEST base_prob of any factor, deliberately: "
        "this is the only rule anchored to an exact, dated federal policy (CMS stopped "
        "recognizing CPT consultation codes 99241-99245/99251-99255 for Medicare Part B "
        "payment effective 2010-01-01 -- Transmittal 1875/MLN Matters MM6740) rather than "
        "a marginal survey estimate. Kept below 0.95 (the global clamp) to avoid a fully "
        "deterministic single-feature relationship, consistent with why is_denied is "
        "probabilistic at all -- see module docstring. Discovered via investigation of "
        "duplicate_claim's near-zero activation rate, not planned in advance -- see "
        "data/TARGET_DEFINITION.md Addendum #6 for the full discovery trail.",
    },
    "duplicate_claim": {
        "carc_code": "18",
        "label": "Exact duplicate claim/service",
        "base_prob": 0.35,
        "source_note": "Added 2026-09-11. Anchored downward from real duplicate-claim "
        "denials' strong grounding (~31% of all denials in a Louisiana Medicaid "
        "transparency report, second-largest single category) to account for this "
        "proxy's false-positive risk from legitimate recurring services (dialysis, "
        "PT, DME rentals) that a coarse code+date+beneficiary+provider match can't "
        "distinguish from a true duplicate. See data/TARGET_DEFINITION.md Addendum #4. "
        "NOTE: on real data this rule's near-zero rate turned out to be driven almost "
        "entirely by deprecated_code claims (HCPCS 99241), not genuine duplicate billing "
        "-- see Addendum #6.",
    },
    "missing_prior_auth": {
        "carc_code": "197",
        "label": "Precertification/authorization absent",
        "base_prob": 0.22,  # illustrative -- see calibration note below
        "source_note": "Anchored to Experian 2025's ~35% top-driver citation share, "
        "scaled down since 'cited as a top driver' != 'always denies'",
    },
    "dx_procedure_mismatch": {
        "carc_code": "11",
        "label": "Diagnosis inconsistent with procedure",
        "base_prob": 0.15,
        "source_note": "Anchored to coding/documentation issue compilations "
        "(implicated in up to ~49% of denials, scaled down for the same reason)",
    },
    "provider_outlier": {
        "carc_code": "16",
        "label": "Claim lacks information / provider billing pattern flagged",
        "base_prob": 0.08,
        "source_note": "No direct published benchmark for this specific proxy -- "
        "set low deliberately since it's the weakest-grounded rule (rough outlier "
        "proxy, not a real audit flag). Shares CARC 16 with missing_hcpcs (added "
        "2026-09-16) -- see that entry's source_note for why the overlap is "
        "intentional, not an oversight.",
    },
}

RISK_FACTOR_FUNCS = {
    "missing_hcpcs": rule_missing_hcpcs,
    "deprecated_code": rule_deprecated_code,
    "duplicate_claim": rule_duplicate_claim,
    "missing_prior_auth": rule_missing_prior_auth,
    "dx_procedure_mismatch": rule_dx_procedure_mismatch,
    "provider_outlier": rule_provider_outlier,
}

# ---------------------------------------------------------------------------
# Factors EXCLUDED from CALIBRATION_SCALE's uniform multiplier. Added
# 2026-09-17 -- see missing_hcpcs's REASON_CATALOG source_note above for the
# full arithmetic behind why this was forced, not a preference. Every other
# factor activates on a small population share (well under 25%), so a single
# shared multiplier tuned against that group works coherently. missing_hcpcs
# activates on ~34% of ALL claims -- large enough that scaling it by the same
# knob as everything else makes CALIBRATION_SCALE unsolvable: any value that
# keeps missing_hcpcs's contribution sane would undershoot the other five
# factors, which were already correctly calibrated on their own before this
# factor existed. missing_hcpcs's base_prob (0.08) is therefore fixed
# directly in REASON_CATALOG and used as-is, bypassing calibration_scale
# entirely -- add a factor here only if it shows the same
# large-population-share property, not merely because its calibration is
# inconvenient.
UNCALIBRATED_FACTORS = {"missing_hcpcs"}

# Fixed processing-order priority for reason-code assignment when 2+ factors
# are active on the same denied claim -- DETERMINISTIC, not weighted-random
# (changed 2026-09-11; see data/TARGET_DEFINITION.md Addendum #3 for the full
# rationale). Ordered by real-world adjudication stage:
#   1. missing_hcpcs -- added 2026-09-16, placed FIRST, ahead of even
#      deprecated_code: a real adjudication system verifies a procedure code
#      is PRESENT before it can meaningfully ask whether that code is
#      deprecated, mismatched with the diagnosis, or duplicated. This isn't
#      just a priority-ordering choice -- it's a precondition. See
#      denial_rules.py Rule 8's docstring for the full reframe: the other
#      four HCPCS-dependent rules were never buggy for failing to fire on a
#      missing code; they each presuppose one exists.
#   2. deprecated_code -- added 2026-09-12: code-validity/recognition is a
#      harder, more upfront system check than even duplicate detection -- a
#      claims system needs a currently-recognized procedure code before it's
#      even meaningful to check whether that code has been billed twice.
#   3. duplicate_claim -- early/front-end system check.
#   4. missing_prior_auth (CARC 197) -- front-end, pre/early-adjudication gate.
#   5. dx_procedure_mismatch (CARC 11) -- mid-adjudication medical-policy edit.
#   6. provider_outlier -- ALWAYS last. It's structurally a retrospective/
#      post-payment audit mechanism, not a same-stage adjudication edit, so by
#      the time it could fire, any real-time reason would already have been
#      recorded. This ordering doesn't affect modeling (denial_reason_carc_1/2
#      is excluded from the feature set) -- it exists purely so Phase 4/5 demo
#      claims are reproducible ("same active factors -> same recorded reason")
#      rather than exhibiting spurious run-to-run attribution randomness.
REASON_PRIORITY = [
    "missing_hcpcs",
    "deprecated_code",
    "duplicate_claim",
    "missing_prior_auth",
    "dx_procedure_mismatch",
    "provider_outlier",
]


def compute_risk_factors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run each risk-factor detector, return one boolean column per factor."""
    out = pd.DataFrame(index=df.index)
    out["missing_hcpcs"] = RISK_FACTOR_FUNCS["missing_hcpcs"](
        df,
        kwargs.get("hcpcs_col", "HCPCS_CD"),
        kwargs.get("source_file_col", "_source_file"),
        kwargs.get("line_num_col", "LINE_NUM"),
    )
    out["deprecated_code"] = RISK_FACTOR_FUNCS["deprecated_code"](
        df,
        kwargs.get("hcpcs_col", "HCPCS_CD"),
        kwargs.get("claim_date_col", "CLM_FROM_DT"),
    )
    out["duplicate_claim"] = RISK_FACTOR_FUNCS["duplicate_claim"](
        df,
        kwargs.get("bene_id_col", "BENE_ID"),
        kwargs.get("hcpcs_col", "HCPCS_CD"),
        kwargs.get("claim_date_col", "CLM_FROM_DT"),
        kwargs.get("provider_col", "PRVDR_NUM"),
        kwargs.get("claim_id_col", "CLM_ID"),
    )
    out["missing_prior_auth"] = RISK_FACTOR_FUNCS["missing_prior_auth"](
        df,
        kwargs.get("bene_id_col", "BENE_ID"),
        kwargs.get("hcpcs_col", "HCPCS_CD"),
        kwargs.get("claim_date_col", "CLM_FROM_DT"),
    )
    out["dx_procedure_mismatch"] = RISK_FACTOR_FUNCS["dx_procedure_mismatch"](
        df,
        kwargs.get("hcpcs_col", "HCPCS_CD"),
        kwargs.get("dx_cols", ("PRNCPAL_DGNS_CD",)),
    )
    out["provider_outlier"] = RISK_FACTOR_FUNCS["provider_outlier"](
        df,
        kwargs.get("provider_col", "PRVDR_NUM"),
        kwargs.get("specialty_col", None),
        kwargs.get("payment_col", "CLM_PMT_AMT"),
    )
    return out


def noisy_or_probability(risk_factors: pd.DataFrame, base_probs: dict[str, float]) -> pd.Series:
    """P(denied) = 1 - product(1 - p_i) over ACTIVE risk factors only."""
    survival = pd.Series(1.0, index=risk_factors.index)
    for factor, active in risk_factors.items():
        p = base_probs[factor]
        survival = survival * np.where(active, 1 - p, 1.0)
    return 1 - survival


def sample_denials(
    df: pd.DataFrame,
    seed: int = 42,
    calibration_scale: float = 1.0,
    **detector_kwargs,
) -> pd.DataFrame:
    """Full pipeline: detect risk factors -> noisy-OR probability -> Bernoulli
    sample is_denied -> assign 1+ reason(s) to each denied claim by fixed
    processing-order priority (REASON_PRIORITY), not weighted-random.

    `calibration_scale` multiplies every factor's base_prob EXCEPT those
    listed in UNCALIBRATED_FACTORS (currently just missing_hcpcs) -- the one
    knob to turn if the resulting population denial rate lands outside the
    documented 10-15% industry range (see calibration workflow below). Note
    this scales each factor's pᵢ BEFORE noisy-OR combination, not the combined
    p_denied afterward -- see data/TARGET_DEFINITION.md Addendum #2 for why
    that's not the same thing and why there's no closed-form guarantee this
    lands the population rate exactly in 10-15%; check calibration_report().

    UPDATED 2026-09-17: missing_hcpcs is now excluded from calibration_scale's
    multiplier entirely (see UNCALIBRATED_FACTORS above and its
    REASON_CATALOG source_note for the full arithmetic) -- its base_prob
    (0.08) is fixed and used as-is regardless of calibration_scale's value.
    This was forced by missing_hcpcs's ~34% population activation share,
    which made it mathematically impossible to find one calibration_scale
    value that works for both this factor and the other five (much rarer)
    ones simultaneously. CALIBRATION_SCALE itself (1.4, in
    build_target_and_split.py) was already correctly tuned for those other
    five factors and should not need to change because of this fix -- but
    still verify with calibration_report() after any real-data run, per this
    project's standing practice of checking rather than assuming.
    """
    rng = np.random.default_rng(seed)

    risk_factors = compute_risk_factors(df, **detector_kwargs)
    base_probs = {
        k: (v["base_prob"] if k in UNCALIBRATED_FACTORS else v["base_prob"] * calibration_scale)
        for k, v in REASON_CATALOG.items()
    }
    base_probs = {k: min(p, 0.95) for k, p in base_probs.items()}  # keep sane bounds

    p_denied = noisy_or_probability(risk_factors, base_probs)
    is_denied = (rng.random(len(df)) < p_denied.to_numpy()).astype(int)

    # Reason assignment for denied claims only: DETERMINISTIC fixed-priority
    # order (REASON_PRIORITY), not weighted-random sampling. Changed
    # 2026-09-11 -- this column is dropped from the Phase 2 feature set, so
    # the selection mechanism has zero effect on modeling; its only real
    # consumers are Phase 4 (SHAP narrative, RAG corpus, grounding check) and
    # Phase 5 (report language), where randomizing which active factor gets
    # blamed on different runs hurts demo reproducibility without adding any
    # real-world realism (real adjudication systems generally use a fixed
    # rule-priority hierarchy, not a probability-weighted lottery, to decide
    # which edit gets recorded when several fire). ~30% of multi-factor
    # denials still get a second reason recorded, matching how real EOBs
    # sometimes carry 2 CARC lines -- documented assumption, not a measured
    # rate; this piece of randomness is kept because it reflects real
    # population variety (some denials have one contributing factor, some
    # have several), not "which one is blamed."
    primary_reason = pd.Series(pd.NA, index=df.index, dtype="object")
    secondary_reason = pd.Series(pd.NA, index=df.index, dtype="object")

    denied_idx = df.index[is_denied == 1]
    for idx in denied_idx:
        active = [f for f in REASON_PRIORITY if f in risk_factors.columns and risk_factors.loc[idx, f]]
        if not active:
            # Denied with no active factor is impossible under noisy-OR by
            # construction (p_denied would be 0), so this branch shouldn't
            # fire -- kept as a defensive fallback only.
            continue
        primary_reason.loc[idx] = REASON_CATALOG[active[0]]["carc_code"]
        if len(active) > 1 and rng.random() < 0.30:
            secondary_reason.loc[idx] = REASON_CATALOG[active[1]]["carc_code"]

    result = risk_factors.add_prefix("risk_")
    result["p_denied_model"] = p_denied
    result["is_denied"] = is_denied
    result["denial_reason_carc_1"] = primary_reason
    result["denial_reason_carc_2"] = secondary_reason
    return result


def apply_payment_consequence(
    df: pd.DataFrame,
    result: pd.DataFrame,
    payment_col: str = "CLM_PMT_AMT",
    seed: int = 43,
) -> pd.DataFrame:
    """Make CLM_PMT_AMT internally consistent with is_denied: denied claims
    get zeroed out (small residual noise for coinsurance-only edge cases),
    same as a real $0-pay denied line would look.

    IMPORTANT: this makes CLM_PMT_AMT a near-deterministic CONSEQUENCE of the
    label, not an independent feature. It must be dropped from the Phase 2
    feature set -- see data/data_dictionary.md. This function exists purely
    for internal consistency (so EDA/sanity checks on payment amounts don't
    look broken), not to create a usable feature.

    This is also why CLM_PMT_AMT's NATIVE, pre-this-function value is what
    run_eda.py's validate_native_payment_vs_risk_factors() checks against the
    risk factors -- by the time this function has run, CLM_PMT_AMT no longer
    reflects anything but the label itself. See
    data/TARGET_DEFINITION.md Addendum #5.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    denied_mask = result["is_denied"] == 1
    noise = rng.uniform(0, 2.00, size=denied_mask.sum())  # small coinsurance-like residual
    out.loc[denied_mask, payment_col] = noise
    return out


def calibration_report(result: pd.DataFrame) -> None:
    """Print the numbers you need before trusting a run: overall rate,
    per-factor activation rate, and multi-reason share."""
    rate = result["is_denied"].mean()
    print(f"Overall is_denied rate: {rate:.1%}  (target: 10-15%, hard bound 5-20%)")

    risk_cols = [c for c in result.columns if c.startswith("risk_")]
    print("\nRisk factor activation rates (all claims):")
    print(result[risk_cols].mean().sort_values(ascending=False))

    denied = result[result["is_denied"] == 1]
    if len(denied):
        multi = denied["denial_reason_carc_2"].notna().mean()
        print(f"\nOf denied claims, {multi:.1%} carry a second reason code")
        print("\nPrimary reason distribution (of denied claims):")
        print(denied["denial_reason_carc_1"].value_counts(normalize=True))
