"""
Probabilistic, multi-reason denial label engine.

Supersedes the pure boolean-OR combination in denial_rules.py (the individual
risk-factor detector functions there -- rule_missing_prior_auth,
rule_dx_procedure_mismatch, rule_provider_outlier, rule_duplicate_claim --
are still used here as inputs; only the combination logic changes).

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
PUF has a real denial field. The per-rule base rates below are therefore
documented, reasoned ASSUMPTIONS anchored loosely to those marginal
benchmarks, not measured facts -- tunable knobs, disclosed as such in the
README/report, not presented as ground truth.

NOTE ON NOISY-OR INDEPENDENCE ASSUMPTION: the noisy-OR combination below
treats all active risk factors as statistically independent given they're
active. This is a disclosed simplification, not a measured fact -- e.g. a
provider outlier is plausibly correlated with dx/procedure mismatches (same
underlying sloppy-billing cause), which independence doesn't capture. See
data/TARGET_DEFINITION.md Addendum #1.

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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from denial_rules import (
    rule_dx_procedure_mismatch,
    rule_duplicate_claim,
    rule_missing_prior_auth,
    rule_provider_outlier,
)

# ---------------------------------------------------------------------------
# Reason catalog: risk factor key -> (CARC code, label, base conditional
# probability of denial GIVEN the risk factor is active, source note)
# ---------------------------------------------------------------------------
REASON_CATALOG = {
    "duplicate_claim": {
        "carc_code": "18",
        "label": "Exact duplicate claim/service",
        "base_prob": 0.35,
        "source_note": "Added 2026-09-11. Anchored downward from real duplicate-claim "
        "denials' strong grounding (~31% of all denials in a Louisiana Medicaid "
        "transparency report, second-largest single category) to account for this "
        "proxy's false-positive risk from legitimate recurring services (dialysis, "
        "PT, DME rentals) that a coarse code+date+beneficiary+provider match can't "
        "distinguish from a true duplicate. See data/TARGET_DEFINITION.md Addendum #4.",
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
        "proxy, not a real audit flag)",
    },
}

RISK_FACTOR_FUNCS = {
    "duplicate_claim": rule_duplicate_claim,
    "missing_prior_auth": rule_missing_prior_auth,
    "dx_procedure_mismatch": rule_dx_procedure_mismatch,
    "provider_outlier": rule_provider_outlier,
}

# Fixed processing-order priority for reason-code assignment when 2+ factors
# are active on the same denied claim -- DETERMINISTIC, not weighted-random
# (changed 2026-09-11; see data/TARGET_DEFINITION.md Addendum #3 for the full
# rationale). Ordered by real-world adjudication stage: duplicate-claim
# detection and the prior-auth gate are early/front-end checks, dx/procedure
# mismatch is a mid-adjudication medical-policy edit, and provider_outlier is
# ALWAYS last -- it's structurally a retrospective/post-payment audit
# mechanism, not a same-stage adjudication edit, so by the time it could fire,
# any real-time reason would already have been recorded. This ordering
# doesn't affect modeling (denial_reason_carc_1/2 is excluded from the
# feature set) -- it exists purely so Phase 4/5 demo claims are reproducible
# ("same active factors -> same recorded reason") rather than exhibiting
# spurious run-to-run attribution randomness.
REASON_PRIORITY = [
    "duplicate_claim",
    "missing_prior_auth",
    "dx_procedure_mismatch",
    "provider_outlier",
]


def compute_risk_factors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run each risk-factor detector, return one boolean column per factor."""
    out = pd.DataFrame(index=df.index)
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

    `calibration_scale` multiplies all base_prob values uniformly -- the one
    knob to turn if the resulting population denial rate lands outside the
    documented 10-15% industry range (see calibration workflow below). Note
    this scales each factor's pᵢ BEFORE noisy-OR combination, not the combined
    p_denied afterward -- see data/TARGET_DEFINITION.md Addendum #2 for why
    that's not the same thing and why there's no closed-form guarantee this
    lands the population rate exactly in 10-15%; check calibration_report().
    """
    rng = np.random.default_rng(seed)

    risk_factors = compute_risk_factors(df, **detector_kwargs)
    base_probs = {k: v["base_prob"] * calibration_scale for k, v in REASON_CATALOG.items()}
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
