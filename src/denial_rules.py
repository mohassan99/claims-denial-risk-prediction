"""
Rule-based construction of the `is_denied` target for the CMS Synthetic Claims PUF.

Why this file exists at all (don't delete this docstring -- it's the answer to the
first interview question this project will get): the public CMS synthetic claims
PUF has no real denial/non-payment field -- every candidate field
(CARR_CLM_PMT_DNL_CD, CLM_MDCR_NON_PMT_RSN_CD, CLM_DISP_CD) is a fixed constant
across all 8,671 beneficiaries. See data/TARGET_DEFINITION.md for the full
verification trail. `is_denied` here is therefore an engineered, documented proxy
label, not observed ground truth.

Each rule is a standalone function returning a boolean Series so you can inspect
which rule(s) fired on which claims, and report each rule's individual hit rate,
not just the combined label. That per-rule transparency is what makes this
defensible under interview follow-up.

Column names below follow standard Medicare RIF / CCW naming (CLM_PMT_AMT,
PRNCPAL_DGNS_CD, HCPCS_CD, CLM_FROM_DT, etc.). Adapt to your actual downloaded
column names if CMS has renamed anything since this was written -- print
`df.columns.tolist()` first and diff against the calls below before trusting any
rule's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rule 1 — Zero / near-zero payment on a billed claim
# ---------------------------------------------------------------------------
# SUPERSEDED as a denial CAUSE -- see denial_reasons.py. Using CLM_PMT_AMT to
# *detect* is_denied and then also feeding CLM_PMT_AMT (or anything derived
# from it) into the Phase 2 model is raw-feature target leakage: the model
# would just be recovering the label from a field that near-determines it by
# construction. Kept here only as a POST-HOC CONSISTENCY CHECK -- after
# denial_reasons.py assigns is_denied probabilistically, use this to verify
# denied claims actually got their CLM_PMT_AMT zeroed out downstream (see
# apply_payment_consequence() in denial_reasons.py), never as a model input.
# CLM_PMT_AMT must be excluded from the Phase 2 feature set entirely once
# denial_reasons.py has run -- document this explicitly in the data
# dictionary, don't rely on remembering it.
#
# NOTE (2026-09-11): also verified NOT usable as the target itself, even
# reframed as "predict $0-pay claims" rather than "predict denial" -- $0
# insurer payment also happens for deductible/coinsurance absorption and
# bundling, neither of which is a denial. See data/TARGET_DEFINITION.md
# Addendum #5 for the full reasoning and the native-payment validation check
# in run_eda.py that tests this empirically against this project's own data.
ZERO_PAY_THRESHOLD = 1.00  # dollars


def rule_zero_payment(
    df: pd.DataFrame,
    payment_col: str = "CLM_PMT_AMT",
    billed_col: str | None = None,
    min_billed: float = 25.00,
) -> pd.Series:
    """Flag claims paid ~$0 despite a non-trivial billed amount.

    If `billed_col` isn't available in your extract (some CMS claim files don't
    expose a clean submitted-charge total), this degrades to flagging any claim
    with payment below ZERO_PAY_THRESHOLD -- document that degradation in your
    README if you end up here, since it's a materially weaker signal.
    """
    paid = df[payment_col].fillna(0)
    low_pay = paid < ZERO_PAY_THRESHOLD

    if billed_col is not None and billed_col in df.columns:
        billed = df[billed_col].fillna(0)
        return low_pay & (billed >= min_billed)

    return low_pay


# ---------------------------------------------------------------------------
# Rule 2 — Missing prior-authorization proxy
# ---------------------------------------------------------------------------
# Real prior-auth flags aren't in the public PUF. Proxy: high-cost service
# categories (DME, selected outpatient procedure families) where the
# beneficiary has no prior claim in the lookback window that would plausibly
# represent the authorizing encounter (e.g. a preceding outpatient visit for
# DME, or a preceding diagnosis-supporting encounter for major procedures).
HIGH_COST_HCPCS_PREFIXES = ("E", "K")  # DME HCPCS Level II prefixes, adapt as needed
LOOKBACK_DAYS = 90


def rule_missing_prior_auth(
    df: pd.DataFrame,
    bene_id_col: str = "BENE_ID",
    hcpcs_col: str = "HCPCS_CD",
    claim_date_col: str = "CLM_FROM_DT",
) -> pd.Series:
    """Flag high-cost-category claims with no supporting claim in the prior
    LOOKBACK_DAYS for the same beneficiary.

    This is a coarse proxy, not real auth data -- document it as such. False
    positives are expected for beneficiaries whose supporting encounter fell
    just outside the lookback window or in a claim file this rule doesn't scan.

    NOTE (2026-09-16): like every other rule below that takes hcpcs_col, this
    is STRUCTURALLY unable to flag a claim with missing HCPCS_CD -- a NaN
    cast via .astype(str) becomes "nan", which never starts with "E"/"K". See
    rule_missing_hcpcs below for why that's correct (this rule checks a
    property OF a procedure code; missing_hcpcs checks whether one exists at
    all), not a gap in this rule specifically.
    """
    is_high_cost = df[hcpcs_col].astype(str).str.startswith(HIGH_COST_HCPCS_PREFIXES)

    dates = pd.to_datetime(df[claim_date_col], errors="coerce")
    df_sorted = df.assign(_date=dates).sort_values([bene_id_col, "_date"])

    # Days since beneficiary's previous claim of any kind (any file/category
    # you've concatenated in). NaN (first claim on record) treated as "no
    # supporting history" -> flagged, which is intentional and conservative.
    gap_days = (
        df_sorted.groupby(bene_id_col)["_date"]
        .diff()
        .dt.days
    )
    no_recent_history = gap_days.isna() | (gap_days > LOOKBACK_DAYS)

    # Realign to original index (groupby/sort changed row order)
    no_recent_history = no_recent_history.reindex(df.index)

    return is_high_cost & no_recent_history.fillna(True)


# ---------------------------------------------------------------------------
# Rule 3 — Diagnosis / procedure mismatch (medical-necessity proxy)
# ---------------------------------------------------------------------------
# Coarse category-level check: does the procedure's typical diagnosis category
# (first character of ICD-10-CM, mapped to a broad chapter) match any
# diagnosis billed on the same claim? This is deliberately simple -- a real
# payer's medical-necessity edit is far more granular (specific LCD/NCD policy
# per code, often requiring an exact diagnosis code, not just a chapter
# match). Document the simplification.

# Built 2026-09-12 from this project's ACTUAL top-30 HCPCS codes
# (df["HCPCS_CD"].value_counts() on the real downloaded data), each verified
# against real Medicare coverage/billing documentation -- not guessed. Full
# citations in data/TARGET_DEFINITION.md's "Real HCPCS mapping" addendum.
# Supersedes the original 3-code illustrative placeholder (93000/71046/80053),
# none of which actually appeared in this project's real data at any
# meaningful volume.
#
# Codes NOT in this table are simply never flagged by this rule (see
# docstring below) -- several other high-volume real codes (96156, 99408,
# 99495, 99401, M1069, and others) were deliberately left unmapped rather
# than guessed: either the code's required diagnosis is too broad to reduce
# to a single ICD-10 chapter check (e.g. 99495 transitional care management
# can legitimately follow almost any admitting diagnosis), or the mapping
# wasn't independently verified against a real source. Extend this table
# only when you can cite where a specific code's required diagnosis is
# coming from -- the whole point of this rule losing its 3-illustrative-code
# placeholder status was verification, not just volume.
PROCEDURE_TO_EXPECTED_DX_PREFIX = {
    # Medicare Annual Wellness Visit screening add-ons -- require a
    # screening/encounter Z-code, not a disease-specific code. Verified:
    # Z13.31 (preferred since Oct 2021), Z13.39, Z13.89, Z00.00 all accepted.
    "G0444": ("Z",),  # Annual depression screening
    "G0442": ("Z",),  # Annual alcohol misuse screening

    # Brief emotional/behavioral assessment (e.g. depression inventory) --
    # Z-code if used as a screening tool, F-code (mental/behavioral chapter)
    # if a positive result is what's being coded.
    "96127": ("Z", "F"),

    # Hemodialysis, single physician evaluation -- requires an ESRD/CKD
    # diagnosis (N18.x, ICD-10 genitourinary chapter). Hypertensive CKD
    # (I12.x/I13.x) or diabetic nephropathy (E11.22) are legitimate comorbid
    # alternates/additions per nephrology billing guidance, not substitutes.
    "90935": ("N", "I", "E"),

    # CPAP/PAP therapy supplies -- near-universally require G47.33
    # (obstructive sleep apnea, ICD-10 nervous-system chapter) per CMS LCD
    # L33718 and its companion policy article. Billing with the unspecified
    # sleep-apnea code (G47.30) instead is a documented, common cause of
    # denial specifically for these codes.
    "A7030": ("G",),  # Full face CPAP mask
    "A7031": ("G",),  # Full face mask cushion, replacement
    "A7034": ("G",),  # Nasal CPAP interface
    "A7035": ("G",),  # CPAP headgear
    "A7037": ("G",),  # CPAP tubing
    "A7038": ("G",),  # CPAP disposable filter
    "A4604": ("G",),  # Heated tubing, used with CPAP
    "E0601": ("G",),  # CPAP device (purchase/rental)
}


def rule_dx_procedure_mismatch(
    df: pd.DataFrame,
    hcpcs_col: str = "HCPCS_CD",
    dx_cols: tuple[str, ...] = ("PRNCPAL_DGNS_CD",),
) -> pd.Series:
    """Flag claims where none of the billed diagnosis codes fall in the
    procedure's expected ICD-10-CM chapter range, for the procedures in
    PROCEDURE_TO_EXPECTED_DX_PREFIX.

    Returns False (not flagged) for any procedure code not in the mapping --
    this rule only ever evaluates the codes it has a verified mapping for;
    everything else passes through unflagged rather than being guessed at.

    NOTE (2026-09-16): also structurally unable to flag a claim with missing
    HCPCS_CD (a NaN cast to "nan" never equals any mapped code) -- see
    rule_missing_hcpcs below for why that's the correct division of labor,
    not a gap here.
    """
    hcpcs = df[hcpcs_col].astype(str)
    flagged = pd.Series(False, index=df.index)

    for code, expected_prefixes in PROCEDURE_TO_EXPECTED_DX_PREFIX.items():
        mask = hcpcs == code
        if not mask.any():
            continue

        dx_matches = pd.Series(False, index=df.index)
        for dx_col in dx_cols:
            if dx_col not in df.columns:
                continue
            dx_prefix = df[dx_col].astype(str).str[0]
            dx_matches |= dx_prefix.isin(expected_prefixes)

        flagged |= mask & ~dx_matches

    return flagged


# ---------------------------------------------------------------------------
# Rule 4 — Timely filing violation
# ---------------------------------------------------------------------------
# NOT IMPLEMENTED. FI_CLM_PROC_DT (claim processing date) is documented as
# blank/fixed in this synthetic release, so days-between-service-and-
# submission can't be computed. Left here as a stub so the gap is visible in
# code, not just prose -- and so it's trivial to wire in if you switch to a
# data source that does populate a processing date.
def rule_timely_filing(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "FI_CLM_PROC_DT is blank/fixed in the CMS Synthetic Claims PUF -- "
        "timely-filing logic has no signal to run on in this data source. "
        "Documented in data/TARGET_DEFINITION.md as a disclosed limitation."
    )


# ---------------------------------------------------------------------------
# Rule 5 — Provider outlier billing pattern
# ---------------------------------------------------------------------------
def rule_provider_outlier(
    df: pd.DataFrame,
    provider_col: str = "PRVDR_NUM",
    specialty_col: str | None = None,
    payment_col: str = "CLM_PMT_AMT",
    percentile: float = 0.95,
) -> pd.Series:
    """Flag claims from providers in the top `percentile` of total claim
    volume OR total payment amount, within specialty if specialty_col is
    available, else across all providers.

    Rough proxy for audit-flagged/outlier providers -- a real driver of
    denials and payer reviews. Not a fraud determination.

    NOTE (2026-09-11): structurally this is a RETROSPECTIVE/post-payment audit
    signal, not a same-stage real-time adjudication edit like the other rules
    in this file -- see data/TARGET_DEFINITION.md Addendum #3. That's why it's
    given fixed last priority in denial_reasons.REASON_PRIORITY: by the time a
    retrospective outlier-billing review could flag a claim, any real-time
    adjudication reason would already have been recorded first.
    """
    group_cols = [specialty_col] if specialty_col and specialty_col in df.columns else []

    provider_stats = (
        df.groupby(group_cols + [provider_col])
        .agg(claim_count=(payment_col, "size"), total_paid=(payment_col, "sum"))
        .reset_index()
    )

    def _flag_group(g: pd.DataFrame) -> pd.DataFrame:
        vol_cut = g["claim_count"].quantile(percentile)
        pay_cut = g["total_paid"].quantile(percentile)
        g["_outlier_provider"] = (g["claim_count"] >= vol_cut) | (g["total_paid"] >= pay_cut)
        return g

    if group_cols:
        provider_stats = provider_stats.groupby(group_cols, group_keys=False).apply(_flag_group)
    else:
        provider_stats = _flag_group(provider_stats)

    outlier_providers = set(
        provider_stats.loc[provider_stats["_outlier_provider"], provider_col]
    )
    return df[provider_col].isin(outlier_providers)


# ---------------------------------------------------------------------------
# Rule 6 — Duplicate claim
# ---------------------------------------------------------------------------
# Added 2026-09-11 per design review -- see data/TARGET_DEFINITION.md
# Addendum #4 for full grounding. Duplicate-claim denials (CARC 18) are one of
# the best-grounded denial categories available: a Louisiana Medicaid
# transparency report found duplicates at ~31% of all denials -- the
# second-largest single category, behind invalid procedure/modifier
# combinations and ahead of missing prior authorization. Also structurally
# one of the earliest checks a real adjudication system runs (cheap, and
# usually ahead of medical-policy edits) -- see REASON_PRIORITY in
# denial_reasons.py.
#
# NOTE (2026-09-12): the first real-data run of this rule flagged 982 claims,
# 968 of which were HCPCS code 99241 -- investigated and traced to Rule 7
# below, not a duplicate-billing pattern. See data/TARGET_DEFINITION.md's
# Phase 1 real-data run log for the full investigation.
#
# NOTE (2026-09-16): also structurally unable to flag a claim with missing
# HCPCS_CD -- groupby() drops NaN keys by default, so those rows never join
# any group and .transform("nunique") returns NaN for them, which compares
# False against > 1. See rule_missing_hcpcs below for why that's the correct
# division of labor, not a gap here.
def rule_duplicate_claim(
    df: pd.DataFrame,
    bene_id_col: str = "BENE_ID",
    hcpcs_col: str = "HCPCS_CD",
    claim_date_col: str = "CLM_FROM_DT",
    provider_col: str = "PRVDR_NUM",
    claim_id_col: str = "CLM_ID",
) -> pd.Series:
    """Flag claims that share (beneficiary, procedure, service date, provider)
    with at least one other claim carrying a different CLM_ID.

    KNOWN FALSE-POSITIVE RISK: legitimately recurring services (dialysis,
    physical therapy, some DME rentals) can share code+date+beneficiary+
    provider across multiple real, non-duplicate claims. A real duplicate
    edit typically also checks exact-match units/modifiers, which aren't
    reliably available in this release -- documented simplification, not a
    hidden gap.
    """
    if claim_id_col not in df.columns:
        # Can't tell "same claim re-billed" from "same claim re-read" without
        # a claim identifier -- degrade to all-False rather than silently
        # mis-flagging on the group key alone.
        return pd.Series(False, index=df.index)

    group_cols = [bene_id_col, hcpcs_col, claim_date_col, provider_col]
    dup_counts = df.groupby(group_cols)[claim_id_col].transform("nunique")
    return dup_counts > 1


# ---------------------------------------------------------------------------
# Rule 7 — Deprecated / Medicare-non-payable procedure code
# ---------------------------------------------------------------------------
# Added 2026-09-12. This is the SINGLE BEST-GROUNDED rule in this file --
# every other rule here is anchored to a marginal survey estimate or a coarse
# proxy; this one is anchored to an exact, dated, verifiable federal policy.
#
# Effective January 1, 2010, CMS stopped recognizing CPT consultation codes
# (99241-99245 office/outpatient, 99251-99255 inpatient) for Medicare Part B
# payment (CMS Transmittal 1875 / MLN Matters MM6740, IOM Pub 100-04 Ch. 12
# section 30.6.10) -- physicians were instructed to bill standard E/M codes
# instead. Code 99241 was further deleted from the CPT code set entirely
# effective January 1, 2021 (99251 deleted effective 2023). A Medicare claim
# billed with any of these codes on or after 2010-01-01 is, by CMS's own
# stated policy, not a payable service -- this isn't a coarse proxy for
# denial risk, it's closer to the real mechanism itself.
#
# Origin story worth keeping for the interview: this rule wasn't planned --
# it was DISCOVERED. rule_duplicate_claim's first real-data run flagged 982
# claims, 968 of which were HCPCS 99241. A control comparison against other
# high-volume, single-line-per-claim codes (90935: 222,786 claims, 0% flagged)
# ruled out "just high volume" as the explanation. That result was the cue to
# check whether something about code 99241 itself was unusual -- which led
# directly to this rule. See data/TARGET_DEFINITION.md's Phase 1 real-data
# run log for the full investigation trail.
#
# NOTE (2026-09-16): also structurally unable to flag a claim with missing
# HCPCS_CD (codes.isin(...) is False for the literal string "nan") -- see
# rule_missing_hcpcs below for why that's the correct division of labor, not
# a gap here.
MEDICARE_NONPAYABLE_CONSULT_CODES = {
    "99241", "99242", "99243", "99244", "99245",  # office/outpatient consultations
    "99251", "99252", "99253", "99254", "99255",  # inpatient consultations
}
MEDICARE_CONSULT_NONPAY_EFFECTIVE_DATE = pd.Timestamp("2010-01-01")


def rule_deprecated_code(
    df: pd.DataFrame,
    hcpcs_col: str = "HCPCS_CD",
    claim_date_col: str = "CLM_FROM_DT",
) -> pd.Series:
    """Flag claims billed with a Medicare-non-payable consultation code on or
    after the 2010-01-01 CMS policy effective date.

    Scope note: this project's data (2015-2023 per the beneficiary files)
    falls entirely after the effective date, so in practice this collapses to
    "is the code in MEDICARE_NONPAYABLE_CONSULT_CODES" -- the date check is
    kept explicit anyway so the rule states its actual real-world condition
    rather than an assumption baked silently into which codes are listed.
    """
    codes = df[hcpcs_col].astype(str)
    dates = pd.to_datetime(df[claim_date_col], errors="coerce")
    return codes.isin(MEDICARE_NONPAYABLE_CONSULT_CODES) & (
        dates >= MEDICARE_CONSULT_NONPAY_EFFECTIVE_DATE
    )


# ---------------------------------------------------------------------------
# Rule 8 — Missing procedure code
# ---------------------------------------------------------------------------
# Added 2026-09-16. Discovered via a real-data anomaly during Phase 2
# feature-engineering work, not planned in advance -- see
# data/TARGET_DEFINITION.md's real-data run log for the full investigation.
#
# THE PROBLEM THIS FIXES: every other rule in this file that takes hcpcs_col
# (rule_missing_prior_auth, rule_dx_procedure_mismatch, rule_duplicate_claim,
# rule_deprecated_code) is STRUCTURALLY UNABLE to fire when HCPCS_CD is
# missing -- each one either casts NaN to the literal string "nan" (which
# never matches any trigger condition) or drops NaN group keys entirely
# (rule_duplicate_claim's groupby). Before this rule existed, that meant a
# missing procedure code mechanically forced is_denied toward 0% for those
# claims, independent of anything real about them -- empirically confirmed
# at exactly 0 denials out of 448,567 real HCPCS-missing carrier rows (62.5%
# of all carrier claims in this dataset). That's not a real-world pattern,
# it's a target-construction artifact -- and it's backwards from reality: a
# claim submitted with NO procedure code at all is a textbook denial trigger
# in an actual payer system (CARC 16, "claim/service lacks information"),
# not a safe one.
#
# THE REFRAME, not just the patch: the other four rules' inability to
# evaluate a missing HCPCS was never a bug in THEM -- each checks a property
# OF a procedure code (is it deprecated? does it match the diagnosis? is it
# duplicated?), which presupposes one exists. The actual gap was that
# nothing checked the precondition itself. This rule is that precondition
# check -- which is also why it's placed FIRST in REASON_PRIORITY below: a
# real adjudication system verifies a procedure code is present before it
# can meaningfully ask anything else about it.
#
# CORRECTION 2026-09-16 -- carrier's LINE_NUM=1 excluded, and the reason is
# stated carefully, not overclaimed. Carrier's LINE_NUM="1" rows are 100%
# null on HCPCS_CD across all 58,040 rows checked, zero exceptions -- but
# they are NOT sparse/placeholder rows: diagnosis fields (ICD_DGNS_CD1-12,
# their version-code companions) are populated on line 1 at rates comparable
# to (sometimes higher than) other lines, and LINE_SRVC_CNT/LINE_NCH_PMT_AMT
# both have real, non-null values there too -- just with somewhat different
# distributions (LINE_SRVC_CNT can be 0 on line 1, unlike every other line's
# minimum of 1; LINE_NCH_PMT_AMT has no mass at $0 and a narrower range than
# other lines' heavy-tailed pattern). WHAT WE ACTUALLY KNOW: this is a real,
# populated line that structurally, deterministically lacks a procedure
# code. WHAT WE DON'T KNOW: why it's isolated to line 1 specifically, or why
# those two distributions differ -- no claim is made here about WHAT kind of
# record line 1 "really is." One genuine, cited structural explanation for
# the general phenomenon (a carrier record existing with a diagnosis but no
# procedure code): the CMS User Guide for this dataset (May 2023, Table 3-2,
# "Filter for Non-Exportable Events") documents that Synthea's RIF exporter
# only requires a mappable DIAGNOSIS for Carrier records to be exportable --
# unlike Outpatient/Inpatient, procedure presence is never a Carrier export
# condition. Real Medicare billing rules require every carrier line to carry
# a procedure code (CMS BSA Carrier Line Items PUF documentation), but
# Synthea's own exporter logic doesn't enforce that -- a documented gap
# between the synthetic generator and the real-world rule it approximates.
# DME was checked directly and shows the ordinary ~0% missing rate on both
# line 1 and other lines -- this exclusion is carrier-specific, not a
# general LINE_NUM=1 rule across claim types.
def rule_missing_hcpcs(
    df: pd.DataFrame,
    hcpcs_col: str = "HCPCS_CD",
    source_file_col: str = "_source_file",
    line_num_col: str = "LINE_NUM",
) -> pd.Series:
    """Flag lines with no procedure code at all, EXCLUDING carrier's
    LINE_NUM=1 -- a confirmed, structurally distinct pattern (see the module
    comment above this function for exactly what's confirmed and what isn't).

    base_prob for this rule (see REASON_CATALOG in denial_reasons.py) is a
    REASONED JUDGMENT, not sourced to a specific published benchmark the way
    rule_deprecated_code's federal-policy citation is -- a missing procedure
    code is a fundamental adjudication blocker in real Medicare claims
    processing (professional/DME line items cannot be priced or paid without
    one), so it's set high, but disclosed as design reasoning, not measured
    fact.
    """
    is_missing = df[hcpcs_col].isna()
    is_carrier_line_1 = (df[source_file_col] == "carrier.csv") & (df[line_num_col] == "1")
    return is_missing & ~is_carrier_line_1


# ---------------------------------------------------------------------------
# Provider identity across claim types (added 2026-09-24)
# ---------------------------------------------------------------------------
# BUG FIXED HERE. rule_provider_outlier and rule_duplicate_claim both
# identify "the provider" with PRVDR_NUM, the institutional provider number
# (CCN). That field exists on outpatient and DME claims, but it is 100% null
# on every carrier (professional) claim -- carrier claims carry the provider
# in CARR_CLM_BLG_NPI_NUM, the billing NPI. pandas' groupby drops null keys
# without a word, so for ~62% of all claims neither rule could ever fire:
# carrier's denial rate came out 5.6% vs 24.1% outpatient / 16.3% DME, and
# 34 of 37 carrier HCPCS codes had zero denials. Found 2026-09-24 in Phase 2
# while chasing separation in the Chow test (FEATURE_ENGINEERING.md
# Section 6; TARGET_DEFINITION.md addendum of the same date).
#
# The fix: one explicit, per-claim-type provider key. Verified on the full
# combined file before use: CARR_CLM_BLG_NPI_NUM is 0% null on carrier
# (5,358 distinct billing NPIs; identical to ORG_NPI_NUM on every carrier
# row), and PRVDR_NUM is 0% / 0.03% null on DME / outpatient. Values are
# namespaced ("NPI:" vs "PRV:") so an NPI and a provider number can never
# collide by accident. Outpatient and DME keep exactly their old grouping,
# so their risk flags are unchanged -- checked bit-for-bit after the fix.
PROVIDER_KEY_SOURCE = {
    "carrier.csv": ("CARR_CLM_BLG_NPI_NUM", "NPI:"),
    "outpatient.csv": ("PRVDR_NUM", "PRV:"),
    "dme.csv": ("PRVDR_NUM", "PRV:"),
}

# Which providers are compared with which in rule_provider_outlier's
# top-5% cut. Outpatient and DME stay pooled together exactly as before (so
# their flags don't move); carrier's billing physicians/groups form their
# own pool, since a professional billing NPI and a facility/supplier aren't
# on the same volume or payment scale.
PROVIDER_POOL = {
    "carrier.csv": "professional",
    "outpatient.csv": "institutional_dme",
    "dme.csv": "institutional_dme",
}


def build_provider_key(df: pd.DataFrame, source_file_col: str = "_source_file") -> pd.Series:
    """Namespaced provider identifier per claim type (see block comment
    above). Raises on a claim file with no declared provider field rather
    than silently returning null for it -- the exact failure mode this
    replaces."""
    unknown = set(df[source_file_col].unique()) - set(PROVIDER_KEY_SOURCE)
    if unknown:
        raise ValueError(
            f"No provider-key field declared for claim file(s) {sorted(unknown)} -- "
            "add them to PROVIDER_KEY_SOURCE (after checking which field is populated)."
        )
    key = pd.Series(pd.NA, index=df.index, dtype="object")
    for source, (col, prefix) in PROVIDER_KEY_SOURCE.items():
        mask = (df[source_file_col] == source).to_numpy()
        if mask.any():
            vals = df.loc[mask, col]
            key.loc[mask] = (prefix + vals.astype(str)).where(vals.notna(), pd.NA)
    return key


def build_provider_pool(df: pd.DataFrame, source_file_col: str = "_source_file") -> pd.Series:
    return df[source_file_col].map(PROVIDER_POOL)


def require_key_coverage(
    df: pd.DataFrame,
    rule_name: str,
    key_cols: list[str],
    source_file_col: str = "_source_file",
    max_null_share: float = 0.01,
    allowed_gaps: dict[tuple[str, str], str] | None = None,
) -> None:
    """GUARDRAIL (added 2026-09-24). Before a rule groups on key columns,
    confirm every key is actually populated in every claim type. pandas
    drops rows with a null group key silently, which is how the PRVDR_NUM
    bug hid for two weeks. Raises unless each key's null share is at most
    max_null_share in every claim type, except for gaps explicitly declared
    in allowed_gaps as {(key_col, source_file): "reason"}."""
    allowed_gaps = allowed_gaps or {}
    problems = []
    for col in key_cols:
        shares = df[col].isna().groupby(df[source_file_col]).mean()
        for source, share in shares.items():
            if share > max_null_share and (col, source) not in allowed_gaps:
                problems.append(f"{col} is {share:.1%} null in {source}")
    if problems:
        raise ValueError(
            f"{rule_name}: group key(s) not populated -- rows with a null key would be "
            f"silently dropped by groupby, so the rule could never fire for them: "
            + "; ".join(problems)
            + ". Fix the key, or declare the gap in allowed_gaps with a reason."
        )


# ---------------------------------------------------------------------------
# Combined label
# ---------------------------------------------------------------------------
def build_is_denied(
    df: pd.DataFrame,
    *,
    payment_col: str = "CLM_PMT_AMT",
    billed_col: str | None = None,
    bene_id_col: str = "BENE_ID",
    hcpcs_col: str = "HCPCS_CD",
    claim_date_col: str = "CLM_FROM_DT",
    dx_cols: tuple[str, ...] = ("PRNCPAL_DGNS_CD",),
    provider_col: str = "PRVDR_NUM",
    specialty_col: str | None = None,
    use_rules: tuple[str, ...] = ("missing_prior_auth", "provider_outlier"),
) -> pd.DataFrame:
    """DEPRECATED as the primary label source -- use denial_reasons.sample_denials()
    instead, which combines these same risk factors probabilistically (noisy-OR)
    and assigns multi-reason CARC labels, rather than a hard boolean OR. Kept
    here for inspecting individual rule hit rates during development.

    `zero_payment` is excluded from the default `use_rules` -- it's a raw
    leakage risk (CLM_PMT_AMT would need to be dropped from features if used
    as a label input; see the note above rule_zero_payment). `dx_procedure_mismatch`
    is excluded here by default for consistency with the original scope of this
    deprecated function -- pass it in use_rules if you want it; its mapping now
    covers real verified codes (see PROCEDURE_TO_EXPECTED_DX_PREFIX above), not
    illustrative placeholders. `duplicate_claim` (Rule 6), `deprecated_code`
    (Rule 7), and `missing_hcpcs` (Rule 8) aren't wired into this deprecated
    function at all -- use denial_reasons.sample_denials() to get them.
    """
    flags = pd.DataFrame(index=df.index)

    if "zero_payment" in use_rules:
        flags["rule_zero_payment"] = rule_zero_payment(df, payment_col, billed_col)
    if "missing_prior_auth" in use_rules:
        flags["rule_missing_prior_auth"] = rule_missing_prior_auth(
            df, bene_id_col, hcpcs_col, claim_date_col
        )
    if "dx_procedure_mismatch" in use_rules:
        flags["rule_dx_procedure_mismatch"] = rule_dx_procedure_mismatch(df, hcpcs_col, dx_cols)
    if "provider_outlier" in use_rules:
        flags["rule_provider_outlier"] = rule_provider_outlier(
            df, provider_col, specialty_col, payment_col
        )

    flags["is_denied"] = flags.any(axis=1).astype(int)
    return flags


def report_rule_hit_rates(flags: pd.DataFrame) -> pd.Series:
    """Per-rule hit rate + combined rate -- put this table directly in the
    README next to the target definition. Reviewers will ask 'what fraction
    of denials came from each rule', have the number ready.
    """
    rule_cols = [c for c in flags.columns if c.startswith("rule_")]
    rates = flags[rule_cols + ["is_denied"]].mean().sort_values(ascending=False)
    return rates


# ---------------------------------------------------------------------------
# Calibration note
# ---------------------------------------------------------------------------
# Sanity-checked against a schema-realistic mock (see repo test run in the
# build session) at ~30% combined denial rate -- above the 5-20% range real
# payer denial rates typically fall in (Phase 1 Step 2 pitfall note). Two
# knobs to pull if your real-data combined rate lands outside 5-20% after
# Phase 1 Step 4 EDA:
#   - `percentile` in rule_provider_outlier (higher = fewer flagged providers)
#   - `LOOKBACK_DAYS` in rule_missing_prior_auth (shorter window = fewer flags,
#     since real per-beneficiary claims cluster in time far more than the
#     mock's uniform-random dates do, so this may self-correct on real data)
# Check df['is_denied'].value_counts(normalize=True) right after Step 3
# wrangling, before moving on to Phase 2 -- don't discover an off-spec class
# balance after you've already built the train/val/test split.
#
# UPDATED 2026-09-16: rule_missing_hcpcs (Rule 8) fired on 38.9% of ALL
# claims at first real-data run (before the carrier-LINE_NUM=1 exclusion
# above), driving is_denied to 46.4% overall -- far outside the 5-20% hard
# bound. The exclusion should shrink this materially (carrier's LINE_NUM=1
# rows are roughly an eighth of the previously-counted missing-HCPCS carrier
# rows), but CALIBRATION_SCALE (in build_target_and_split.py, currently 1.4)
# will still likely need to come DOWN -- rerun calibration_report() after
# this fix and expect to iterate, not to land in range on the first try.
