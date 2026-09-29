# Policy corpus for the `search_policy` tool

Four public CMS documents behind the risk rules in `data/TARGET_DEFINITION.md`. CMS publications are
US government works. Each file below is a **condensed text rendering** produced on 2026-09-29 by
fetching the page and asking a fetch tool to reproduce its policy text. It is not a verbatim copy and
may omit detail. Treat each file as a study aid and check the source URL before relying on a
specific statement. CPT code descriptors (AMA copyright) are deliberately left out; codes are
referred to by number only.

| File | Source | Risk rule it backs |
|---|---|---|
| `cms_r1875cp_consultation_codes.md` | CMS Transmittal 1875 (Claims Processing Manual Ch. 12, Sec. 30.6.10 change) | `deprecated_code` (CARC 181) |
| `lcd_l33718_pap_devices.md` | CMS Local Coverage Determination L33718 | `dx_procedure_mismatch` (CARC 11), CPAP codes |
| `article_a52467_pap_coding.md` | CMS Policy Article A52467 | `dx_procedure_mismatch` (CARC 11), CPAP codes |
| `dmepos_prior_auth_overview.md` | CMS DMEPOS prior authorization process page | `missing_prior_auth` (CARC 197) |

Two source URLs guessed earlier (the MLN Matters MM6740 PDF and the DMEPOS list page) returned 404, so
they are not in the corpus. Documents for the duplicate-claim and provider-outlier rules were not
collected: those rules are anchored to survey statistics, not to a CMS policy text (see
`src/denial_reasons.py`), so `search_policy` correctly has nothing to say about them.
