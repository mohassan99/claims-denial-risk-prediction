"""Offline tests for the Phase 4 agent layer. No API key, no model files, no network needed
except test_tools_end_to_end, which is skipped when the model or claim sample is missing."""

import json
from pathlib import Path

import pytest

from src.agent import grounding
from src.agent.tools import chunk_text, lookup_carc, search_policy

REPO = Path(__file__).resolve().parents[1]


def _calls():
    return [
        {"name": "score_claim", "input": {"claim_id": 42},
         "output": {"claim_id": 42, "hcpcs_cd": "99241", "principal_diagnosis": "Z733", "probability_denied": 0.1152}},
        {"name": "explain_shap", "input": {"claim_id": 42},
         "output": {"top_features": [{"feature": "HCPCS_CD", "shap_log_odds": -0.321, "odds_change_percent": -27}]}},
        {"name": "lookup_carc", "input": {"code": "181"}, "output": {"code": "181", "found": True, "description": "x"}},
    ]


def test_grounded_numbers_and_codes():
    g = grounding.grade("Claim 42 (HCPCS 99241, dx Z73.3) scores 11.5%, or 0.1152. SHAP -0.32, odds down 27%. CARC 181.", _calls(), "Explain claim 42")
    assert g["ungrounded"] == []


def test_model_arithmetic_is_ungrounded():
    g = grounding.grade("The score is 1.9 times higher, about 64 points.", _calls())
    assert {r["token"] for r in g["ungrounded"]} == {"1.9", "64"}


def test_carc_needs_a_lookup():
    g = grounding.grade("Likely CARC 197.", _calls())
    assert [r["token"] for r in g["ungrounded"]] == ["CARC 197"]


def test_invented_code_description_is_flagged():
    g = grounding.grade("HCPCS 99241 (annual depression screening) applies.", _calls())
    assert any(r["kind"] == "desc" and not r["grounded"] for r in g["tokens"])


def test_small_integers_are_skipped_not_scored():
    g = grounding.grade("The top 3 features matter.", _calls())
    assert g["total"] == 0 and g["skipped_small_integers"] == ["3"]


def test_percent_and_rounding_boundary():
    calls = [{"name": "score_claim", "input": {}, "output": {"average": 0.1805}}]
    assert grounding.grade("The average is 18.0%.", calls)["ungrounded"] == []


def test_lookup_carc_known_and_unknown():
    assert lookup_carc("181")["found"] and "official_list_url" in lookup_carc("181")
    assert lookup_carc("CARC 197")["found"]
    assert lookup_carc("45")["found"] is False


def test_chunking_keeps_all_text_and_respects_size():
    body = "\n\n".join(f"para {i} " + "word " * 50 for i in range(12))
    chunks = chunk_text(body, max_words=100, overlap_words=10)
    assert all(len(c.split()) <= 115 for c in chunks)
    assert all(f"para {i} " in " ".join(chunks) for i in range(12))


def test_policy_search_finds_the_right_document():
    top = search_policy("Medicare consultation codes 99241 no longer recognized")["results"][0]
    assert top["doc_id"] == "cms_r1875cp_consultation_codes"
    assert search_policy("xylophone quasar")["results"] == []


@pytest.mark.skipif(not (REPO / "reports" / "xgboost_model.json").exists()
                    or not (REPO / "data" / "processed" / "val_sample_v2.parquet").exists(),
                    reason="needs the downloaded model and the val sample")
def test_tools_end_to_end():
    from src.agent.tools import ClaimStore, explain_shap, score_claim

    store = ClaimStore()
    cid = store.claim_ids()[0]
    s = score_claim(store, cid)
    e = explain_shap(store, cid)
    assert 0 <= s["probability_denied"] <= 1
    # SHAP must add up to the model's own probability (same check fit_shap.py makes)
    assert abs(e["final_probability"] - s["probability_denied"]) < 2e-3
    json.dumps(s), json.dumps(e)


# ---- sourced code and field meanings (data/agent/code_reference.json, added 2026-09-29) ----


def test_describe_value_covers_the_four_cases():
    from src.agent.tools import describe_value

    known = describe_value("HCPCS_CD", "A7038")
    assert known["meaning"] and known["source"].startswith("https://")
    pos = describe_value("LINE_PLACE_OF_SRVC_CD", "20")
    assert pos["meaning"] == "Urgent care facility"
    dx = describe_value("PRNCPAL_DGNS_CD", "Z733")
    assert dx["meaning"].startswith("Z73.3")
    # CPT codes are AMA-licensed: the reference carries only our paraphrase of a CMS document, marked as such
    cpt = describe_value("HCPCS_CD", "99495")
    assert cpt["meaning"] and cpt["source"].startswith("https://www.cms.gov/") and "not the official AMA wording" in cpt["note"]
    # a CPT code with no CMS source on file declines with an explicit reason
    nocms = describe_value("HCPCS_CD", "99401")
    assert nocms["meaning"] is None and "CPT" in nocms["note"]
    # placeholder value and a value that was never verified both decline instead of guessing
    assert describe_value("LINE_PLACE_OF_SRVC_CD", "NOT_APPLICABLE")["meaning"] is None
    assert describe_value("HCPCS_CD", "Q9999")["meaning"] is None  # never looked up
    # a field that holds no coded value gets nothing
    assert describe_value("TAX_NUM", "999996021") is None


def test_field_meaning_order_dictionary_then_reference_then_undocumented():
    from src.agent.tools import feature_meaning_full

    m, documented, src = feature_meaning_full("HCPCS_CD")
    assert documented and src == "data/data_dictionary.md"
    m, documented, src = feature_meaning_full("TAX_NUM")
    assert not documented and src.startswith("https://resdac.org/")
    assert feature_meaning_full("ICD_DGNS_CD7")[2].startswith("https://resdac.org/")  # numbered pattern
    m, documented, src = feature_meaning_full("NOT_A_REAL_FIELD")
    assert not documented and src is None and m.startswith("No entry")


def test_code_reference_integrity():
    ref = json.loads((Path(__file__).resolve().parents[1] / "data" / "agent" / "code_reference.json").read_text())
    left_out = ref["not_verified_left_out"]
    for code in left_out["hcpcs"]:
        assert code not in ref["hcpcs"], f"{code} was never verified; it must not be in the reference"
    for code in left_out["icd10cm"]:
        assert code not in ref["icd10cm"]
    for code in left_out["cpt"]:
        assert code not in ref["cpt"], f"{code} has no CMS source on file; it must not be described"
    assert all(not k.isdigit() for k in ref["hcpcs"]), "CPT (numeric) codes belong in the cpt section, not hcpcs"
    for k, v in ref["cpt"].items():
        assert k.isdigit() and len(k) == 5, k
        assert v["meaning"].strip() and v["source"].startswith("https://www.cms.gov/"), f"{k}: needs a CMS source"
        assert len(v["meaning"].split()) <= 60, f"{k}: keep CPT entries to a short paraphrase, not a copied descriptor"
    for k, v in ref["hcpcs"].items():
        assert (v if isinstance(v, str) else v["meaning"]).strip(), k
        if isinstance(v, dict):
            assert v["source"].strip(), k
    for k, v in ref["icd10cm"].items():
        assert v["code"].replace(".", "").rstrip("-").startswith(k[:3]) and v["title"].strip(), k
    for k, v in ref["fields"].items():
        assert v["source"].startswith("https://resdac.org/") and v["meaning"].strip(), k
    assert set(ref["place_of_service"]) == {"11", "12", "20", "21", "22", "23", "31", "34"}


def test_value_meanings_flow_into_the_tools_and_ground():
    from src.agent.tools import ClaimStore, explain_shap, score_claim

    store = ClaimStore()
    cid = next(c for c in store.claim_ids() if score_claim(store, c).get("hcpcs_cd") not in (None, "NOT_APPLICABLE"))
    s = score_claim(store, cid)
    assert "hcpcs_cd_meaning" in s and "principal_diagnosis_meaning" in s
    e = explain_shap(store, cid, top_k=10)
    assert "meaning_note" in e
    assert all("meaning_source" in f for f in e["top_features"])
    # a description quoted from a tool output counts as grounded; the same words with no tool call do not
    d = s["principal_diagnosis_meaning"].get("meaning")
    if d:
        code = s["principal_diagnosis"]
        calls = [{"name": "score_claim", "input": {"claim_id": cid}, "output": s}]
        assert grounding.grade(f"The diagnosis {code} ({d}).", calls)["ungrounded"] == []


def test_decline_counter():
    assert grounding.count_declines("The project data dictionary has no entry for this field, so I can't say what it means.") == 2
    assert grounding.count_declines("It raises the odds by 66 percent.") == 0
    assert grounding.count_declines("") == 0


def test_decline_split_separates_refusal_from_provenance():
    refusal = "The project data dictionary has no entry for this field, so I can't say what it means."
    provenance = "It is the billing NPI. It is undocumented in the dictionary, and the value is not a real NPI."
    assert grounding.count_cannot_describe(refusal) == 1 and grounding.count_cannot_describe(provenance) == 0
    assert grounding.count_not_in_dictionary(provenance) == 1
    assert grounding.count_declines(provenance) == 1  # the combined count cannot tell them apart


def test_tool_coverage_counts_meanings_not_wording():
    from src.agent.run_eval import tool_coverage

    calls = [{"name": "explain_shap", "input": {}, "output": {"top_features": [
        {"feature": "HCPCS_CD", "value": "A7038", "documented_in_data_dictionary": True,
         "meaning_source": "data/data_dictionary.md", "value_meaning": {"meaning": "Filter"}},
        {"feature": "LINE_PLACE_OF_SRVC_CD", "value": "NOT_APPLICABLE", "documented_in_data_dictionary": False, "meaning_source": None},
        {"feature": "TAX_NUM", "value": "1", "documented_in_data_dictionary": False, "meaning_source": None},
        {"feature": "PRNCPAL_DGNS_CD", "value": "Z733", "documented_in_data_dictionary": True},  # old-style transcript
    ]}}]
    assert tool_coverage(calls) == {"features_shown": 4, "features_with_meaning": 2, "coded_values_shown": 2, "coded_values_with_meaning": 1}
