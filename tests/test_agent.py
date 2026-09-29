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
