"""
The agent: a tool-use loop on the Anthropic Messages API around src/agent/tools.py.

The model decides which of the four tools to call and writes the explanation. Everything it may
quote as a number or code comes from tool outputs; src/agent/grounding.py measures how well it
does that. Every run is saved as a JSON transcript, so results can be replayed and re-graded
offline with no API key (see run_eval.py --replay).

Cost control: a hard budget on the running total, kept in reports/phase4/cost_ledger.json so the cap
holds across separate runs. Prices are per million tokens (USD), from
https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-09-29).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .tools import TOOL_SCHEMAS, ClaimStore, call_tool

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "reports" / "phase4" / "cost_ledger.json"

PRICES = {  # model: (input, output) USD per million tokens
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
CACHE_WRITE_X, CACHE_READ_X = 1.25, 0.10

# Prompt versions are kept side by side so the failure-and-fix story stays reproducible.
SYSTEM_PROMPTS = {
    "v1": (
        "You explain why a health-insurance claim received its denial-risk score. You work for a billing "
        "analytics team. Use the tools to look things up: score the claim, explain the score with SHAP, "
        "look up any denial reason code you want to mention, and search the policy documents when a "
        "policy question comes up. Then write a short, plain-language explanation (about 150 words). "
        "Be accurate. Mention that the denial label in this project is synthetic."
    ),
    "v2": (
        "You explain why a health-insurance claim received its denial-risk score, for a billing analytics "
        "team.\n\n"
        "Process: call score_claim, then explain_shap. Call lookup_carc before you mention any CARC code. "
        "Call search_policy only for policy questions.\n\n"
        "Rules for what you write:\n"
        "1. Numbers: quote numbers exactly as the tools return them. Never calculate, convert, round or count "
        "anything yourself. The tools already give you every number you need (probability_denied, "
        "average_probability_in_validation_set, times_validation_average, points_versus_validation_average, "
        "odds_change_percent, features_not_shown). If you want a number no tool returned, leave it out.\n"
        "2. Codes: you may name a diagnosis or procedure code, but do not say what it means unless a tool result "
        "says so. Do not use your own knowledge of what codes mean.\n"
        "3. Features: describe a feature only with the meaning the tool gives. If the tool says a feature is "
        "undocumented, say that.\n"
        "4. Policy: state a policy fact only if search_policy returned it, and name the document. If no result "
        "covers the point, say the policy documents do not cover it. Do not say an item is on a policy list "
        "unless the result names it.\n"
        "5. SHAP values describe what the model weighs, not what causes a denial. Say 'the model weighs', "
        "not 'causes'.\n"
        "6. Say once, briefly, that the denial label in this project is synthetic.\n"
        "7. After your tool calls, always finish with a written explanation of about 150 words."
    ),
}

QUESTION_TEMPLATES = [
    "Explain why claim {cid} got its denial-risk score.",
    "Claim {cid}: why is its denial risk what it is, and what could a billing team check before submitting?",
    "For claim {cid}, which denial reason code is most plausible, and how does the model's score relate to it?",
]


class BudgetExceeded(RuntimeError):
    pass


def _read_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"total_usd": 0.0, "entries": []}


def ledger_total() -> float:
    return float(_read_ledger()["total_usd"])


def _cost(model: str, usage) -> float:
    pin, pout = PRICES[model]
    inp = usage.input_tokens or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = usage.output_tokens or 0
    return (inp * pin + cw * pin * CACHE_WRITE_X + cr * pin * CACHE_READ_X + out * pout) / 1e6


def record_spend(run_id: str, model: str, usd: float, claims: int) -> float:
    led = _read_ledger()
    led["entries"].append({"run_id": run_id, "model": model, "usd": round(usd, 5), "claims": claims,
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    led["total_usd"] = round(sum(e["usd"] for e in led["entries"]), 5)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))
    return led["total_usd"]


def make_client():
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see docs/LOCAL_SETUP.md). The tools and the "
                           "grounding check run without it; only live agent runs need it.")
    return anthropic.Anthropic(api_key=key)


def run_agent(store: ClaimStore, client, claim_id: int, question: str, model: str, prompt_version: str,
              budget_usd: float, spent_before: float, max_turns: int = 8) -> dict:
    """One claim, one question. Returns a transcript dict (also what gets saved)."""
    system = [{"type": "text", "text": SYSTEM_PROMPTS[prompt_version], "cache_control": {"type": "ephemeral"}}]
    tools = [dict(t) for t in TOOL_SCHEMAS]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    messages = [{"role": "user", "content": question}]
    tool_calls, spent, turns, stop_reasons, nudged = [], 0.0, 0, [], False
    usage_totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    final_text = ""
    while turns < max_turns:
        if spent_before + spent >= budget_usd:
            raise BudgetExceeded(f"budget ${budget_usd:.2f} reached (spent ${spent_before + spent:.3f})")
        turns += 1
        resp = client.messages.create(model=model, max_tokens=1200, system=system, tools=tools, messages=messages)
        spent += _cost(model, resp.usage)
        usage_totals["input"] += resp.usage.input_tokens or 0
        usage_totals["output"] += resp.usage.output_tokens or 0
        usage_totals["cache_write"] += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        usage_totals["cache_read"] += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        stop_reasons.append(resp.stop_reason)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not final_text and not nudged:
                # Seen in 2 of 30 v1 runs: the model stopped with no text (cause unconfirmed; stop reasons were
                # not logged then). Ask once for the written explanation instead of returning nothing.
                nudged = True
                messages.append({"role": "user", "content": "Please write the explanation now."})
                continue
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = call_tool(store, b.name, dict(b.input))
                tool_calls.append({"name": b.name, "input": dict(b.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return {
        "claim_id": int(claim_id),
        "question": question,
        "model": model,
        "prompt_version": prompt_version,
        "turns": turns,
        "stop_reasons": stop_reasons,
        "tool_calls": tool_calls,
        "final_text": final_text,
        "usage": usage_totals,
        "cost_usd": round(spent, 5),
        "finished": bool(final_text),
        "nudged": nudged,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def new_run_id(tag: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
