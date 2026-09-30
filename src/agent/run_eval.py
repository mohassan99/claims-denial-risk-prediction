"""
Run or replay the Phase 4 grounding evaluation.

  Live (needs ANTHROPIC_API_KEY, spends money, stops at the budget):
    python -m src.agent.run_eval --model claude-haiku-4-5 --prompt v1 --n-per-type 1 --tag smoke
    python -m src.agent.run_eval --model claude-sonnet-5-5 --prompt v1 --n-per-type 10 --tag final-v1

  Replay (no API key, no cost): re-grade saved transcripts
    python -m src.agent.run_eval --replay reports/phase4/runs/<run_id>

Claims are chosen deterministically (seed 42): per claim type, half denied and half not denied,
so explanations cover both outcomes. The agent is never told the label.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import grounding
from .agent import (QUESTION_TEMPLATES, BudgetExceeded, ledger_total, make_client, new_run_id, record_spend,
                    run_agent)
from .tools import DEFAULT_SAMPLE, ClaimStore

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "reports" / "phase4" / "runs"


def pick_claims(df: pd.DataFrame, n_per_type: int, seed: int = 42, exclude: set[int] | None = None) -> list[int]:
    rng = np.random.RandomState(seed)
    if exclude:
        df = df[~df["_row_id"].isin(exclude)]
    chosen: list[int] = []
    for ct in ("carrier", "outpatient", "dme"):
        sub = df[df[f"claim_type_{ct}"].astype(float) == 1]
        for label, n in ((1, (n_per_type + 1) // 2), (0, n_per_type // 2)):
            pool = sub[sub["is_denied"].astype(int) == label]["_row_id"].to_numpy()
            take = min(n, len(pool))
            chosen += [int(x) for x in rng.choice(pool, size=take, replace=False)]
    return chosen


_CODED = re.compile(r"^(HCPCS_CD|PRNCPAL_DGNS_CD|LINE_PLACE_OF_SRVC_CD|ICD_DGNS_CD[0-9]+)$")


def tool_coverage(tool_calls: list[dict]) -> dict:
    """How much of what the agent was SHOWN came with a usable meaning. Wording-free: it reads the tool outputs,
    not the answer text, so it measures what the agent could say, whatever it chose to say.
    Works on old transcripts too (they have no meaning_source or value_meaning, so those count as absent)."""
    shown = with_meaning = coded = coded_with_meaning = 0
    for c in tool_calls:
        if c["name"] != "explain_shap" or not isinstance(c["output"], dict):
            continue
        for f in c["output"].get("top_features", []):
            shown += 1
            if f.get("documented_in_data_dictionary") or f.get("meaning_source"):
                with_meaning += 1
            if _CODED.match(f["feature"]) and f.get("value") not in (None, "NOT_APPLICABLE"):
                coded += 1
                vm = f.get("value_meaning")
                if vm and vm.get("meaning"):
                    coded_with_meaning += 1
    return {"features_shown": shown, "features_with_meaning": with_meaning,
            "coded_values_shown": coded, "coded_values_with_meaning": coded_with_meaning}


def summarize(run_dir: Path) -> dict:
    graded, rows, cov = [], [], []
    for f in sorted(run_dir.glob("claim_*.json")):
        t = json.loads(f.read_text())
        g = grounding.grade(t["final_text"], t["tool_calls"], t["question"])
        graded.append(g)
        cov.append(tool_coverage(t["tool_calls"]))
        rows.append({"claim_id": t["claim_id"], "question": t["question"], "tools": [c["name"] for c in t["tool_calls"]],
                     "tokens": g["total"], "grounded": g["grounded"], "share": g["share"],
                     "ungrounded": [r["token"] for r in g["ungrounded"]], "cost_usd": t["cost_usd"],
                     "declines": grounding.count_declines(t["final_text"]),
                     "cannot_describe": grounding.count_cannot_describe(t["final_text"]),
                     "not_in_dictionary": grounding.count_not_in_dictionary(t["final_text"])})
    empty = sum(1 for f in run_dir.glob("claim_*.json") if not json.loads(f.read_text())["final_text"])
    pooled = grounding.pooled(graded)
    pooled["empty_explanations"] = empty
    pooled["tool_coverage"] = {k: sum(c[k] for c in cov) for k in cov[0]} if cov else {}
    pooled["explanations_with_a_decline"] = sum(1 for r in rows if r["declines"])
    pooled["decline_phrases"] = sum(r["declines"] for r in rows)
    pooled["explanations_that_cannot_describe_something"] = sum(1 for r in rows if r["cannot_describe"])
    pooled["cannot_describe_phrases"] = sum(r["cannot_describe"] for r in rows)
    pooled["explanations_noting_not_in_dictionary"] = sum(1 for r in rows if r["not_in_dictionary"])
    pooled["not_in_dictionary_phrases"] = sum(r["not_in_dictionary"] for r in rows)
    out = {"run": run_dir.name, "pooled": pooled, "per_claim": rows,
           "cost_usd": round(sum(r["cost_usd"] for r in rows), 4)}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def print_summary(s: dict) -> None:
    p = s["pooled"]
    print(f"\nrun {s['run']}: {p['explanations']} explanations, {p['tokens']} checkable tokens, "
          f"{p['grounded_tokens']} grounded = {100 * (p['share'] or 0):.1f}%; "
          f"{p['explanations_fully_grounded']} fully grounded, {p.get('empty_explanations', 0)} empty; cost ${s['cost_usd']:.3f}")
    print(f"   declines: {p.get('explanations_with_a_decline', 0)} of {p['explanations']} explanations say they can't describe "
          f"something ({p.get('decline_phrases', 0)} phrases)")
    print(f"     of which: cannot describe what an item is: {p.get('explanations_that_cannot_describe_something', 0)} explanations "
          f"({p.get('cannot_describe_phrases', 0)} phrases); notes 'not in the dictionary': "
          f"{p.get('explanations_noting_not_in_dictionary', 0)} explanations ({p.get('not_in_dictionary_phrases', 0)} phrases)")
    tc = p.get("tool_coverage")
    if tc and tc.get("features_shown"):
        print(f"   tool coverage: {tc['features_with_meaning']}/{tc['features_shown']} features shown came with a meaning; "
              f"{tc['coded_values_with_meaning']}/{tc['coded_values_shown']} code values (procedure, diagnosis, place of service) came with a meaning")
    for k, v in p["by_kind"].items():
        print(f"   {k:7} {v['grounded']}/{v['tokens']} = {100 * v['share']:.1f}%")
    for r in s["per_claim"]:
        if r["ungrounded"]:
            print(f"   claim {r['claim_id']}: ungrounded {r['ungrounded']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--budget", type=float, default=2.50, help="stop when the ledger total reaches this (USD)")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-run", type=Path, help="skip claims used in this earlier run (a held-out check)")
    ap.add_argument("--claims-from", type=Path, help="re-run exactly the claims of this earlier run (a controlled before/after)")
    a = ap.parse_args()

    if a.replay:
        print_summary(summarize(a.replay))
        return 0

    store = ClaimStore(a.sample)
    client = make_client()
    excl = set(json.loads((a.exclude_run / "config.json").read_text())["claims"]) if a.exclude_run else None
    if a.claims_from:
        claims = [int(c) for c in json.loads((a.claims_from / "config.json").read_text())["claims"]]
    else:
        claims = pick_claims(store.df, a.n_per_type, a.seed, excl)
    run_id = new_run_id(a.tag)
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": a.model, "prompt": a.prompt, "claims": claims,
                                                     "sample": str(a.sample.name), "budget_usd": a.budget, "seed": a.seed,
                                                     "excluded_run": a.exclude_run.name if a.exclude_run else None,
                                                     "claims_from": a.claims_from.name if a.claims_from else None}, indent=2))
    spent_before, run_spent = ledger_total(), 0.0
    print(f"{run_id}: {len(claims)} claims, model {a.model}, prompt {a.prompt}, ledger so far ${spent_before:.3f}, cap ${a.budget:.2f}")
    try:
        for i, cid in enumerate(claims):
            q = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)].format(cid=cid)
            t = run_agent(store, client, cid, q, a.model, a.prompt, a.budget, spent_before + run_spent)
            run_spent += t["cost_usd"]
            (run_dir / f"claim_{cid}.json").write_text(json.dumps(t, indent=2))
            g = grounding.grade(t["final_text"], t["tool_calls"], t["question"])
            print(f"  [{i + 1}/{len(claims)}] claim {cid}: {len(t['tool_calls'])} tool calls, "
                  f"grounded {g['grounded']}/{g['total']}, ${t['cost_usd']:.4f}, run total ${run_spent:.3f}")
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    finally:
        total = record_spend(run_id, a.model, run_spent, len(list(run_dir.glob('claim_*.json'))))
        print(f"ledger total now ${total:.3f}")
    print_summary(summarize(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
