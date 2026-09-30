"""
Grounding check: which numbers and codes in an explanation trace back to tool outputs?

Deterministic and offline. No LLM is involved, so anyone can rerun it on a saved transcript
and get the same score.

WHAT IT CHECKS. From the explanation text it extracts three kinds of checkable token:
  * code   HCPCS/CPT-style codes (A7038, E0601, 99241), ICD-10 diagnosis codes (Z608, G47.33),
           and policy IDs (L33718, A52467). Grounded if the exact string appears in some tool output.
  * carc   Claim Adjustment Reason Codes written as "CARC 181". Stricter: grounded only if the
           agent called lookup_carc on that code and it was found. A code the model recalled from
           its own training is ungrounded even when it happens to be right.
  * desc   a description attached to a code in parentheses, e.g. "G0444 (annual depression screening)".
           Grounded if at least 60% of the description's content words appear somewhere in the tool
           outputs. This catches a model that describes a code from its own memory. It tests
           traceability, not truth: a correct description that no tool returned still counts as ungrounded.
  * number every other number. Grounded if some number in a tool output matches it after the
           conversions a reader would accept: same value at the precision written, sign ignored
           ("+0.32" vs 0.321 written as 0.32), and probability vs percent (0.1152 vs 11.5%).

WHAT IT DOES NOT CHECK (say this out loud in any write-up):
  * Truth. A grounded number was returned by a tool; the tool could still be wrong.
  * Meaning. "0.32" could be attached to the wrong feature and still match.
  * Bare integers 0 to 10 written without a % or decimal point ("top 3", "2 factors"). They are
    almost always counts or list positions and would match by chance, so they are reported
    separately (`skipped_small_integers`) and left out of the score.
  * Numbers the model derives itself (a difference, a ratio) are NOT grounded unless a tool
    returned that value. That is deliberate: it is how arithmetic errors get caught.
"""

from __future__ import annotations

import json
import re

_CODE_PATTERNS = [
    re.compile(r"\b[A-Z]\d{4,5}\b"),  # A7038, E0601, L33718, A52467
    re.compile(r"\b\d{5}\b"),  # 99241
    re.compile(r"\b[A-TV-Z]\d{2}\.\d{1,4}\b"),  # G47.33
    re.compile(r"\b[A-TV-Z]\d{2,3}[A-Z0-9]{1,4}\b"),  # Z608, T7432X, J329
]
_ANY_CODE = r"(?:[A-Z]\d{4,5}|\d{5}|[A-TV-Z]\d{2}\.\d{1,4}|[A-TV-Z]\d{2,3}[A-Z0-9]{1,4})"
_DESCRIBED = re.compile(rf"(?:\*\*)?\b({_ANY_CODE})\b(?:\*\*)?\s*\(([^)]{{3,90}})\)")
_STOP = {"code", "codes", "with", "that", "this", "from", "used", "type", "kind", "related", "procedure", "service",
         "services", "diagnosis", "billing", "claim", "claims", "medical", "the", "and", "for", "a", "an"}
_CARC = re.compile(r"(?i)\bCARC\s*(?:code\s*)?(\d{1,3})\b")
_NUMBER = re.compile(r"(?<![\w.])[-+\u2212\u2013]?\d[\d,]*(?:\.\d+)?%?")


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def extract_tokens(text: str) -> dict:
    """Split an explanation into carc / code / number tokens (plus skipped small integers)."""
    carcs, spans = [], []
    for m in _CARC.finditer(text):
        carcs.append(m.group(1))
        spans.append(m.span())
    text2 = _mask(text, spans)
    codes, spans = [], []
    for pat in _CODE_PATTERNS:
        for m in pat.finditer(text2):
            if not any(a <= m.start() < b for a, b in spans):
                codes.append(m.group(0))
                spans.append(m.span())
    text3 = _mask(text2, spans)
    numbers, skipped = [], []
    for m in _NUMBER.finditer(text3):
        raw = m.group(0)
        clean = raw.replace(",", "").replace("\u2212", "-").replace("\u2013", "-")
        pct = clean.endswith("%")
        digits = clean.rstrip("%").lstrip("+-")
        try:
            val = float(digits)
        except ValueError:
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if not pct and decimals == 0 and val <= 10:
            skipped.append(raw)
            continue
        numbers.append({"text": raw, "value": val, "decimals": decimals, "percent": pct})
    return {"carc": carcs, "code": list(dict.fromkeys(codes)), "number": numbers, "skipped_small_integers": skipped}


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 4 else w


def _content_words(desc: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z]{4,}", desc.lower()) if w not in _STOP]


def _walk(obj, out_nums: list, out_text: list):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, out_nums, out_text)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out_nums, out_text)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out_nums.append(abs(float(obj)))
    else:
        s = str(obj)
        out_text.append(s)
        for m in _NUMBER.finditer(s):
            try:
                out_nums.append(abs(float(m.group(0).replace(",", "").rstrip("%").lstrip("+-\u2212\u2013"))))
            except ValueError:
                pass


def evidence_from_calls(tool_calls: list[dict], question: str = "") -> dict:
    """tool_calls: [{"name", "input", "output"}]. Returns the pool the explanation is checked against.
    The question and the tool inputs count as evidence too: an id the user typed and the agent passed
    to a tool is not something the model invented."""
    nums: list[float] = []
    texts: list[str] = []
    carc_found: set[str] = set()
    if question:
        _walk(question, nums, texts)
    for c in tool_calls:
        out = c["output"]
        _walk(c.get("input", {}), nums, texts)
        _walk(out, nums, texts)
        if c["name"] == "lookup_carc" and isinstance(out, dict) and out.get("found"):
            carc_found.add(str(out["code"]))
    return {"numbers": nums, "text": "\n".join(texts).upper(), "carc_found": carc_found}


def _number_grounded(tok: dict, pool: list[float]) -> bool:
    v, d = tok["value"], tok["decimals"]
    tol = 0.5 * 10 ** (-d) + 1e-9
    for t in pool:
        if abs(round(t, d) - v) <= tol or abs(t - v) <= tol:
            return True
        if tok["percent"] and (abs(round(t * 100, d) - v) <= tol or abs(t * 100 - v) <= tol):
            return True
    return False


def grade(explanation: str, tool_calls: list[dict], question: str = "") -> dict:
    """Grade one explanation against the tool outputs (and question) of the same run."""
    toks = extract_tokens(explanation)
    ev = evidence_from_calls(tool_calls, question)
    rows = []
    for c in toks["carc"]:
        rows.append({"kind": "carc", "token": f"CARC {c}", "grounded": c in ev["carc_found"]})
    for c in toks["code"]:
        # ICD-10 codes are stored without the dot in the claim data (Z733) but written with it in prose
        # (Z73.3); accept either spelling.
        forms = {c.upper(), c.upper().replace(".", "")}
        hit = any(re.search(rf"(?<![A-Z0-9]){re.escape(f)}(?![A-Z0-9])", ev["text"]) for f in forms)
        if not hit and c.isdigit():
            # a 5-digit number can be an ordinary number a tool returned as a JSON int (odds_change_percent 10330)
            hit = any(abs(t - float(c)) < 1e-9 for t in ev["numbers"])
        rows.append({"kind": "code", "token": c, "grounded": hit})
    for m in _DESCRIBED.finditer(explanation):
        if re.search(r"\d", m.group(2)):
            continue  # a parenthetical holding numbers is not a description; its numbers are graded as numbers
        words = _content_words(m.group(2))
        if not words:
            continue
        low = ev["text"].lower()
        hit = sum(1 for w in words if w in low)
        rows.append({"kind": "desc", "token": f"{m.group(1)} ({m.group(2)})", "grounded": hit / len(words) >= 0.6})
    for n in toks["number"]:
        rows.append({"kind": "number", "token": n["text"], "grounded": _number_grounded(n, ev["numbers"])})
    total = len(rows)
    ok = sum(r["grounded"] for r in rows)
    return {
        "total": total,
        "grounded": ok,
        "share": (ok / total) if total else None,
        "ungrounded": [r for r in rows if not r["grounded"]],
        "tokens": rows,
        "skipped_small_integers": toks["skipped_small_integers"],
    }


# How often does an explanation decline to say what something means? Added 2026-09-29 to measure whether
# sourced definitions (data/agent/code_reference.json) make the answers say more. Grounding alone cannot
# show that: an answer that says nothing is trivially 100% grounded. The phrases were read off the saved
# v2 answers before any definitions were added; a phrase count is a proxy, not a quality score.
DECLINE = re.compile(
    r"(can(?:no|')t say|cannot say|can't tell|cannot tell|no entry|not in (?:the|this) (?:project(?:'s)? )?(?:data dictionary|"
    r"reference)|undocumented|no documented|not documented|don't know what|doesn't say what|does not say what|"
    r"isn't documented|not described|no description)",
    re.I,
)


# The combined count above turned out to mix two different things, which only showed once the definitions
# were added (see docs/PHASE4.md): an answer can now give a field's sourced meaning AND note that the data
# dictionary has no entry for it. That is honest provenance, not a refusal. So the same phrases are also
# split. These two patterns were written AFTER seeing the with-definitions answers, so treat the split as
# post hoc; the combined count (fixed in advance) is reported next to it.
CANNOT_DESCRIBE = re.compile(
    r"(can(?:no|')t (?:say|tell|describe|confirm) what|does not say what|doesn't say what|don't know what|"
    r"can(?:no|')t say what it (?:is|means|describes))",
    re.I,
)
NOT_IN_DICTIONARY = re.compile(
    r"(undocumented|no entry|not in (?:the|this) (?:project(?:'s)? )?data dictionary|not documented|"
    r"no (?:documented )?(?:description|meaning) )",
    re.I,
)


def count_declines(text: str) -> int:
    """Number of decline phrases ("I can't say what this means", "undocumented", ...) in an explanation."""
    return len(DECLINE.findall(text or ""))


def count_cannot_describe(text: str) -> int:
    """Refusals to say what a code or field is ("I can't say what it means")."""
    return len(CANNOT_DESCRIBE.findall(text or ""))


def count_not_in_dictionary(text: str) -> int:
    """Statements that the project data dictionary has no entry (provenance; may sit next to a sourced meaning)."""
    return len(NOT_IN_DICTIONARY.findall(text or ""))


def pooled(graded: list[dict]) -> dict:
    """Micro-average over many explanations (every token counts once)."""
    total = sum(g["total"] for g in graded)
    ok = sum(g["grounded"] for g in graded)
    by_kind: dict[str, list[int]] = {}
    for g in graded:
        for r in g["tokens"]:
            k = by_kind.setdefault(r["kind"], [0, 0])
            k[0] += 1
            k[1] += int(r["grounded"])
    full = sum(1 for g in graded if g["total"] and g["grounded"] == g["total"])
    return {
        "explanations": len(graded),
        "tokens": total,
        "grounded_tokens": ok,
        "share": (ok / total) if total else None,
        "explanations_fully_grounded": full,
        "by_kind": {k: {"tokens": v[0], "grounded": v[1], "share": v[1] / v[0]} for k, v in by_kind.items()},
    }


if __name__ == "__main__":  # python -m src.agent.grounding <transcript.json>
    import sys

    t = json.load(open(sys.argv[1]))
    print(json.dumps(grade(t["final_text"], t["tool_calls"], t.get("question", "")), indent=2))
