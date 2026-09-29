"""
Phase 4 tools for the agentic claim explainer.

Four plain Python functions. None of them calls an LLM, so the whole layer can be
run, tested and demonstrated with no API key:

    python -m src.agent.tools list_claims
    python -m src.agent.tools score_claim <claim_id>
    python -m src.agent.tools explain_shap <claim_id>
    python -m src.agent.tools lookup_carc 181
    python -m src.agent.tools search_policy "prior authorization DME"

A "claim_id" here is the `_row_id` of a row in the claim sample file (default
data/processed/val_sample_v2.parquet, drawn from the val split). `_row_id` is the global
row identifier the label pipeline keys on; it is not a CMS field.

Design rules (they exist so the agent's numbers can be traced):
  * Every number the agent may quote is computed here, rounded here, and returned here.
    The model is told never to do arithmetic on them; derived values (probability
    equivalents, odds ratios) are precomputed in the tool output.
  * SHAP values are in log-odds (logit) units. Log-odds is the scale a logistic-style
    model adds up on before converting to a probability: p = 1 / (1 + exp(-x)).
    Positive raises denial risk, negative lowers it. Every SHAP row says so.
  * Undocumented features are labeled as undocumented rather than described from memory.
"""

from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
DEFAULT_SAMPLE = REPO / "data" / "processed" / "val_sample_v2.parquet"
DATA_DICTIONARY = REPO / "data" / "data_dictionary.md"
CARC_REFERENCE = REPO / "data" / "agent" / "carc_reference.json"
POLICY_DIR = REPO / "data" / "policy"

sys.path.insert(0, str(REPO / "deploy"))  # deploy/score.py is the single scoring implementation
import score as _score  # noqa: E402

_CLAIM_TYPES = ("carrier", "outpatient", "dme")
# Columns that identify a claim for humans; shown by score_claim, never model features by themselves.
_ID_FIELDS = {"HCPCS_CD": "hcpcs_cd", "PRNCPAL_DGNS_CD": "principal_diagnosis"}


def _num(x, nd: int):
    """Round a float for output; NaN/None -> None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- claims + model


class ClaimStore:
    """Loads the model once and serves claim rows by `_row_id`."""

    def __init__(self, sample_path: Path | str = DEFAULT_SAMPLE):
        self.sample_path = Path(sample_path)
        if not self.sample_path.exists():
            raise FileNotFoundError(
                f"{self.sample_path} not found. Build it from Databricks or from val_model.parquet "
                "(see docs/PHASE4.md, 'Getting claims')."
            )
        self.df = pd.read_parquet(self.sample_path).reset_index(drop=True)
        if "_row_id" not in self.df.columns:
            raise ValueError("claim sample needs a _row_id column")
        self.df["_row_id"] = self.df["_row_id"].astype("int64")
        self._by_id = {int(r): i for i, r in enumerate(self.df["_row_id"])}
        _score.init()
        self.model = _score._model
        fit = json.loads((REPORTS / "xgboost_fit.json").read_text())
        self.val_mean_predicted = fit["val_metrics_overall"]["mean_predicted"]
        self._explainer = None
        self._X_cache: dict[int, tuple[pd.DataFrame, int]] = {}

    def claim_ids(self) -> list[int]:
        return [int(x) for x in self.df["_row_id"]]

    def row(self, claim_id: int) -> pd.Series:
        try:
            return self.df.iloc[self._by_id[int(claim_id)]]
        except (KeyError, ValueError):
            raise KeyError(f"unknown claim_id {claim_id}") from None

    def claim_type(self, row: pd.Series) -> str:
        for ct in _CLAIM_TYPES:
            if float(row.get(f"claim_type_{ct}", 0) or 0) == 1:
                return ct
        return "unknown"

    def features(self, claim_id: int) -> tuple[pd.DataFrame, int]:
        """(1-row model frame, count of category values the model never saw in training)."""
        cid = int(claim_id)
        if cid not in self._X_cache:
            row = self.row(cid)
            rec = json.loads(row.drop(labels=[c for c in ("is_denied", "_rn") if c in row.index]).to_json())
            X, unseen = _score.prepare_features([rec])
            self._X_cache[cid] = (X, int(unseen[0]))
        return self._X_cache[cid]

    def explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
        return self._explainer


@lru_cache(maxsize=1)
def _feature_meanings() -> dict[str, str]:
    """Parse data/data_dictionary.md: table rows whose first cell holds backticked column names."""
    meanings: dict[str, str] = {}
    for line in DATA_DICTIONARY.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        names = re.findall(r"`([A-Za-z0-9_]+)`", cells[0])
        for n in names:
            meanings.setdefault(n, cells[1])
    return meanings


_UNDOCUMENTED = (
    "No entry in data/data_dictionary.md. It is a native CMS claim field (see the CMS/ResDAC variable "
    "documentation); its meaning is not described in this project's dictionary."
)


def feature_meaning(name: str) -> tuple[str, bool]:
    m = _feature_meanings().get(name)
    return (m, True) if m else (_UNDOCUMENTED, False)


# --------------------------------------------------------------------------- tool 1


def score_claim(store: ClaimStore, claim_id: int) -> dict:
    """Run the local XGBoost model on one claim."""
    row = store.row(claim_id)
    X, unseen = store.features(claim_id)
    p = float(store.model.predict_proba(X)[:, 1][0])
    out = {
        "claim_id": int(claim_id),
        "claim_type": store.claim_type(row),
        "probability_denied": _num(p, 4),
        "average_probability_in_validation_set": _num(store.val_mean_predicted, 4),
        # Precomputed so the agent never has to do arithmetic (see grounding.py and docs/PHASE4.md).
        "times_validation_average": _num(p / store.val_mean_predicted, 2),
        "points_versus_validation_average": _num((p - store.val_mean_predicted) * 100, 1),
        "points_note": "percentage points; negative means below the validation average",
        "unseen_category_values": unseen,
        "model": "claims-denial-xgboost v2 (content hash 11f409f72fc4c45e), run locally",
        "label_caveat": "The denial label is synthetic (engineered from risk rules), so this is the "
        "probability of the constructed label, not of a real payer decision.",
    }
    for col, key in _ID_FIELDS.items():
        v = row.get(col)
        out[key] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    return out


# --------------------------------------------------------------------------- tool 2


def explain_shap(store: ClaimStore, claim_id: int, top_k: int = 6) -> dict:
    """Top SHAP contributions for one claim, joined to feature meanings."""
    top_k = max(1, min(int(top_k), 10))
    X, _ = store.features(claim_id)
    exp = store.explainer()
    sv = np.asarray(exp.shap_values(X))[0]
    base = float(np.ravel(exp.expected_value)[0])
    total = base + float(sv.sum())
    order = np.argsort(-np.abs(sv))
    feats = []
    for i in order[:top_k]:
        name = X.columns[i]
        raw = X.iloc[0, i]
        val = None if pd.isna(raw) else (str(raw) if str(X[name].dtype) == "category" else _num(raw, 4))
        meaning, documented = feature_meaning(name)
        s = float(sv[i])
        feats.append(
            {
                "feature": name,
                "value": val,
                "shap_log_odds": _num(s, 3),
                "effect": "raises denial risk" if s > 0 else "lowers denial risk",
                "odds_multiplier": _num(math.exp(s), 2),
                "odds_change_percent": int(round((math.exp(s) - 1) * 100)),
                "meaning": meaning,
                "documented_in_data_dictionary": documented,
            }
        )
    rest = float(sv[order[top_k:]].sum())
    return {
        "claim_id": int(claim_id),
        "units": "log-odds. Positive raises denial risk, negative lowers it. odds_multiplier = exp(shap_log_odds), the factor this "
        "feature applies to the odds of denial; odds_change_percent = the same as a percent change (negative lowers the odds).",
        "baseline_log_odds": _num(base, 3),
        "baseline_probability": _num(_sigmoid(base), 4),
        "final_log_odds": _num(total, 3),
        "final_probability": _num(_sigmoid(total), 4),
        "top_features": feats,
        "all_other_features_log_odds": _num(rest, 3),
        "number_of_features": int(len(sv)),
        "features_not_shown": int(len(sv) - top_k),
    }


# --------------------------------------------------------------------------- tool 3


@lru_cache(maxsize=1)
def _carc() -> dict:
    return json.loads(CARC_REFERENCE.read_text(encoding="utf-8"))


def lookup_carc(code: str) -> dict:
    """Paraphrased CARC description for the codes this project uses."""
    ref = _carc()
    key = str(code).strip().upper().replace("CARC", "").strip()
    hit = ref["codes"].get(key)
    if not hit:
        return {
            "code": key,
            "found": False,
            "message": "This code is not in the project's small reference (it covers "
            + ", ".join(sorted(ref["codes"], key=int))
            + "). Check the official list; do not guess its meaning.",
            "official_list_url": ref["official_list_url"],
        }
    return {
        "code": key,
        "found": True,
        "description": hit["description"],
        "project_rule": hit["project_rule"],
        "note": ref["source_note"],
        "label_caveat": ref["label_caveat"],
        "official_list_url": ref["official_list_url"],
    }


# --------------------------------------------------------------------------- tool 4


def _split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        meta = {}
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body.strip()
    return {}, text


def chunk_text(body: str, max_words: int = 140, overlap_words: int = 30) -> list[str]:
    """Basic chunking: keep paragraphs together, pack them up to max_words, carry a short tail
    forward as overlap. A paragraph longer than max_words is split by words. No tuning study."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= max_words:
            pieces.append(p)
        else:
            step = max_words - overlap_words
            pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), step))
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len((cur + " " + piece).split()) > max_words:
            chunks.append(cur)
            tail = " ".join(cur.split()[-overlap_words:])
            cur = tail + " " + piece
        else:
            cur = (cur + "\n\n" + piece) if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


@lru_cache(maxsize=1)
def _policy_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    records = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
        for i, ch in enumerate(chunk_text(body)):
            records.append(
                {"doc_id": path.stem, "title": meta.get("title", path.stem), "url": meta.get("url", ""),
                 "chunk_id": f"{path.stem}#{i}", "text": ch}
            )
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([r["title"] + " " + r["text"] for r in records])
    return records, vec, mat


def search_policy(query: str, k: int = 3) -> dict:
    """Retrieve the k most relevant policy chunks (TF-IDF cosine similarity, lexical retrieval)."""
    k = max(1, min(int(k), 5))
    records, vec, mat = _policy_index()
    q = vec.transform([query])
    sims = (mat @ q.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    results = [
        {**{kk: records[i][kk] for kk in ("doc_id", "title", "url", "chunk_id", "text")}, "similarity": _num(sims[i], 3)}
        for i in order
        if sims[i] > 0
    ]
    return {
        "query": query,
        "results": results,
        "note": "Chunks come from condensed renderings of public CMS documents (data/policy/README.md); "
        "they are not verbatim and cover only the rules with a CMS policy behind them. "
        "If nothing relevant is returned, say so instead of answering from memory.",
    }


# --------------------------------------------------------------------------- registry + CLI

TOOL_SCHEMAS = [
    {
        "name": "score_claim",
        "description": "Run the trained denial-risk model on one claim from the sample. Returns claim type, "
        "HCPCS code, principal diagnosis, probability of denial and the validation-set average probability.",
        "input_schema": {"type": "object", "properties": {"claim_id": {"type": "integer"}}, "required": ["claim_id"]},
    },
    {
        "name": "explain_shap",
        "description": "Explain one claim's score: the features that pushed it up or down the most (SHAP values in "
        "log-odds), with each feature's meaning from the project data dictionary. Call after score_claim.",
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "integer"}, "top_k": {"type": "integer", "description": "1 to 10, default 6"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "lookup_carc",
        "description": "Look up a Claim Adjustment Reason Code (CARC) in the project's small paraphrased reference "
        "(codes 11, 16, 18, 181, 197). Use it for any code you want to mention.",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    {
        "name": "search_policy",
        "description": "Search the small corpus of public CMS policy documents behind the project's risk rules "
        "(consultation-code rule, CPAP coverage and coding, DMEPOS prior authorization). Use for 'why does Medicare "
        "deny X' or 'what fixes Y' questions.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "description": "1 to 5, default 3"}},
            "required": ["query"],
        },
    },
]


def call_tool(store: ClaimStore, name: str, args: dict) -> dict:
    """Dispatch one tool call. Errors come back as {"error": ...} so the agent can see them."""
    try:
        if name == "score_claim":
            return score_claim(store, int(args["claim_id"]))
        if name == "explain_shap":
            return explain_shap(store, int(args["claim_id"]), int(args.get("top_k", 6)))
        if name == "lookup_carc":
            return lookup_carc(str(args["code"]))
        if name == "search_policy":
            return search_policy(str(args["query"]), int(args.get("k", 3)))
        return {"error": f"unknown tool {name}"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent on purpose
        return {"error": f"{type(exc).__name__}: {exc}"}


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"list_claims", "score_claim", "explain_shap", "lookup_carc", "search_policy"}:
        print(__doc__)
        return 2
    cmd, rest = argv[1], argv[2:]
    if cmd == "lookup_carc":
        print(json.dumps(lookup_carc(rest[0]), indent=2))
    elif cmd == "search_policy":
        print(json.dumps(search_policy(" ".join(rest)), indent=2))
    else:
        store = ClaimStore()
        if cmd == "list_claims":
            cols = ["_row_id"] + [c for c in ("HCPCS_CD", "PRNCPAL_DGNS_CD", "is_denied") if c in store.df.columns]
            print(store.df[cols].head(30).to_string(index=False))
        elif cmd == "score_claim":
            print(json.dumps(score_claim(store, int(rest[0])), indent=2))
        else:
            print(json.dumps(explain_shap(store, int(rest[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
