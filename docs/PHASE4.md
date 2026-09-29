# Phase 4: agentic claim explainer

An LLM agent that explains why one claim got its denial-risk score, plus a deterministic check that
measures how much of what it says can be traced back to the tools it called.

The point of this phase is the measurement, not the chat interface. The model can be fluent and wrong.
The grounding check shows how often it stays inside what the tools returned, and the failures below
show what happened when it did not.

## Terms used here

- **Agent / tool use:** the LLM is given a list of functions (tools). It decides which to call and with
  what arguments, reads the results, and writes the answer. The code runs the functions.
- **SHAP:** a method that splits one prediction into per-feature contributions that add up to the score.
- **Log-odds:** the scale the model adds contributions on before converting to a probability,
  p = 1 / (1 + exp(-x)). Positive raises denial risk, negative lowers it. exp(x) is the factor applied
  to the odds of denial, so a contribution of -0.32 multiplies the odds by 0.73.
- **CARC:** Claim Adjustment Reason Code, the standard code payers use to say why a claim was adjusted or
  denied. X12 owns the official list, so this repo carries short paraphrases and links to it.
- **RAG:** retrieval-augmented generation. Fetch the most relevant text chunks from a document
  collection and hand them to the model, so it answers from them instead of from memory.
- **Grounded:** a number or code in the answer that appears in a tool output (rules below).

## What is in the repo

| Path | What |
|---|---|
| `src/agent/tools.py` | The four tools as plain Python functions, plus a command-line entry point. No LLM, no API key. |
| `src/agent/agent.py` | The tool-use loop (Anthropic Messages API), prompt versions v1 and v2, cost ledger and budget cap. |
| `src/agent/grounding.py` | The grounding check. Deterministic, offline. |
| `src/agent/run_eval.py` | Runs the 30-claim evaluation live, or re-grades saved transcripts offline (`--replay`). |
| `data/agent/carc_reference.json` | Paraphrased CARC descriptions for 11, 16, 18, 181, 197. |
| `data/policy/` | Four CMS documents behind the risk rules, chunked for `search_policy`. Read its README first. |
| `reports/phase4/runs/` | Every run: config, one JSON transcript per claim (tool calls, outputs, answer, cost), summary. |
| `reports/phase4/cost_ledger.json` | Running API spend across all runs. |
| `scripts/make_val_sample.py` | Builds the claim sample from Databricks. |
| `tests/test_agent.py` | Offline tests for the grader, CARC lookup, chunking, policy search, and SHAP-sums-to-score. |

## The four tools

1. `score_claim(claim_id)`: runs the XGBoost model (v2, hash `11f409f72fc4c45e`) locally through
   `deploy/score.py`, the same code the Azure endpoint runs. Returns the probability, the validation-set
   average, and precomputed comparisons.
2. `explain_shap(claim_id, top_k)`: SHAP values for that claim in log-odds, each joined to its meaning from
   `data/data_dictionary.md`. A feature with no dictionary entry is marked undocumented rather than described.
   The SHAP values plus the baseline reproduce the model's probability (tested).
3. `lookup_carc(code)`: paraphrased description and which project rule uses the code. Unknown codes return
   "not in the reference".
4. `search_policy(query, k)`: retrieval over the policy corpus. Chunks are packed paragraphs of up to 140
   words with a 30-word overlap. Retrieval is TF-IDF cosine similarity (lexical matching, no embedding
   model), chosen so it needs no download and gives the same result every time. There is no chunking study
   and no retrieval-quality evaluation. Swapping in embeddings would be a small change in `_policy_index`.

## How grounding is scored

For each explanation the checker extracts four kinds of token and looks for them in that run's tool
outputs (plus the question and the tool inputs, so a claim id the user typed is not flagged):

- **code:** HCPCS/CPT-style, ICD-10 (written with or without the dot) and policy IDs. Exact string match.
- **carc:** "CARC 181". Grounded only if the agent called `lookup_carc` on that code and it was found.
- **desc:** a description in parentheses after a code, such as "G0444 (annual depression screening)".
  Grounded if 60% of its content words appear in tool outputs. Traceability, not truth.
- **number:** anything else numeric, matched after rounding to the written precision, ignoring sign, and
  allowing probability vs percent. Bare integers 0 to 10 are skipped (list positions and counts would match by chance).

What it cannot tell you: whether a grounded number is attached to the right feature, whether the tool was
right, or whether the explanation is useful. A number the model computes itself counts as ungrounded on
purpose. That is how arithmetic errors are caught.

## Results

30 claims per run (10 per claim type, half denied and half not, seed 42), one answer per claim, Claude
Sonnet 5.5, claims from the current val split. The agent is never told the label. `held-out` uses 30
different claims (seed 7, none shared with the first set).

| Run | Prompt | Tool outputs | Checkable tokens | Grounded | Share | Empty answers |
|---|---|---|---|---|---|---|
| final-v1 | v1 | original | 364 | 355 | 97.5% | 2 |
| final-v1-newtools | v1 | + precomputed numbers | 409 | 406 | 99.3% | 0 |
| final-v2-newtools | v2 | + precomputed numbers | 540 | 540 | 100.0% | 0 |
| heldout-v2 | v2 | + precomputed numbers | 542 | 542 | 100.0% | 0 |

Cost: about $0.017 to $0.019 per claim on Sonnet 5.5 ($0.52 to $0.57 per 30-claim run). Total API spend
for the phase, including the failed and repeated runs, is in `reports/phase4/cost_ledger.json`.

Read these numbers with care:

- The v2 prompt was written after looking at the first 30 claims, so `final-v2-newtools` is not a clean
  test. `heldout-v2` is the honest one, and it agrees.
- Grounded means traceable, not correct or useful. v2 is deliberately cautious: it often says "I can't say
  what this code means" instead of guessing. That raises grounding and lowers how much the answer tells you.
- One sample per claim. The empty answers below were not reproducible, which is a reminder that LLM output varies.
- The `desc` check only sees parenthetical descriptions. I also searched the v2 answers by hand for
  unparenthesized ones and found none: all 11 hits were the model declining to describe a code.
- The label is synthetic (engineered from rules), and so are the CARC codes attached to claims. An
  explanation says why the model scored a claim, not why a payer would deny it.

## Failures, and what fixed them

**1. The model did arithmetic, and some of it was ungrounded.** In v1, 9 of 364 checkable tokens were
numbers the model derived itself: an odds multiplier of 0.72 became "lowers the odds by 28%", a probability
ratio became "1.5 times as risky", and 159 features minus the 6 shown became "the other 150 or so". Nothing was
wrong with the tools; they had returned 0.72 and left the conversion to the model. Fix: the tools now
return those values precomputed (`odds_change_percent`, `times_validation_average`,
`points_versus_validation_average`, `features_not_shown`), and prompt v2 forbids calculating, converting,
rounding or counting. Effect on the same 30 claims: 9 ungrounded tokens fell to 0 with the new tool fields
alone (prompt unchanged), and stayed 0 with v2.

**2. The model described codes from its own memory, and once got it wrong.** In the first 3-claim smoke
run (Haiku), claim 420689 was explained as "HCPCS code G0444 (a consultation code)". Per this project's
own documentation (`data/TARGET_DEFINITION.md`), G0444 is annual depression screening. The code string
was in the tool output, so a code-only check passed it. All 4 parenthetical descriptions in that
run were missing from the tool outputs. Fix: a `desc` check in the grader, and rule 2 in prompt v2 (name a code, but
do not say what it means unless a tool did). With the tool fields fixed but the v1 prompt, Sonnet still
added 3 untraceable descriptions of 16 ("99397 (a preventive visit)"); under v2 there were none.

**3. The model over-stated a policy.** In the smoke run, claim 1721550 (DME code A7037) was explained as
"consistent with CMS policy that respiratory equipment requires prior authorization". The retrieved
DMEPOS page lists orthoses, power mobility devices, pressure-reducing surfaces, lower-limb prostheses and
compression devices, not respiratory equipment. Numbers and codes all matched, so the grounding score could
not see it; I found it by reading. Fix: v2 rule 4 (state a policy fact only if a search result returned it,
name the document, and do not say an item is on a list unless the result names it). Under v2 the agent said
things like "no policy result says A7038 is a PAP item". This one is a judgment from reading a few answers,
not a measured rate.

**4. Two answers came back empty (v1 run, 2 of 30).** The loop ended with no text after the tool calls.
I did not log stop reasons then, and 2 re-runs of one of those claims both worked, so the cause is
unconfirmed. Changes made: stop reasons are now stored, one automatic "please write the explanation now"
nudge if a final turn has no text, and a larger output limit (1,200 tokens instead of 900). There have been
0 empty answers in the 90 answers since, and the nudge has never fired, so I cannot say which change (if any) mattered.

**5. My first grader was wrong.** The first score for the v1 run was 94.0%, not 97.5%. Three bugs made
13 tokens look ungrounded: ICD-10 codes written with a dot (Z73.3) did not match the dotless form in the
data (Z733); the claim id from the question was not counted as evidence; and 18.05 written as 18.0% failed
a rounding comparison. Later, a 5-digit number ("10330 percent") was read as a code, and a parenthetical
holding numbers was read as a description. All are fixed, every saved run was re-graded with the final
grader, and the tests pin each case. Grading the grader before trusting it is why the table above uses one
grader version throughout.

## Run it yourself

Setup once: `docs/LOCAL_SETUP.md` (creates `.env`, gets the model). The claim sample:

```bash
python scripts/make_val_sample.py            # needs DATABRICKS_HOST and DATABRICKS_TOKEN in .env
```

No API key needed:

```bash
python -m src.agent.tools list_claims
python -m src.agent.tools score_claim <claim_id>
python -m src.agent.tools explain_shap <claim_id>
python -m src.agent.tools lookup_carc 181
python -m src.agent.tools search_policy "prior authorization DME"
python -m src.agent.run_eval --replay reports/phase4/runs/20260928-224007-heldout-v2   # re-grade saved answers
python -m pytest tests/test_agent.py -q
```

Live agent runs need `ANTHROPIC_API_KEY` in `.env` and cost money (about $0.02 per claim on Sonnet 5.5):

```bash
python -m src.agent.run_eval --model claude-sonnet-5-5 --prompt v2 --n-per-type 10 --tag mine --budget 1.00
```

`--budget` stops the run when the ledger total reaches that many dollars.

## Known limits and next steps

- The dictionary covers few features, so many top SHAP features are reported as undocumented. Adding
  verified feature and code meanings (with sources) would let the agent say more without guessing.
- Retrieval is lexical and unevaluated. Only rules with a CMS policy behind them are covered: nothing
  supports duplicate-claim or provider-outlier questions, and the agent says so.
- The Azure endpoint was not used. `score_claim` runs the same scoring code locally. Restarting the
  endpoint and pointing the tool at it is possible but was out of scope and costs money.
