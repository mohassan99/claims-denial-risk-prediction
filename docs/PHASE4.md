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
| `data/agent/code_reference.json` | Sourced meanings for fields, place-of-service codes, HCPCS Level II codes and ICD-10-CM codes (added 2026-09-29, see below). |
| `data/policy/` | Four CMS documents behind the risk rules, chunked for `search_policy`. Read its README first. |
| `reports/phase4/runs/` | Every run: config, one JSON transcript per claim (tool calls, outputs, answer, cost), summary. |
| `reports/phase4/cost_ledger.json` | Running API spend across all runs. |
| `scripts/make_val_sample.py` | Builds the claim sample from Databricks. |
| `tests/test_agent.py` | Offline tests for the grader, CARC lookup, chunking, policy search, and SHAP-sums-to-score. |

## The four tools

1. `score_claim(claim_id)`: runs the XGBoost model (v2, hash `11f409f72fc4c45e`) locally through
   `deploy/score.py`, the same code the Azure endpoint runs. Returns the probability, the validation-set
   average, and precomputed comparisons.
2. `explain_shap(claim_id, top_k)`: SHAP values for that claim in log-odds, each joined to its meaning: first
   from `data/data_dictionary.md`, then from the sourced reference `data/agent/code_reference.json`. For coded
   fields (procedure, diagnosis, place of service) it also returns what the value means. Anything in neither
   place is marked undocumented rather than described, and every meaning names its source. The SHAP values
   plus the baseline reproduce the model's probability (tested). `score_claim` returns the same value
   meanings for the claim's procedure and principal diagnosis.
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

## Adding sourced definitions (2026-09-29)

Why this was done, what was done, how it was tested, and what it did and did not fix. Written so a reader can
follow the reasoning without having been there.

### The problem, measured before anything was changed

After the Phase 4 numbers came in (100% grounded), I read one answer end to end (claim 420689). It was fully
grounded and honest, and it told a billing reviewer almost nothing: it said "the data dictionary has no entry
for this field, so I can't say what it means" about the biggest driver. Grounding cannot see this problem,
because an answer that says nothing is trivially 100% grounded. So I measured it. In the 30 held-out answers
(old tools), 27 said in some form that they could not describe something (79 phrases), and only 68 of the 180
top features the agent was shown came with a meaning. None of the 74 coded values (procedure, diagnosis,
place of service) came with one.

The features doing the most work were the same handful every time: place of service (`LINE_PLACE_OF_SRVC_CD`),
the provider identifier fields (`TAX_NUM`, `CARR_CLM_BLG_NPI_NUM`, `ORG_NPI_NUM`), `LINE_NUM`, and the values of
the procedure and diagnosis codes themselves.

### The options I considered

1. **Let the model use what it knows about codes.** Rejected. This is exactly the failure from the first smoke run,
   where a model called G0444 a consultation code when it is annual depression screening. Prompt rule 2 exists to stop it.
2. **Put the definitions in the prompt.** Rejected: a long static list is paid for on every call, cannot be
   sourced entry by entry, and gives no way to say "this one was not verified".
3. **Add a fifth tool, `lookup_code`.** Rejected: the agent would have to decide to call it, and the meanings are
   only needed for the values already in front of it. Extra tool calls also cost money and add a place to go wrong.
4. **Join the meanings into the tools' existing output (chosen).** `score_claim` and `explain_shap` already return the
   claim's fields and codes, so they now return each meaning next to its value, with its source. The prompt
   (v2) was left exactly as it was, so the only thing that changed between the two runs is the tool output.

A separate file, `data/agent/code_reference.json`, holds the meanings instead of editing `data/data_dictionary.md`.
The dictionary is the project's own reference for its columns; the new file holds outside facts, and each entry
carries its outside source. The tools check the dictionary first, then the reference, and report which one
answered (`meaning_source`).

### Which codes, and how they were picked

Codes: HCPCS codes seen 3 or more times, and diagnosis codes seen 4 or more times, in the 600-row validation
sample. That is 26 HCPCS codes (73.7% of rows) and 24 diagnosis codes (83.7% of rows), plus all 9 place-of-service
values in the sample. Fields: I counted which features appeared most often among the top features shown to the agent
in the two earlier 30-claim runs, and defined the ones that came up repeatedly and were undocumented.

Two honest limits on independence. First, the field list was chosen partly from the held-out run's own tool
outputs, so the after test is not independent for fields. Second, the 600-row sample contains the held-out
claims, so a code that is common in the sample is more likely to appear in the evaluation. Both favor the after
run a little. I did not read any answer text to choose entries, and I dropped one code (E1390) that I had
verified because it appeared only in the evaluation claims and would have been a direct advantage.

### Where the definitions came from, and what could not be verified

Each entry was checked against an official source in this session, not written from memory:

| What | Source | Count verified |
|---|---|---|
| Field meanings (`TAX_NUM`, NPI fields, `LINE_NUM`, place of service, pricing locality, and others) | ResDAC (the Research Data Assistance Center) variable pages for CMS claims files | 9 fields |
| Place of service codes | CMS Place of Service Code Set page | 8 codes |
| HCPCS Level II codes (the letter-prefixed ones) | Long descriptions from the National Library of Medicine's Clinical Table Search Service, which serves the CMS code set | 13 codes |
| ICD-10-CM diagnosis codes | Code titles from the same NLM service, which serves the CMS and NCHS code set | 22 codes |

Things that did not go to plan, and how each was handled:

- **Rate limits.** Six lookups were refused with an HTTP 429 error and the proxy said not to retry those pages.
  They are HCPCS A4604, G9572, M1069, H2001 and ICD-10 C50.929, M54.50. I did not retry, and I did not fill them in from
  memory. Five of them (all but A4604, below) are listed in the reference under `not_verified_left_out`, and a test fails if any of them is
  ever added without being verified.
- **A4604** (tubing for a heated humidifier) was the one HCPCS code whose lookup was refused but that a CMS document
  already in this repo describes (policy article A52467, condensed in `data/policy/`). It is included, and its source line says
  the description comes from the policy article, not from the HCPCS file.
- **CPT codes are left without descriptions on purpose.** A 5-digit numeric code in `HCPCS_CD` is a CPT code
  (HCPCS Level I), and the American Medical Association owns and licenses CPT descriptions. These public
  sources do not carry them, and copying them would not be right. The tools tell the agent it is a CPT code and give
  the reason, so it can decline with a reason. This matters: 90935 and 99241 are the two most common codes in the sample.
- **The sources corrected me.** I had guessed that diagnosis `T7432X` meant psychological abuse. The official
  title is "Child psychological abuse, confirmed", and the official codes in that family carry a seventh character
  the data value lacks. That is why entries are looked up, not remembered.
- **`CARR_CLM_RFRNG_PIN_NUM` was left out.** The only page I found for it listed a different short name, so its
  name could not be matched exactly. The rule was: a field is included only if a source matches its name.
- **These are mirrors, not the CMS files themselves.** The NLM service republishes the CMS code sets. That is a
  stated source, not a claim that I read the CMS release file.

The reference also carries a note that the claims are synthetic, and the tool output repeats it, so the agent says
what a code is and does not tell a story about a patient. I read the answers for this (below).

### How it was tested

1. **Unit tests** (17 in total now, from 10): meanings are returned for known values; CPT codes, the
   `NOT_APPLICABLE` placeholder and unverified values decline instead of guessing; the order of lookup is dictionary,
   then reference, then undocumented; an integrity test fails if an unverified code sneaks in, if a CPT code is given
   a description, or if any entry lacks a source; a description quoted from a tool output grades as grounded.
2. **A controlled before and after.** The same 30 held-out claims, the same prompt (v2), the same model
   (Claude Sonnet 5.5), the same questions. Only the tool output changed. Run `20260929-174122-heldout-v2-defs`
   against `20260928-224007-heldout-v2`. Added `--claims-from` to the runner to re-run exactly a previous run's claims.
3. **A hand read** of the after answers for anything the reference did not support, especially the diagnosis
   codes with sensitive titles.

### Results

| Measure (30 held-out claims) | Before | After |
|---|---|---|
| Checkable tokens grounded | 542 of 542 | 511 of 511 |
| Top features shown that came with a meaning | 68 of 180 (38%) | 178 of 180 (99%) |
| Coded values (procedure, diagnosis, place of service) that came with a meaning | 0 of 74 | 56 of 74 (76%) |
| Explanations saying they cannot say what an item is | 14 of 30 (16 phrases) | 8 of 30 (10 phrases) |
| Explanations noting "not in the dictionary" | 23 of 30 (53 phrases) | 18 of 30 (36 phrases) |
| Description checks passed (a description after a code) | 7 of 7 | 14 of 14 |
| Cost per claim | $0.019 | $0.021 |

What the numbers say, and one measurement mistake I made on the way:

- **Grounding held at 100%.** Giving the agent more to say did not make it invent anything the tools did not return.
- **Coverage is the clean number.** It reads the tool outputs, not the answer wording, so it measures what the agent was
  able to say. Features with a meaning went from 38% to 99%; code values from 0% to 76%.
- **The wording counts moved less, and my first proxy was misleading.** I first counted "decline phrases" as one number.
  It barely moved (27 to 28 explanations) and I nearly read that as failure. Reading the sentences showed why: the agent now
  gives a field's sourced meaning and then adds "it is undocumented in the dictionary" as a provenance note. That is
  honest, but the phrase counter reads it as a refusal. I split the counter into refusals to say what an item is
  and provenance notes. **The split was written after I saw the after answers, so it is post hoc.** The
  original combined count is kept in the summary files next to it, unchanged.
- **The 18 code values still without a meaning:** 11 are CPT codes (99241 four times, 90935 three times, and one each of
  45378, 99408, 99495, 96127), and 7 are rare diagnosis codes outside the frequency cutoff (K011, J029, L209, E034,
  O039, Y0703, J441). The 2 features without a meaning are `CARR_CLM_RFRNG_PIN_NUM`.
- **The hand read found no story about a patient.** The answers said things like "the tool lists this as Child
  psychological abuse, confirmed" and, in the same place, that the encounter type is unknown. They attributed
  meanings to the tool, and several added useful reviewer checks ("confirm that the place-of-service code matches where the
  item was supplied"). That read was by eye on 46 sentences. It is not a measured rate.
- **Checkable tokens fell from 542 to 511.** I did not investigate why. One guess is that answers describe more
  and quote fewer numbers, but that is a guess.
- **One sample per claim.** Both runs are single samples, so a difference of a few explanations is inside the run-to-run
  variation this project has already seen.

### What is still open

- CPT descriptions. This is the largest remaining gap, because CPT codes are the most common codes in the data.
  Options: a CMS source for the specific codes, or accept the honest decline with its stated reason.
- The 5 unverified items above (A4604 is covered by a CMS policy article), and the rare diagnosis codes outside the cutoff.
- The retrieval weakness seen in the demo (the most relevant policy chunk ranked last) is unchanged.

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

- Meanings for fields and common codes were added on 2026-09-29 (see "Adding sourced definitions"). CPT
  descriptions, 5 unverified items and rare diagnosis codes are still missing, so the agent still declines on those.
- Retrieval is lexical and unevaluated. Only rules with a CMS policy behind them are covered: nothing
  supports duplicate-claim or provider-outlier questions, and the agent says so.
- The Azure endpoint was not used. `score_claim` runs the same scoring code locally. Restarting the
  endpoint and pointing the tool at it is possible but was out of scope and costs money.
