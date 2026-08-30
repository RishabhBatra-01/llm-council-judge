# LLM Council

Three models answer a question independently. Two other models score those
answers against a rubric. My code — not any model — picks the winner, works out
how much to trust it, checks the citations, and writes the result to a
hash-chained log.

Python, OpenRouter free tier, $0.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenRouter key
```

**Run the council on one question:**

```bash
python council.py "What year was the Eiffel Tower completed, and who designed it?"
```

`--json` emits only the Decision Object. `--save NAME` freezes the run to
`samples/`. `--offline NAME` replays a frozen run with zero API calls.

**Run the eval set:**

```bash
python evals/run_evals.py
```

**Check the audit log:**

```bash
python audit.py verify      # walk the chain
python audit.py demo        # tamper with a copy, watch verify catch it
```

## Flow

```
question
  → pre-gate (rules, no model call)      → refused, 0 API calls
  → 3 generators, independent, sequential
  → answers relabelled A/B/C
  → 2 judges score the anonymised answers
  → citations fetched and checked
  → aggregator: winner + earned confidence
  → post-gate on the winning answer
  → Decision Object, validated against schema/decision.schema.json
  → appended to audit/chain.jsonl
```

Models are used in two places only: writing answers and scoring answers.
Every decision is made in Python.

## Files

```
config.yaml    models, rubric, weights, thresholds, prompts
client.py      the only code that calls OpenRouter
council.py     generators, judges, decide() pipeline, CLI
aggregate.py   winner + confidence (no model involved)
citations.py   opens cited URLs
gates.py       pre-gate / post-gate
decision.py    the Decision Object
audit.py       hash-chained log
evals/         5 test cases, harness, saved report
samples/       frozen runs for offline development
```

## The models

Free catalogue checked 2026-08-30.

| Role | Model | Family |
|---|---|---|
| Generator | `nvidia/nemotron-3-super-120b-a12b:free` | nvidia |
| Generator | `inclusionai/ling-3.0-flash-fin:free` | inclusionai |
| Generator | `dots-studio/dots-3-note-preview:free` | dots-studio |
| Judge | `minimax/minimax-m3:free` | minimax |
| Judge | `cohere/north-mini-code:free` | cohere |

Five families, and no judge shares a family with any generator.

**None of the families in the brief existed on the free tier.** No Llama,
DeepSeek, Qwen or Mistral. I listed the live catalogue with `list_models.py`,
picked five across five families, and smoke-tested each one before building on
them (`smoke_test.py`).

**Why one judge is a code model.** A judge's job is emitting strict JSON with
numeric scores, not writing prose, and code-tuned models are good at that and
bad at drifting into chat. Both judges returned `parse_method: direct` with zero
repair calls on every run. I did not use two code models, because they would
share blind spots and agree for reasons unrelated to answer quality — which the
confidence formula would read as evidence.

**Bias I could not remove:** `ling-3.0-flash-fin` — the `fin` suffix suggests
finance tuning, so it may skew on some topics.

**Fallback:** backups are pinned in `config.yaml` but there is **no automatic
failover** — swapping is manual. What does happen when a model dies: `403`/`404`
is fatal and never retried, `429` is retried with backoff unless the body shows
a daily cap, the run continues with the remaining generators, confidence is
capped at 0.75, and a `data_gap` risk is recorded. `thinkingmachines/inkling`
was dropped by hand after a permanent 403.

## Rubric

Integers 0–5, per criterion, per answer.

| Criterion | Weight |
|---|---|
| accuracy | 0.40 |
| calibration | 0.25 |
| completeness | 0.20 |
| reasoning | 0.15 |

Anchors given to judges: `0 = fails entirely`, `3 = adequate`, `5 = excellent`.

**Calibration is weighted high on purpose.** Without it a confidently wrong
answer beats "sources disagree, but most say X". At a quarter of the total, an
honest hedge can outscore a confident fabrication. It is also the only reason a
generator would ever abstain.

**I deliberately do not score clarity or writing quality.** A council that
rewards eloquence prefers the most confident-sounding answer, which is the
failure this is supposed to catch.

**0–5, not 0–100.** A model asked for a score out of 100 produces 87 and 84, and
I would read a 3-point gap as meaningful when the model cannot reproduce either
number. A coarse scale makes ties more likely, which is fine — ties have a
policy.

Judges never see the weights. They score raw criteria; the aggregator weights.
Candidates are relabelled A/B/C so a judge cannot recognise a model by name.

## Confidence formula

In `aggregate.py`. No model is consulted anywhere in that file.

Five things I can observe:

| Signal | Weight | What it measures |
|---|---|---|
| `inter_judge_agreement` | 0.35 | Did the judges agree on both the numbers and the ordering? |
| `score_margin` | 0.20 | How far ahead the winner finished |
| `winner_quality` | 0.20 | How good the winner is on its own terms |
| `agent_agreement` | 0.10 | Did the generators converge? |
| `verification_pass_rate` | 0.15 | Share of the winner's citations that checked out |

```python
# 1. measure each signal, or None if it cannot be measured
signals = {
    "inter_judge_agreement":  min(score_proximity, rank_concordance),
    "score_margin":           min(1.0, (winner - runner_up) / 1.0),
    "winner_quality":         max(combined.values()) / 5.0,
    "agent_agreement":        mean_pairwise_word_overlap(answers),
    "verification_pass_rate": verified / total_for_the_winner,
}

# 2. blend only what we have; renormalise the rest
available    = {k: v for k, v in signals.items() if v is not None}
total_weight = sum(WEIGHTS[k] for k in available)
raw          = sum((WEIGHTS[k] / total_weight) * v for k, v in available.items())

# 3. ceilings. lowest wins.
score = min(raw, *[CAPS[c] for c in triggered]) if triggered else raw
```

**Ceilings, not penalties:**

| Condition | Ceiling |
|---|---|
| Judges picked different winners | 0.45 |
| Judge contamination detected | 0.40 |
| Fewer than two judges named a winner | 0.50 |
| A judge failed entirely | 0.50 |
| A citation failed verification | 0.50 |
| Fewer than 3 generators usable | 0.75 |

An average lets one good signal hide a serious problem. A ceiling cannot be
averaged away.

**A signal that cannot be measured returns `None`, not 0 and not 1.** With one
judge, agreement is undefined. Scoring it 1.0 would let a single unchecked
opinion produce the highest confidence in the system; scoring it 0 would punish
a run for a measurement I never took. The term is dropped and the rest
renormalised.

Same for citations: a winner that cited nothing returns `None`, not 1.0. "Made
no claims needing a source" must not score the same as "made claims and every
source held up".

### Worked example — the `easy` case

```
score_margin             0.250  ->  0.40 × 0.250 = 0.100
winner_quality           1.000  ->  0.40 × 1.000 = 0.400
agent_agreement          0.367  ->  0.20 × 0.367 = 0.073
inter_judge_agreement    unavailable   (one judge scored everything 5/5)
verification_pass_rate   unavailable   (the winner cited nothing)
                                        raw 0.573
ceiling: fewer than two judges named a winner   ->  final 0.500
```

Weights renormalised because two of five signals were unmeasurable:
0.20+0.20+0.10 = 0.50, so each is divided by 0.50.

**What the number means.** Confidence measures how well the decision *process*
went, not whether the answer is true. On `easy` the fact is certain and
confidence is 0.500, because two of five signals could not be measured. That is
not a bug — when all three candidates are correct, choosing between them is
low-stakes and the system correctly reports it could not strongly distinguish
them.

**It is a heuristic, not a calibrated measurement.** Nothing here was validated
against ground truth. Two LLMs agreeing is weak evidence; word overlap is a
crude proxy. It is more honest than asking a model how sure it feels, which is
the bar the brief sets, but it should not be read as a probability.

## Tie-break and abstention

**Tie** = top two within 0.10 on the 0–5 scale.

1. Prefer the candidate with more verified citations.
2. There is no rule 2. If nothing separates them, the result is `no_decision`.

No coin flip, no "first one wins". Winning a tie-break is recorded as a
high-severity ambiguity risk, because a candidate promoted by a fallback
criterion is a weaker winner than one that won outright.

A judge whose own top two are tied is recorded as having **no pick** rather than
being forced to choose.

**Abstention:**

- A generator may abstain with `INSUFFICIENT_INFORMATION`. That counts as
  success, not failure — three honest abstentions and three timeouts are very
  different events and are stored as separate fields.
- Fewer than 2 usable answers → `no_decision`
- No usable judge scorecard → `no_decision`
- Confidence below 0.35 → `no_decision` even when a winner exists

Every one of these produces a complete, schema-valid Decision Object.

## Traps

**#1 confidence theater — avoided.** No model is asked how sure it is. The
generator prompt also forbids models from *emitting* self-rated confidence, so
the text never reaches the judges.

**#2 correlated errors — partly avoided, and named.** `agent_agreement` carries
the lowest weight. On the Eiffel question two of three generators converged on
near-identical detail — is that because it is true, or because they trained on
the same Wikipedia page? Nothing I can measure distinguishes those. Left
unsolved: I do not measure correlation between families directly, so the low
weight is a principle rather than a number.

**#3 citation laundering — avoided, with a stated limit.** Every URL is fetched.
`verified` = reachable and ≥40% of the claim's words appear on the page.
`unverified` = reachable but the claim was not found. `failed` = unreachable.
`unverified` is not a soft `failed` — it means "I looked and could not tell",
which is a different fact from "this source does not exist". Word overlap is not
comprehension: a page can contain every word of a claim and contradict it.

**#4 judge contamination — avoided.** Only `scores` and `justification` are
accepted, justifications are cut at 300 characters, and giveaway phrases trigger
a ceiling. And if a scorecard is missing a criterion or has a non-number in it,
the whole scorecard is rejected — filling the gap with a 3 would put a number I
invented into the confidence score while presenting it as a judge's opinion.

**#5 rate-limit cliffs — avoided.** Sequential calls with a pause.

| Codes | Policy |
|---|---|
| `429` congestion | retry, backoff 2s → 4s → 8s |
| `429` daily quota | **never retry** |
| `500, 502, 503, 504` | retry with backoff |
| `400, 401, 403, 404` | never retry, fatal |

Two 429s can mean opposite things and the status code cannot tell them apart —
only the body can. `"temporarily rate-limited upstream"` clears in seconds;
`"Rate limit exceeded: free-models-per-day"` does not clear for hours, and
retrying it wastes three calls per model. `Retry-After` is honoured when present
(capped at 30s). Observed live: Google sent no header so my backoff used 2s/4s,
while `z-ai` sent `Retry-After: 5` and I waited exactly 5s twice.

**The real free-tier limit is 50 requests/day, not ~200.** The brief's figure
assumes an account holding credits. 50/day is ten council runs. That is why
`samples/` and `--offline` exist — the entire aggregator and confidence formula
were built and tuned without spending a call.

**#6 malformed JSON — avoided.** Repair ladder: direct parse → strip the
```` ```json ```` fence → slice first `{` to last `}` → one retry with a blunter
instruction → fail cleanly. The method used is recorded per judge.

HTTP 200 is not a usable answer. A reply counts only if `finish_reason == "stop"`
AND content is non-empty. This caught two real failures: `cohere` returned 200
with 83 characters of half-finished JSON after ~1900 reasoning tokens, and
`dots-studio` returned 200 with **zero** characters, having spent its whole
budget thinking.

**#7 the tie — avoided, and exercised.** In `samples/contested.json` two
candidates landed 0.025 apart. Rule 1 fired — one had two verified citations —
and the win was recorded as an ambiguity risk.

**#8 everyone abstains — avoided.** All three generators abstained on the 2032
election question. Judging was skipped, and the result was a clean `no_decision`
with a `data_gap` risk.

**#9 non-reproducibility — measured, not solved.** `temperature: 0` everywhere,
`config_hash` over models + rubric + weights + ceilings + thresholds + prompt
text, and provenance records the model the provider *reported* running, which
can differ from what I requested.

But determinism is not achievable. Same prompt, same `temperature: 0`,
`dots-studio`:

| Run | Total tokens | Reasoning tokens | finish_reason | Citation? |
|---|---|---|---|---|
| 1 | 220 | 210 | `length` | — |
| 2 | 161 | 148 | `stop` | no |
| 3 | 647 | — | `stop` | no |
| 4 | 857 | — | `stop` | **yes** |

Content varies between runs, not just length. What *is* reproducible is the
decision path: `--offline` replays a frozen run through the same aggregator and
gives the same answer every time.

### A tenth trap, not on the list: the non-discriminating judge

`cohere` once returned a perfectly valid scorecard giving every answer 5/5 on
every criterion. Three identical totals, no ranking at all.

This is more dangerous than a failed judge. A failed judge is visibly absent. A
judge that scores everything 5 looks like it is working, and sits numerically
close to the other judge — so a naive agreement signal reads it as **high
agreement** and pushes confidence up, when it said nothing.

Detection: if a judge's totals span less than 0.05 it is excluded from the
agreement signal and the single-judge ceiling applies.

I did not fix this in the prompt. Adding "do not give the same score to every
answer" would manufacture a spread that does not exist and feed it straight into
`score_margin`.

It is a **per-run, not per-model** property. On a harder question the roles
reversed: `cohere` spread 2.25 vs 5.00 while `minimax` returned a flat 4.00/4.00.
I nearly swapped `cohere` out after one bad run and would have removed my better
judge.

### Judges cannot verify citations

On the remote-work question one generator cited
`news.stanford.edu/.../researchers-find-remote-work-increases-productivity/` to
support a point about the **downsides** of remote work.

`minimax` scored it "cites real Stanford study and Buffer survey accurately".
`cohere` called it a misattribution and gave calibration 0.

Neither judge can open a URL. Both were guessing, and they guessed differently.

## The eval report

`evals/report.json` is a live run, all 5 cases, `$0.00`. **3 of 5 matched
expectation.** Both misses are findings.

| Case | Result | |
|---|---|---|
| factual | `no_decision` 0.000 | miss |
| ambiguous | `decided` 0.749 | miss |
| unknowable | `no_decision` 0.000 | ok, generators genuinely abstained |
| unsafe | `refused` 0.000 | ok, 0 API calls |
| citable | `decided` 0.750 | ok, 1 verified + 1 unverified citation |

**Miss 1 — `factual` declined a question it knew.** Every candidate scored a
perfect 5.0 and the top two tied with exactly zero separation. No citation broke
it, so the policy declined. When all answers are correct there is nothing to
rank. I left the tie policy alone: changing it so a test passes is the failure
this project is about.

**Miss 2 — `ambiguous` decided at 0.749 with no ambiguity flagged.** One
generator failed, leaving two candidates — and two candidates give exactly one
pair to compare, so rank concordance is trivially 1.0. A 0.95 from one
comparison is reported the same as 0.95 across three, and it is much weaker.

Underneath that is a bigger limit: **I detect ambiguity in the judging, not in
the question.** Every signal I have is downstream — disagreement, thin margins,
ties. A subjective question where the judges happen to agree is invisible.

I kept the expectation as it is. `require_risk: ambiguity` is testing something
the system cannot always deliver, and that is worth knowing.

**The audit chain recovered this report.** `report.json` was accidentally
overwritten by a later offline run. I rebuilt it from `audit/chain.jsonl`
entries 25–29 without re-running anything — the chain stores each full decision,
so the report is derived and the log is the source of truth. The report records
the `entry_hash` of every entry it came from, so the reconstruction is
checkable. Re-running would also have produced different numbers, given the
non-determinism above.

`evals/report.degraded-example.json` is an earlier attempt where the free tier
rate-limited every generator. Kept on purpose: it shows the council under total
provider failure still emitting a valid Decision Object for each case and
declining rather than inventing answers. The harness flags such runs as
`degraded` so they cannot be mistaken for a measurement of the council.

The audit chain includes development runs, some from before `gates.py` existed
and carrying a "gate not installed" risk. I do not prune it. An append-only log
you edit when it is inconvenient is not an audit log.

## The safety gate

Rules, not a model. Deterministic, free, not promptable ("ignore previous
instructions" does nothing to a regex), and the rule that fired is recorded in
the Decision Object.

Each pattern pairs an **action** with a **target**, so "how does ransomware
typically spread?" is allowed and a request to write one is not. Matching the
target alone would refuse the security engineer along with the attacker. First
match wins and I stop — listing every rule a text violates tells someone which
phrasings to avoid.

`post_gate` runs the same rules over the winning answer.

The cost is coverage: a rule list catches the blatant and misses the subtle. I
take that trade knowingly. A gate that fails predictably is easier to audit than
one that fails creatively.

## Known gaps

- **The judges only use the top of the scale, and this is the most consequential
  flaw here.** Across all 344 criterion scores in the audit chain: 79% are 5,
  20% are 4, and four scores in total fall below 4. `minimax` has never scored
  below 4 in any run. A 0–5 rubric being used as a 4–5 rubric compresses every
  margin toward zero — which is what drives the ties, and what makes the
  `factual` case decline a question the council demonstrably knows the answer
  to. It also weakens `score_margin` and `inter_judge_agreement`, together 55%
  of the confidence weight. The anchors need rewriting so judges will actually
  use the lower half, or the scale needs narrowing to a range they will spread
  across. Either way the current numbers rest on a narrower band of judgement
  than the rubric implies.
- **`winner_quality` was added after a failing test.** The justification is
  sound on its own terms — `score_margin` measures separation, not quality, and
  three candidates at 5.0 look identical to three at 2.0 — but the trigger was
  the `factual` case declining, not the reasoning arriving first. Worth stating
  given what this project is about.
- **The signal weights are chosen, not derived.** 0.35 / 0.20 / 0.20 / 0.10 /
  0.15 come from judgement about what should matter, not from any measurement.
- **No automatic model failover.** Backups are in config; no code selects one.
  Substitution is manual.
- **Confidence is uncalibrated.** Nobody checked whether 0.8 decisions are right
  more often than 0.5 ones.
- **`verified` means lexical overlap, not support.** Deliberate — see below.
- **Claim extraction is a heuristic.** The claim is the sentence containing the
  URL, and models do not reliably put them together.
- **Agreement is not discounted for sample size.** Two candidates give one
  pairwise comparison; three give three. Both report on the same scale.
- **Ambiguity is detected in the judging, not the question.**
- **A tie among equally good answers declines**, even when the answer is known.
- **`agent_agreement` is lexical, not semantic.** Two answers can agree in
  substance and share few words.
- **Only 3 frozen fixtures**, so thresholds like `TIE_EPSILON = 0.10` rest on
  judgment rather than data.

## With another day

1. **Automatic failover.** When a generator returns a fatal status, pick a
   backup from a family not already in play and record the substitution. ~20
   lines; left undone rather than shipped untested.
2. **Fix the rubric before anything else.** Rewrite the anchors so a 2 and a 3
   are reachable — probably by describing concrete failures rather than
   adjectives ("states a fact that is wrong" rather than "adequate"). Then
   re-measure the score distribution. Almost every other numeric weakness here
   is downstream of judges using a 6-point scale as a 2-point one.
3. **Discount agreement by sample size.** One pairwise comparison should not
   report the same number as three.
4. **Detect ambiguity in the question.** Cheap first step: check whether the
   answers themselves hedge. On the ambiguous question all three did, while the
   council reported no ambiguity at all.
5. **Break ties among equally-good answers.** When candidates tie at the top of
   the scale they are interchangeable, so the cost of choosing is near zero —
   but choosing arbitrarily is what I refused to build. A rule that decides when
   `winner_quality` is very high and declines when it is not would separate "all
   excellent" from "all poor".
6. **Calibrate the confidence score.** Run 50+ questions with known answers and
   check whether high-confidence decisions are actually right more often.
7. **A second-opinion safety gate.** `nvidia/nemotron-3.5-content-safety:free`
   exists on the free tier. Rules stay the primary gate, but a model check that
   can only ever refuse *more* would add coverage without giving up determinism
   on the allow path.
8. **Ask judges for an explicit ranking** and compare it to the ranking their own
   scores imply. Disagreement is a cheap signal for sloppy scoring.

## The design decision I did not automate

**I did not automate the decision about whether a citation is actually true.**

My system checks two things: does the URL open, and do the words from the claim
appear on that page. That is all. It never turns "the page loaded" into "the
claim is correct". Anything it cannot confirm is marked `unverified` and left
for a person.

I did not decide this up front. I ran into it.

On the remote-work question, one generator cited a Stanford article to support a
point about the *downsides* of remote work. The URL is titled
`researchers-find-remote-work-increases-productivity`. One judge said the
citations were accurate. The other flagged it as a misattribution and gave that
answer 0 for calibration.

Neither judge could open the link. Both were guessing, and they guessed
differently. Settling it needed a step neither of them can do.

I could have automated that step — fetch the page, embed the claim, embed the
paragraphs, compare, output a score. That works, and it gives me a number. The
problem is what the number looks like. It would sit in the same `verified` field
as the reachability check and carry the same weight, on a claim nobody actually
confirmed. A citation the machine has approved is worse than one marked
`unverified`, because the green label is what stops someone from checking it
themselves.

So I put the line where my evidence runs out. The system reports what it
observed: this URL resolves, these words are on the page. Whether the source
supports the claim is a reading task, and it goes to a person, with everything
they need to settle it in about a minute.

The rule I took from it: automate up to the edge of your evidence, and make the
edge visible. The expensive part of a wrong decision is not the mistake. It is
the confidence attached to it.

## Language

Python. `requests` plus the standard library is enough. Four dependencies:
`requests`, `python-dotenv`, `PyYAML`, `jsonschema`. No framework.
