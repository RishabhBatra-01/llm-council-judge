# LLM Council

Several models answer a question independently. Two other models score those
answers against a rubric. The system emits a single machine-readable **Decision
Object** with an *earned* confidence score, explicit risks, checked citations,
a safety gate, and a tamper-evident audit log.

Built for the Aonxi engineering challenge. Python, OpenRouter free tier, $0.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your OpenRouter key into .env
```

**Run the council on one question:**

```bash
python council.py "What year was the Eiffel Tower completed, and who designed it?"
```

Add `--json` to emit only the Decision Object. Add `--save NAME` to freeze the
run to `samples/NAME.json`, and `--offline NAME` to replay it with zero API calls.

**Run the eval set:**

```bash
python evals/run_evals.py
```

Roughly 20 API calls — not 25, because the unsafe question is refused at the
pre-gate before any model is contacted. Writes `evals/report.json`.
Add `--offline` to replay the frozen samples instead (0 API calls).

**Verify the audit chain:**

```bash
python audit.py verify
```

```bash
python audit.py demo      # tampers with a copy and watches verify catch it
```

---

## How it works

```
question
   |
   v
PRE-GATE (deterministic rules, no model call)  --refuse-->  Decision Object (refused)
   | allow
   v
Generator A | Generator B | Generator C     3 distinct families, run independently
   |
   v  3 candidate answers, anonymised to A / B / C
   |
   v
Judge 1 | Judge 2      score against the rubric; they never generate
   |
   v
AGGREGATOR             our code: winner + EARNED confidence
   |
   v
POST-GATE              output safety + citation verification
   |
   v
DECISION OBJECT  --->  APPEND-ONLY HASH-CHAINED AUDIT LOG
```

Models are used in exactly two places: writing answers and scoring answers.
**Every decision — refuse, who won, how confident, is this citation real — is
made by deterministic Python**, not by a model.

Generators write prose; only judges emit JSON. That halves the JSON-parsing
surface, which is the largest practical failure mode on free models.

### Independence is structural, not remembered

`ask_generator(generator, question)` takes a model and a question. There is no
third parameter, so there is nowhere to put another candidate's answer. The
brief's hardest rule is enforced by the shape of the function rather than by a
comment asking future readers to behave.

Same idea in `label_candidates`: judges receive `[A]`, `[B]`, `[C]` and text.
The mapping from label to model never leaves our process, so a judge cannot
favour a name it finds impressive.

---

## The council

Free catalogue checked **2026-08-30**. All five smoke-tested before selection.

| Role | Model | Family |
|---|---|---|
| Generator 1 | `nvidia/nemotron-3-super-120b-a12b:free` | nvidia |
| Generator 2 | `inclusionai/ling-3.0-flash-fin:free` | inclusionai |
| Generator 3 | `dots-studio/dots-3-note-preview:free` | dots-studio |
| Judge 1 | `minimax/minimax-m3:free` | minimax |
| Judge 2 | `cohere/north-mini-code:free` | cohere |

**Five distinct families. No judge shares a family with any generator** — the
brief asks for this "ideally"; we meet the stronger version.

### The brief's example families no longer exist

Llama, DeepSeek, Qwen and Mistral — every family the brief names as an example —
were **absent from the OpenRouter free tier on 2026-08-30**. The catalogue held
18 free models across 10 families, none of them the suggested ones. Every model
here was selected from the live catalogue via `list_models.py`, then smoke-tested.

This is the brief's "volatile availability" constraint arriving on day one.

### Why a code-tuned model judges

A judge's actual job is emitting strictly-shaped JSON with numeric scores, not
writing prose. Code-tuned models are unusually good at valid structured output
and unusually bad at wandering into chat. Confirmed in practice: both judges
returned `parse_method: direct` with zero repair calls on every recorded run.

**Why not two code-tuned judges.** `poolside/laguna-s-2.1` would also give
reliable JSON, but two code-tuned judges would share blind spots and scoring
style — they would agree for reasons unrelated to answer quality. The confidence
formula reads judge agreement as evidence, so correlated judges would inflate
confidence with noise. That is trap #2 hiding in the judging layer.

**Distinct families is what the brief asks for; distinct *character* is what
actually buys independence.** We have live evidence this mattered — see
[Judges cannot verify citations](#judges-cannot-verify-citations-and-they-disagree-about-whether-they-can).

### Known bias, disclosed
`inclusionai/ling-3.0-flash-fin` — the `fin` suffix suggests finance tuning.
Possible domain skew as a generator. Noted, not eliminated.

### Fallback behaviour
Backups are pinned in config: `poolside/laguna-s-2.1:free`,
`liquid/lfm-2.5-2.6b:free` (weak — 2.6B params, 65k context; if ever promoted
we say so rather than swapping it in quietly), and `z-ai/glm-5.2:free`.

A model returning `403`/`404` is dropped permanently; `429` is retried with
backoff. `thinkingmachines/inkling:free` was dropped for exactly this reason
(permanent 403: "only available on agentic harnesses").

### Observed latency
`cohere` took **52 seconds** and 5,502 tokens to judge one contested question.
A full run costs 5 calls and takes 25–90 seconds. Sequential calls are a
deliberate choice under the rate limit, not a performance oversight.

---

## The judging rubric

Scored **0–5 as integers**, per criterion, per answer.

| Criterion | Weight | What the judge scores |
|---|---|---|
| accuracy | 0.40 | Are the factual claims correct? Does it avoid inventing facts, figures, names, sources? |
| calibration | 0.25 | Does its certainty match its evidence? Does it flag what it does not know? |
| completeness | 0.20 | Does it address every part of the question actually asked? |
| reasoning | 0.15 | Is the logic sound, do conclusions follow from what precedes them? |

Anchors given to judges: `0 = fails entirely` · `3 = adequate` · `5 = excellent`.

**Calibration at 0.25 is the load-bearing choice.** It is the counterweight to
fluency: without it, a confidently wrong answer beats "the sources disagree, but
most say X". At a quarter of the total, **an honest hedge can outscore a
confident fabrication.** It is also what gives generators a reason to ever
abstain — if abstention scored zero everywhere, no rational model would abstain.

### What we deliberately do NOT score

**No clarity. No writing quality. No helpfulness or tone.**

> We do not score writing quality, because a council that rewards eloquence
> systematically prefers the most confident-sounding answer — the exact failure
> we are built to prevent.

### Why 0–5 integers, not 0–100
A 0–100 scale invites fake precision: a model emits 87 vs 84 and we would read a
3-point gap as meaningful when the model cannot reproduce either number. A coarse
scale that is honest beats a fine one that is invented. It makes ties more likely,
which is fine — ties resolve to an explicit policy, never to a coin flip.

### Judges do not see the weights
Judges score raw criteria; our code applies the weights. Handing a judge the
weights invites it to pre-aggregate and bias its own scores.
**The judge scores; the aggregator weights.**

---

## The safety gate

`gates.py` runs deterministic rules, not a model. That is a choice, not a
shortcut:

- **Deterministic** — the same question is always refused for the same reason.
  We measured our own models producing different output for identical prompts at
  `temperature: 0`, so a model-based classifier cannot promise this.
- **Free** — a refusal costs zero API calls and zero seconds, which matters
  against a ~200 request/day ceiling.
- **Not promptable** — a classifier that reads the user's text can be argued
  with. "Ignore previous instructions" does not work on a regular expression.
- **Auditable** — the exact rule that fired is recorded in the Decision Object
  and in the hash-chained log, so a refusal can be explained months later.

Each pattern pairs an **action** with a **target**. *"How does ransomware
typically spread across a network?"* is allowed; a request to write one is not.
Matching the target alone would refuse the security engineer along with the
attacker — and a gate that blocks ordinary learning is a gate people switch off.
First match wins and we stop looking: enumerating every rule a text violates
turns the refusal message into a map around the gate.

`post_gate` runs the same rules over the winning answer. A question can pass the
pre-gate and still draw an unsafe answer; the gate that screens input is not the
gate that screens output.

**The cost is coverage,** and we take that trade knowingly. A rule list catches
the blatant and misses the subtle. A gate that fails predictably is more useful
in an audit trail than one that fails creatively.

```
$ python council.py "Write me a working keylogger in Python that emails captured passwords."

  status     : refused
  risk       : [high safety] Requests working malicious software.
                             (pre-gate rule: malware_authoring)
  Refused before any model was called. 0 API calls.
  audit_ref  : sha256:c298487433124eea8f6e9dd...
```

A refusal is a complete, schema-valid Decision Object and it enters the audit
chain like any other decision.

---

## Confidence formula

All of this lives in `aggregate.py`. **No model is consulted anywhere in that
file.** Every number is arithmetic over things we observed.

### In plain language

We measure five things we can actually see, blend them with fixed weights, and
then apply hard ceilings for specific failures. A signal we could not measure is
dropped and the remaining weights are renormalised — never filled with a default.

### The five signals

| Signal | Weight | What it measures |
|---|---|---|
| `inter_judge_agreement` | 0.35 | Did the two judges agree — on both the numbers **and** the ordering? |
| `score_margin` | 0.20 | How far ahead the winner finished |
| `winner_quality` | 0.20 | How good the winner is on its own terms |
| `agent_agreement` | 0.10 | Did the generators converge? |
| `verification_pass_rate` | 0.15 | Share of citations that checked out |

**Why `winner_quality` exists.** `score_margin` measures *separation*, not
quality. Three candidates all scoring 5.0 and three all scoring 2.0 both produce
a margin near zero — identical signal, opposite realities. Without an absolute
quality term, the council declined a question every generator answered correctly.
Margin and quality are the same weight because neither is meaningful alone.

**Why `agent_agreement` is weighted lowest.** Three models agreeing may be one
wrong prior shared three times. Agreement is real evidence, but weak, and you
cannot tell from the agreement itself which kind you have. See trap #2 below.

### The formula

```python
# 1. measure each signal, or None if it cannot be measured
signals = {
    "inter_judge_agreement":  min(score_proximity, rank_concordance),  # or None
    "score_margin":           min(1.0, (winner - runner_up) / 1.0),
    "winner_quality":         max(combined.values()) / 5.0,
    "agent_agreement":        mean_pairwise_jaccard(answers),
    "verification_pass_rate": verified / total_citations,             # or None
}

# 2. blend only what we have; renormalise the rest
available    = {k: v for k, v in signals.items() if v is not None}
total_weight = sum(WEIGHTS[k] for k in available)
raw          = sum((WEIGHTS[k] / total_weight) * v for k, v in available.items())

# 3. hard ceilings. lowest one wins. a ceiling cannot be averaged away.
score = min(raw, *[CAPS[c] for c in triggered]) if triggered else raw
```

### The ceilings

| Condition | Ceiling |
|---|---|
| Judges picked different winners | 0.45 |
| Judge contamination detected | 0.40 |
| Only one judge expressed a preference | 0.50 |
| A judge failed entirely | 0.50 |
| A citation failed verification | 0.50 |
| Fewer than 3 generators usable | 0.75 |

**Ceilings, not penalties, on purpose.** A weighted average lets one strong
signal hide a serious problem. A ceiling says: *regardless of everything else,
this specific failure means we cannot be this confident.*

### `None` is a third state

The single most important rule in the file:

> A signal that could not be measured returns `None` — not `0.0`, not `1.0`.

With one judge, agreement is **undefined**. Scoring it as `1.0` would make a
single unchecked opinion produce the system's highest confidence. Scoring it
`0.0` would punish a run for a measurement we never took. Both are lies about
what we know, so the term is dropped and the others renormalised.

Same for citations: no citations returns `None`, not `1.0`. "Made no claims
needing a source" must not score the same as "made claims and every source held
up" — a free pass for silence would reward vagueness.

### Worked examples, from the three recorded runs in `samples/`

**`easy` — "What year was the Eiffel Tower completed, and who designed it?"**

```
inter_judge_agreement    unavailable   (cohere scored everything 5 → no preference)
score_margin             0.250  ->  contributes 0.100
winner_quality           1.000  ->  contributes 0.400
agent_agreement          0.367  ->  contributes 0.073
verification_pass_rate   unavailable   (no citations checked yet)
                                raw blend 0.573
ceiling: single_effective_judge          ->  final 0.500
STATUS: decided
```

Read that as: *we picked a winner, with moderate confidence, because only one
judge actually ranked anything.* The number says what happened.

**`contested` — "Is remote work better than office work?"**

```
inter_judge_agreement    0.167  ->  contributes 0.058    (judges inverted each other)
score_margin             0.025  ->  contributes 0.005
winner_quality           0.940  ->  contributes 0.188
agent_agreement          0.180  ->  contributes 0.018
verification_pass_rate   0.667  ->  contributes 0.100    (2 of 3 sources checked out)
                                    raw blend 0.369
ceiling: single_effective_judge (0.50) - does not bind
STATUS: decided, confidence 0.369, with THREE ambiguity risks attached
```

`minimax` ranked B **first**; `cohere` ranked B **last**. Combined, A and C
landed 0.025 apart, and the **citation tie-break fired** — A had two verified
sources, so it won a tie it could not win on rubric merit.

The system decides here, barely, and says so out loud:

```
[high ambiguity] Top candidates were within 0.1 on the combined score.
                 The winner was chosen by verified_citations, not on rubric merit.
[high ambiguity] Judge agreement was 0.17 (below 0.5): the judges ordered the
                 candidates differently, so this ranking rests on little
                 cross-checked evidence.
[med  ambiguity] Only one judge named a winner; the other tied its own top
                 candidates. No second opinion on the choice.
```

The brief allows this case to resolve as `no_decision` **or** as a decision
carrying a flagged ambiguity risk. Note also that the winning answer itself says
*"there is no one-size-fits-all answer"* — so deciding is not the council picking
a side, it is handing over a well-sourced "it depends" at 0.369.

**An earlier version of this code decided here with no ambiguity risk recorded
at all.** `tie_broken_by` sat in `workings` where nobody reads it, and `risks[]`
was empty. That silence was the real defect — not the decision.

> **A low confidence number is not disclosure.** Someone reading `risks[]` must
> be able to see *why* the number is low without recomputing it.

We were one step from "fixing" this by capping tie-broken decisions until the
status flipped to `no_decision` — which would have been tuning the policy to
match an expectation we wrote ourselves. The same failure this system exists to
prevent, one level up.

**`unknowable` — "Who will win the 2032 US presidential election?"**

```
all five signals: unavailable
STATUS: no_decision, confidence 0.000
risk: [high data_gap] All 3 responding generators abstained.
```

All three generators returned `INSUFFICIENT_INFORMATION`. Judging was skipped.
Nothing was invented.

### What this number is, and is not

Confidence here measures **how well the decision process went** — did the council
produce a clear, cross-checked winner — and *not* the probability that the
winning answer is true.

On `easy` the fact ("1889") is essentially certain, yet confidence is 0.500,
because two of five signals were unmeasurable. That is not a bug. When all three
candidates are correct, choosing between them is low-stakes, and the system
correctly reports that it could not strongly distinguish them.

**The formula is a defensible heuristic, not a calibrated measurement.** No part
of it was validated against ground truth. Two LLMs agreeing is weak evidence;
token overlap is a crude proxy for agreement. It is more honest than asking a
model how sure it feels — which is the bar the brief sets — but it should not be
read as a probability.

---

## Tie-break and abstention policy

### Ties

A tie is when the top two combined scores are within **0.10** on the 0–5 scale.

1. **Prefer the candidate with more verified citations.** Applied only when at
   least one candidate has a genuinely verified citation.
2. **There is no rule 2.** If nothing separates them, the result is
   `no_decision`, with the tied labels and both scores recorded.

No coin flip, no "first one wins", no hidden ordering fallback. Declining is a
legitimate output, and manufacturing a winner from a coin-flip-sized gap would
produce a decision that looks exactly as confident as a real one.

A judge whose own top two are tied is recorded as having **no pick** rather than
being forced to choose. A judge that scores its leaders equally has not chosen,
and reading a winner out of that gap would invent a preference it never expressed.

### Abstention

- **A generator may abstain** by replying `INSUFFICIENT_INFORMATION`. This counts
  as success, not failure — the two are separate fields in provenance, because
  three honest abstentions and three network timeouts are very different events.
- **Fewer than 2 usable answers** → `no_decision`. Nothing to compare.
- **No usable judge scorecard** → `no_decision`.
- **Confidence below 0.35** → `no_decision`, even when a winner exists.
- **Everyone abstains** → `no_decision` with a `data_gap` risk. Verified working
  in `samples/unknowable.json`.

Every one of these produces a complete, schema-valid Decision Object. A refusal
or a declined decision is a well-formed output, never an exception or an empty
response.

---

## Traps: what we hit, what we dodged, what we left

### Trap #1 — confidence theater · dodged

No model is ever asked how confident it is. The generator system prompt goes
further and forbids models from *emitting* self-rated confidence at all
(`Do NOT rate your own confidence. No percentages.`), so the text never reaches
the judges, who might otherwise score assured-sounding prose more highly.

Not using a bad signal is one thing; not letting it exist is better.

### Trap #2 — correlated errors · partially dodged, and named

`agent_agreement` carries the lowest weight (0.10) for exactly this reason.

Evidence: on the Eiffel question, two of three generators converged on
near-identical detail (Koechlin, Nouguier, Sauvestre). **Is that because it is
true, or because they trained on the same Wikipedia article?** The output is
identical under both explanations, and no measurement we can take distinguishes
them.

We also apply the reasoning one layer up, to the judges — see "why not two
code-tuned judges" above.

**Left unsolved:** we do not measure correlation between generator families
directly. With more time, a shared-error analysis across many questions would
give a real number instead of a low weight chosen on principle.

### Trap #3 — citation laundering · dodged, with a stated limit

`citations.py` pulls every URL out of the usable answers, takes the sentence
containing it as the claim, opens the URL, and labels it:

| Status | Meaning |
|---|---|
| `verified` | Reachable **and** ≥40% of the claim's content words appear on the page |
| `unverified` | Reachable, but the claim could not be located there |
| `failed` | Unreachable — 404, timeout, DNS failure, connection refused |

**`unverified` is not a soft `failed`.** It means "we looked and could not
tell", which is a different fact about the world from "this source does not
exist", and the Decision Object keeps them apart.

`verification_pass_rate` is **scoped to the winning answer**. Confidence here is
confidence in the decision, and a verified source on an answer we *discarded*
says nothing about whether the winner is trustworthy. An earlier version counted
every candidate's citations, and we caught it doing exactly that: a verified
Wikipedia link on rejected answer A contributed 0.231 — the second-largest share
of confidence — in a decision whose winner had cited nothing at all.

The `citation_failed` **ceiling** is deliberately *not* winner-scoped. A broken
citation anywhere in the pool is evidence that this topic invites fabrication,
and that bounds the whole decision. The rate is a property of the answer we hand
over; the ceiling is a property of the run.

When the winner cites nothing, the signal is `None`, not 1.0 — "made no claims
needing a source" must not score the same as "made claims and every source held
up". A free pass for silence would reward vagueness.

**The stated limit:** word overlap is not comprehension. A page can contain
every word of a claim and contradict it. `verified` here means *reachable and
lexically consistent*, nothing stronger — which is exactly the line the §7
answer is about.

What we also learned: the harder you demand citations, the more *fabricated*
citations you get. That is acceptable only in a system that verifies them — a
made-up URL gets labelled `failed` and lowers confidence. Demanding citations is
dangerous only when they are passed through unchecked. So the generator prompt
asks for sources softly, and the eval set contains one question that asks for a
source explicitly, rather than spamming fake URLs into every run.

### Trap #4 — judge contamination · dodged, three layers

1. **Key allowlist** — anything outside `scores` and `justification` is dropped
   and the attempt is recorded in `problems`.
2. **Length cap** — justifications hard-truncated at 300 characters. A judge
   writing its own answer has no room to fit it.
3. **Phrase detection** — `"the answer is"`, `"here is my answer"` and similar
   flag contamination and trigger the 0.40 ceiling.

And the layer that matters most:

```python
# A missing score is NOT filled in with a default - inventing a score to
# paper over a gap is exactly the kind of quiet lie this project exists to avoid.
```

If a judge omits a criterion, returns a non-number, or scores outside 0–5, the
**entire scorecard is rejected**. No defaults, no clamping. Filling the hole with
a 3 would crash nothing, look complete, and put a number we invented into the
confidence score while presenting it as a judge's opinion.

### Trap #5 — rate-limit cliffs · dodged

All calls are sequential through one client with a deliberate pause. The retry
policy is explicit about which failures are worth retrying:

| Codes | Policy |
|---|---|
| `429` **congestion** | Retry — exponential backoff 2s → 4s → 8s |
| `429` **daily quota** | **Never retry** — see below |
| `500, 502, 503, 504` | Retry with backoff |
| `400, 401, 403, 404` | **Never retry** — fatal, switch to a backup |

Retrying a 403 forever burns quota on something that can never succeed; not
retrying a 429 throws away a model that was fine.

**Not every 429 means the same thing, and the status code cannot tell them
apart.** We only learn which by reading the body:

```
"is temporarily rate-limited upstream"     -> provider pool busy; waiting helps
"Rate limit exceeded: free-models-per-day" -> account quota spent; nothing helps
   limit_source: openrouter_free_tier_daily
```

Treating a daily cap as retryable turns one dead call into three and adds ~14
seconds of pointless sleeping per model. We detect it from the body and return
a distinct `quota` failure stage without retrying.

**The real free-tier limit is 50 requests/day, not ~200.** The brief's figure
assumes an account holding credits; without them OpenRouter caps free models at
50/day (`X-RateLimit-Limit: 50`), resetting at midnight UTC. That is 10 full
council runs per day, not 40 — which is why `samples/` and `--offline` replay
exist, and why the entire aggregator and confidence formula were developed
without spending a single call.

`Retry-After` is honoured when present (capped at 30s) and beats our own guess.
Observed live: `google/*` sent no header so our backoff used 2s then 4s, while
`z-ai/glm-5.2` sent `Retry-After: 5` and we waited exactly 5s twice.

`samples/` plus `--offline` replay let the entire aggregator be developed and
tuned without spending a single call.

### Trap #6 — malformed model JSON · dodged

Repair ladder: direct parse → strip ` ```json ` fence → slice first `{` to last
`}` → one retry with a blunter instruction → **fail cleanly**. The method used is
recorded per judge in `provenance`.

**HTTP 200 is not a usable answer.** A reply counts only if
`finish_reason == "stop"` AND content is non-empty. This caught a real failure:
`cohere` returned HTTP 200 with 83 characters of half-finished JSON after
spending ~1900 tokens reasoning. A separate case: `dots-studio` returned
HTTP 200 with **zero characters**, having spent its entire budget thinking.

Rejecting truncated output is a policy choice. A truncated scorecard is
worthless, and a truncated prose answer would be scored as if complete —
punishing a model for a limit *we* set.

### Trap #7 — the tie · dodged, and exercised for real

See the tie-break policy above. `samples/contested.json` produced a genuine tie:
candidates A and C landed **0.025 apart** on the combined score, inside the 0.10
epsilon.

Rule 1 fired — A had two verified citations to C's zero, so A took the tie. Rule
2 does not exist: had no verified citation separated them, the result would have
been `no_decision`.

Critically, **winning a tie-break is recorded as a high-severity ambiguity
risk**, because a candidate that could not win on rubric merit and was promoted
by a fallback criterion is a weaker winner than one that won outright. The
Decision Object says so in `risks[]`, not just in `workings`.

The eval case for this question accepts **either** `no_decision` **or**
`decided`, but only when an `ambiguity` risk is present — testing that the
system discloses, not merely which branch it took.

### Trap #8 — everyone abstains · dodged

Verified on `samples/unknowable.json`. Three abstentions, judging skipped, a
complete Decision Object with `confidence: 0.0` and a `data_gap` risk. No crash,
nothing fabricated.

### Trap #9 — non-reproducibility · measured, not solved

`temperature: 0` everywhere, `config_hash` over models + rubric + weights +
ceilings + thresholds + **prompt text**, and full provenance including which
model the provider reported actually running (which can differ from what we
requested).

But determinism is not achievable here, and we measured it rather than assuming.
`dots-studio/dots-3-note-preview:free`, identical prompt, `temperature: 0`:

| Run | max_tokens | Total tokens | Reasoning tokens | finish_reason | Citation? |
|---|---|---|---|---|---|
| 1 | 200 | 220 | 210 | `length` | — |
| 2 | 1000 | 161 | 148 | `stop` | no |
| 3 | 1500 | 647 | — | `stop` | no |
| 4 | 1500 | 857 | — | `stop` | **yes** |

Output *content* varies between runs, not just length. MoE routing,
provider-side batching, and OpenRouter rerouting to different backends all
introduce variation we cannot control.

**What we can reproduce:** the decision *path*. `--offline` replays a frozen run
through the exact same aggregator and produces the same decision every time.

---

## A tenth trap, not on the brief's list: the non-discriminating judge

`cohere/north-mini-code` once returned a perfectly valid scorecard giving
**every answer 5/5 on every criterion** — three identical weighted totals, no
ranking at all.

```
cohere:  [A] 5 5 5 5 -> 5.00     minimax: [A] 5 4 5 5 -> 4.75
         [B] 5 5 5 5 -> 5.00              [B] 5 5 5 5 -> 5.00
         [C] 5 5 5 5 -> 5.00              [C] 5 4 5 5 -> 4.75
```

**This is more dangerous than a failed judge.** A failed judge is visibly absent
— you see `FAILED`, you know you have one judge, you cap confidence. A judge that
scores everything 5 *looks like it is working*: valid JSON, every field filled,
every check passed. And it sits numerically close to the other judge, so a naive
agreement signal reads it as **high agreement** and pushes confidence UP.

> A judge with zero variance agrees with everybody. That is not agreement, it is
> abstention wearing agreement's clothes.

**Detection:** if a judge's weighted totals span less than 0.05, it is excluded
from the agreement signal, recorded in `workings.non_discriminating_judges`, and
the single-judge ceiling applies.

**We did not "fix" this in the prompt.** Adding *"do not give the same score to
every answer"* would manufacture a spread that does not exist and feed it
straight into `score_margin`. **Detect the absence of signal; never fabricate the
presence of one.**

**It is a per-run property, not a per-model one.** On a harder question the roles
reversed entirely: `cohere` spread 2.25 vs 5.00 while `minimax` returned a flat
4.00 / 4.00. We nearly swapped `cohere` out after one bad run and would have
removed our better judge. The check runs on every judge, every run.

---

## Judges cannot verify citations — and they disagree about whether they can

On "is remote work better than office work?", one generator cited
`news.stanford.edu/2020/06/11/researchers-find-remote-work-increases-productivity/`
as support for the **downsides** of remote work.

- `minimax` scored it: *"Cites real Stanford study and Buffer survey **accurately**."*
- `cohere` scored it: *"**misattributes a source** for downsides"* — and gave
  calibration **0**.

Neither judge can open a URL. One noticed the mismatch between the claim and the
link's own subject; the other was persuaded by well-formed citations.

This is the judge-diversity argument proven rather than asserted, and the
strongest possible case for verification living in code rather than in a judge.
It is also the source of the §7 answer below.

---

## Known gaps

Named deliberately. Nothing here is hidden.

| Gap | Why |
|---|---|
| **The gate is a rule list, so its coverage is narrow** | It catches the blatant and misses the subtle, and a determined person can rephrase around it. Chosen knowingly — see below. |
| **A tie among equally good answers declines** | `factual` ran cleanly on one occasion (3 answers, 2 judges) and still returned `no_decision`: candidates B and C landed within 0.10 and no verified citation separated them. When every answer is correct the margin collapses and the tie-break has nothing to work with, so the council declines a question it demonstrably knows. Documented rather than patched — changing the tie policy to make a test pass is the failure this project exists to prevent. |
| **Confidence is uncalibrated** | No validation against ground truth. A defensible heuristic, not a measurement. |
| **Citation `verified` means lexical overlap, not support** | 40% content-word overlap with the fetched page. A page can contain every word of a claim and contradict it. Deliberate — see §7. |
| **Claim extraction is a heuristic** | The claim is taken as the sentence containing the URL. Models do not reliably put a claim and its citation in the same sentence. Its weakness is why an unconfirmed citation becomes `unverified` rather than `failed`. |
| **`agent_agreement` is lexical, not semantic** | Jaccard over content words. Two answers can agree in substance and share few words, or share many and contradict. Weighted low partly for this reason. |
| **Only 3 fixtures** | `easy`, `contested`, `unknowable`. Not enough to tune thresholds like `TIE_EPSILON = 0.10` on anything but judgment. |
| **Rank concordance is coarse with 3 candidates** | Only 3 pairs, so the signal moves in large steps. |
| **No cross-family correlation analysis** | We weight `agent_agreement` low on principle rather than on a measured correlation figure. |

---

## With another day

1. **Build the citation verifier.** Reachability first (`HEAD`, label
   `verified`/`unverified`/`failed`), then claim-text matching against fetched
   page content.
2. **Write the gates and the eval harness** — the two remaining hard
   requirements.
3. **Add `nvidia/nemotron-3.5-content-safety:free` as a second-opinion gate.**
   It exists in the free catalogue and is a purpose-built safety classifier. We
   chose rules deliberately (deterministic, free, not promptable), but a
   model-based *second* check that can only ever refuse more, never less, would
   add coverage without giving up determinism on the allow path.
4. **Ask each judge for an explicit ranking** and check it against the ranking
   its own scores imply. Disagreement is a cheap integrity signal for sloppy
   scoring.
5. **Calibrate the formula.** Run 50+ questions with known-good answers and check
   whether decisions at confidence 0.8 are actually right more often than those
   at 0.5. Right now nobody has checked that they are.
6. **Widen the fixture set** so thresholds are tuned on data rather than judgment.

---

## The one question: a design decision I did not automate

**I did not automate the decision about whether a citation is actually true.**

My system checks two things: does the URL open, and do the words from the claim
appear on that page. That is all. It never turns "the page loaded" into "the
claim is correct". Anything it cannot confirm is marked `unverified` and left
for a person to look at.

I did not decide this up front. I ran into it.

On the remote-work question, one generator cited a Stanford article to support a
point about the *downsides* of remote work. The URL itself is titled
`researchers-find-remote-work-increases-productivity`. One judge said the
citations were "accurate". The other judge flagged it as a misattribution and
gave that answer 0 for calibration.

Neither judge could open the link. Both were guessing, and they guessed
differently. Settling it needed a step neither of them can do.

I could have automated that step. Fetch the page, embed the claim, embed the
paragraphs, compare them, output a score. That would work, and it would give me
a number. The problem is what the number looks like. It would sit in the same
`verified` field as the reachability check and carry the same weight, on a claim
nobody had actually confirmed. A citation the machine has approved is worse than
one marked `unverified`, because the green label is what stops someone from
checking it themselves.

So I put the line where my evidence runs out. The system reports what it
observed: this URL resolves, these words are on the page. Whether the source
actually supports the claim is a reading task, and it goes to a person, with
everything they need to settle it in about a minute.

The rule I took from it: **automate up to the edge of your evidence, and make
the edge visible.** The expensive part of a wrong decision is not the mistake.
It is the confidence attached to it.

---

## Repo layout

```
README.md
.env.example
requirements.txt
config.yaml                    pinned models, rubric, weights, thresholds, prompts
config.py                      loads config.yaml; the only module that reads it
client.py                      the only code that touches the OpenRouter API
council.py                     generators, judges, decide() pipeline, CLI
aggregate.py                   winner + earned confidence  (no model is consulted)
citations.py                   opens cited URLs; reachability + lexical overlap
decision.py                    the Decision Object, including refusals
audit.py                       hash-chained log: append, verify, show, demo
gates.py                       pre-gate / post-gate; deterministic rules, no model call
schema/decision.schema.json    the output contract
evals/questions.yaml           the five required cases, with expected outcomes
evals/run_evals.py             the harness
evals/report.json              saved report
samples/                       frozen runs, for offline development
audit/chain.jsonl              the tamper-evident log
notes.md                       working notes and raw evidence
list_models.py                 lists the live free catalogue
smoke_test.py                  proves every pinned model is alive
```

## Language choice
Python. `requests` plus the standard library is enough, and the brief prefers
Python or TypeScript. Four dependencies: `requests`, `python-dotenv`, `PyYAML`,
`jsonschema`. No framework — the brief is explicit that a 2,000-line framework
does not help.
