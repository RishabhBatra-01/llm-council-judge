# LLM Council

Several models answer a question independently. Two other models score those
answers against a rubric. The system emits a single machine-readable **Decision
Object** with an *earned* confidence score, explicit risks, verified citations,
a safety gate, and a tamper-evident audit log.

Built for the Aonxi engineering challenge. Python, OpenRouter free tier, $0.

> **TODO before submit:** delete every `TODO` marker in this file.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your OpenRouter key into .env
```

**Run the council on one question:**

```bash
TODO
```

**Run the eval set:**

```bash
TODO
```

**Verify the audit chain:**

```bash
TODO
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
returned `parsed=direct` with zero repair calls on every run.

**Why not two code-tuned judges.** `poolside/laguna-s-2.1` would also give
reliable JSON, but two code-tuned judges would share blind spots and scoring
style — they would agree for reasons unrelated to answer quality. The confidence
formula reads judge agreement as evidence, so correlated judges would inflate
confidence with noise. That is trap #2 hiding in the judging layer.

**Distinct families is what the brief asks for; distinct *character* is what
actually buys independence.**

### Known bias, disclosed
`inclusionai/ling-3.0-flash-fin` — the `fin` suffix suggests finance tuning.
Possible domain skew as a generator. Noted, not eliminated.

### Fallback behaviour
Backups are pinned in `config.yaml`. A model returning `403`/`404` is dropped
permanently; `429` is retried with backoff. `thinkingmachines/inkling:free` was
dropped for exactly this reason (permanent 403: "only available on agentic
harnesses").

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

### Candidates are anonymised
Answers are relabelled A / B / C before judging. A judge never learns which model
wrote what, so it cannot favour a sibling's style or a name it finds impressive.

---

## Confidence formula

TODO — plain language + code. This is the core deliverable.

---

## Tie-break and abstention policy

TODO

---

## Traps: what we hit, what we dodged, what we left

### Trap #2 — correlated errors
TODO expand. Evidence: on "who designed the Eiffel Tower", two of three
generators converged on near-identical detail. **Is that because it is true, or
because they trained on the same Wikipedia article?** The output is identical
under both explanations. This is why `agent_agreement` is weighted LOW: agreement
is real evidence but weak, and you can never tell which kind you have.

### Trap #5 — rate-limit cliffs
All calls are sequential through one client with a deliberate pause. Retry policy
is explicit about which failures are worth retrying:

| Codes | Policy |
|---|---|
| `429, 500, 502, 503, 504` | Retry — exponential backoff 2s → 4s → 8s |
| `400, 401, 403, 404` | **Never retry** — fatal, switch to a backup |

`Retry-After` is honoured when present (capped at 30s) and beats our own guess.
Observed live: `google/*` sent no header so our backoff used 2s/4s, while
`z-ai/glm-5.2` sent `Retry-After: 5` and we obeyed it exactly.

### Trap #6 — malformed model JSON
Repair ladder: direct parse → strip ```` ```json ```` fence → slice first `{` to
last `}` → one retry with a blunter instruction → **fail cleanly**. The parse
method used is recorded per judge.

**HTTP 200 is not a usable answer.** A reply counts only if
`finish_reason == "stop"` AND content is non-empty. This caught a real failure:
`cohere` returned HTTP 200 with 83 characters of half-finished JSON after
spending ~1900 tokens reasoning. Rejecting truncated output is a policy choice —
a truncated scorecard is worthless, and a truncated prose answer would be scored
as if complete, punishing a model for a limit *we* set.

### Trap #9 — non-reproducibility
Measured, not assumed. `dots-studio/dots-3-note-preview:free`, identical prompt,
`temperature: 0`:

| Run | max_tokens | Total tokens | Reasoning tokens | finish_reason | Citation? |
|---|---|---|---|---|---|
| 1 | 200 | 220 | 210 | `length` | — |
| 2 | 1000 | 161 | 148 | `stop` | no |
| 3 | 1500 | 647 | — | `stop` | no |
| 4 | 1500 | 857 | — | `stop` | **yes** |

Output *content* varies between runs, not just length. Temperature 0 reduces
randomness but does not eliminate it: MoE routing, provider-side batching, and
OpenRouter rerouting to different backends all introduce variation we cannot
control. We log the model the response *reports* rather than the one we
requested, so provenance records what actually ran.

### A tenth trap, not on the brief's list: the non-discriminating judge

TODO expand. `cohere` once returned a perfectly valid scorecard giving **every
answer 5/5 on every criterion** — three identical weighted totals, no ranking at
all. This is more dangerous than a failed judge: a failed judge is visibly
absent, but a judge that scores everything 5 *looks like it is working* and sits
numerically close to the other judge, so `inter_judge_agreement` would read it as
**high agreement** and push confidence UP.

> **A judge with zero variance agrees with everybody. That is not agreement, it
> is abstention wearing agreement's clothes.**

**We did not "fix" this in the prompt.** Adding *"do not give the same score to
every answer"* would manufacture a spread that does not exist, and feed it
straight into `score_margin`. **Detect the absence of signal; never fabricate the
presence of one.**

Note also that non-discrimination proved to be a **per-run, not per-model**
property: on a harder question the roles reversed entirely — `cohere` spread
2.25 vs 5.00 while `minimax` returned a flat 4.00 / 4.00. The check runs on every
judge on every run.

### Traps we knowingly left
TODO

---

## Known gaps

TODO

---

## With another day

TODO

---

## The one question: a design decision I did not automate

TODO — seed: on "is remote work better than office work?", one generator cited
`news.stanford.edu/.../researchers-find-remote-work-increases-productivity/` as
support for the *downsides* of remote work. One judge called the citations
"accurate"; the other flagged the misattribution and scored calibration 0.
**Neither judge could open the URL.** Both were guessing.

---

## Repo layout

```
README.md
.env.example
config.yaml                    pinned model IDs, rubric, weights, thresholds
client.py                      the only code that touches the network
council.py                     generators, judges, aggregation, gates
schema/decision.schema.json
evals/                         eval set + harness + saved report
audit/                         hash-chained log
samples/                       saved runs, for offline development
notes.md                       working notes and raw evidence
```

## Language choice
Python, for the standard reason: `requests` + stdlib is enough, and the brief
prefers Python or TypeScript. Dependencies kept to four
(`requests`, `python-dotenv`, `pyyaml`, `jsonschema`).
