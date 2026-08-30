# Working notes — LLM Council

Scratch file. Becomes `config.yaml` (Phase 6) and feeds the README.

**Free catalogue checked: 2026-08-30** — 18 free models across 10 families.

---

## The council (all smoke-tested 2026-08-30)

| Role | Model ID | Family | Latency | Why |
|---|---|---|---|---|
| Generator 1 | `nvidia/nemotron-3-super-120b-a12b:free` | nvidia | ~1.4–3.2s | Largest available (120B total / 12B active) — best general knowledge |
| Generator 2 | `inclusionai/ling-3.0-flash-fin:free` | inclusionai | ~1.5s | Fast, reliable in every run |
| Generator 3 | `dots-studio/dots-3-note-preview:free` | dots-studio | ~2.2s | Heavy reasoner — genuinely different thinking style |
| Judge 1 | `minimax/minimax-m3:free` | minimax | ~3.5–4.2s | 1M context, no heavy reasoning overhead |
| Judge 2 | `cohere/north-mini-code:free` | cohere | ~0.8s | Fastest; code-tuned → reliable structured JSON |

**Five distinct families. No judge shares a family with any generator.**
(The brief only asks for this "ideally" — we meet the stronger version.)

### Backups (tested, working)
- `poolside/laguna-s-2.1:free` — poolside, ~1.2s, code-tuned
- `liquid/lfm-2.5-2.6b:free` — liquid, ~1.4s — **weak**: only 2.6B params, 65k context.
  Last resort only; if promoted, say so explicitly in the README.
- `minimax/minimax-m2.7:free` — minimax — slow (~7.5s) AND collides with Judge 1's family
- `z-ai/glm-5.2:free` — z-ai — persistently 429 on 2026-08-30, retry another day

### Unavailable
- `thinkingmachines/inkling:free` — **HTTP 403, permanent**: "only available on
  agentic harnesses". Not retryable. Dropped.
- `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free` — HTTP 429 upstream
  on every attempt 2026-08-30. Not our account (minimax succeeded seconds before).
  Congestion is per-model and time-of-day dependent.

---

## Role-assignment reasoning (for the README)

**Why a code model as a judge.** A judge's actual job is emitting strictly-shaped
JSON with numeric scores, not writing prose. Code-tuned models are unusually good
at valid structured output and unusually bad at wandering into chat. Directly
attacks trap #6 (malformed model JSON).

**Why not two code models as judges.** `poolside` would also give reliable JSON,
but two code-tuned judges would share blind spots and scoring style — they'd agree
for reasons unrelated to answer quality. The confidence formula reads judge
agreement as evidence, so correlated judges would inflate confidence with noise.
That is trap #2 hiding in the judging layer.

→ One code-tuned judge (structural reliability) + one general judge (judgment quality).
**Distinct families is what the brief asks for; distinct character is what actually
buys independence.**

**Known bias to disclose:** `ling-3.0-flash-fin` — the "fin" suffix suggests
finance tuning. Possible domain skew as a generator. Noted, not eliminated.

---

## Catalogue volatility (for the README)

The brief's own example families — Llama, DeepSeek, Qwen, Mistral — **were all absent
from the free tier on 2026-08-30**. Every model here was selected from the live
catalogue, not from the brief's examples. This is the "volatile availability"
constraint arriving on day one.

---

## Evidence: non-determinism at temperature 0 (trap #9)

`dots-studio/dots-3-note-preview:free`, identical prompt, `temperature: 0`:

| Run | max_tokens | Total tokens | Reasoning tokens | finish_reason |
|---|---|---|---|---|
| 1 | 200 | 220 | 210 | `length` |
| 2 | 1000 | 161 | 148 | `stop` |

Reasoning length varied ~40% between runs. Temperature 0 reduces randomness but
does not eliminate it — MoE routing, provider-side batching, and OpenRouter
rerouting to different backends all introduce variation we cannot control.

**Measured, not assumed.**

---

## Settings decided so far

| Setting | Value | Reason |
|---|---|---|
| `temperature` | `0` | Best available shot at reproducibility |
| `max_tokens` | `1000` | 200 starved reasoning models — one used 210 tokens thinking, hit the ceiling, answered nothing. May need raising for judges. |
| `timeout` | `60s` | Slowest observed model was 7.5s; wide margin |
| Retry attempts | `3` | |
| Backoff | `2s → 4s → 8s` | Exponential; doubling lets congestion clear |
| `Retry-After` | Honoured, capped at 30s | Server's number beats our guess (z-ai sent 5s) |

**Retry policy**
- `429, 500, 502, 503, 504` → retryable (busy / hiccup)
- `400, 401, 403, 404` → fatal, never retry (burns quota on the impossible)

**Content gate** — HTTP 200 is not a usable answer. A reply counts only if
`finish_reason == "stop"` **and** content is non-empty.
Rejecting truncated output is a policy choice: a truncated JSON scorecard is
worthless, and a truncated prose answer would be scored as if complete —
punishing a model for a limit we set.

---

## Quota tracking

~200 requests/day free-tier ceiling. Roughly **35 used on 2026-08-30** during
model selection.

---

## Open questions
- [ ] Does `dots-studio` need more than 1000 tokens for a real answer (not "ok")?
- [ ] Do judges need a higher `max_tokens` than generators?
- [ ] `nvidia/nemotron-3.5-content-safety:free` exists — a purpose-built safety
      classifier. We chose a rule-based pre-gate instead (deterministic, free,
      not prompt-injectable). Worth a "with another day" line in the README.

---

## Phase 2 findings — first live council run (2026-08-30)

Question: *"What year was the Eiffel Tower completed, and who designed it?"*
3/3 usable answers, 7.4s of model latency.

### Nobody cited anything
The generator system prompt asks for inline URLs on factual claims. All three
made confident factual claims. **Zero URLs produced.**

**The tension (for the README):** the harder you demand citations, the more
*fabricated* citations you get — trap #3 on demand. That is acceptable here only
*because* we verify: a made-up URL gets labelled `failed` and lowers confidence.
Demanding citations is dangerous only in a system that passes them through unchecked.

**Decision:** keep the soft instruction. Let eval question #5 explicitly ask for a
source, so the citation path is exercised without spamming fake URLs into every run.

### Agreement without a distinguishable cause
- nvidia: "designed by the French engineer Gustave Eiffel"
- inclusionai: Eiffel's *firm*, plus Koechlin, Nouguier, Sauvestre
- dots-studio: Eiffel, plus Koechlin, Nouguier, Sauvestre

Two of three converged on nearly identical detail. **Is that because it is true, or
because they trained on the same Wikipedia article?** The output is identical under
both explanations. This is the concrete argument for weighting `agent_agreement`
LOW in the confidence formula: agreement is real evidence, but weak, and you can
never tell which kind you have.

Note this was a *completeness* difference, not a factual disagreement — which is a
good argument for "completeness" as a rubric criterion.

### Prompt compliance
All three obeyed the 250-word limit. None self-rated confidence (the explicit
"do NOT rate your own confidence" line held). Token spend: 230 / 295 / 468.

---

## The judging rubric (locked 2026-08-30)

Scored **0–5 as integers**, per criterion, per answer.

| Criterion | Weight | What the judge scores |
|---|---|---|
| accuracy | 0.40 | Are the factual claims correct? Does it avoid inventing facts, figures, names, sources? |
| calibration | 0.25 | Does its certainty match its evidence? Does it flag what it does not know? |
| completeness | 0.20 | Does it address every part of the question actually asked? |
| reasoning | 0.15 | Is the logic sound, do conclusions follow from what precedes them? |

**Anchors given to judges:** `0 = fails entirely` · `3 = adequate, no serious problems`
· `5 = excellent, nothing you would change`

### Why these weights
- **Accuracy 0.40** — a wrong answer is worthless regardless of everything else.
  Largest share, but deliberately not a majority: accuracy alone is what a naive
  system optimises for, and it is not sufficient here.
- **Calibration 0.25** — the counterweight to fluency. Without it, a confidently
  wrong answer beats "the sources disagree, but most say X". At a quarter of the
  total, **an honest hedge can outscore a confident fabrication.** It is also what
  gives generators a reason to ever use `INSUFFICIENT_INFORMATION` — if abstention
  scored zero everywhere, no rational model would abstain.
- **Completeness 0.20** — the Eiffel run is the argument: all three answers were
  accurate and differed almost entirely in completeness. A rubric blind to that
  would have scored a three-way tie on a question with a genuine best answer.
- **Reasoning 0.15** — lowest, because short factual questions contain little
  reasoning to assess. Earns its place on ambiguous and unanswerable questions.

### Deliberately EXCLUDED (README-worthy)
**No clarity. No writing quality. No helpfulness or tone.**

> We do not score writing quality, because a council that rewards eloquence
> systematically prefers the most confident-sounding answer — the exact failure
> we are built to prevent.

### Why 0–5 integers, not 0–100
A 0–100 scale invites fake precision: a model will emit 87 vs 84 and we would read
a 3-point gap as meaningful when the model cannot reproduce either number. 0–5 with
written anchors is closer to what a model can actually discriminate. It makes ties
more likely — which is fine, because ties resolve to an explicit policy rather than
a manufactured winner. **A coarse scale that is honest beats a fine one that is invented.**

### Judges do NOT see the weights
Judges score each criterion raw; our code applies the weights. Handing a judge the
weights invites it to pre-aggregate and bias its own scores. **The judge scores;
the aggregator weights.** Keeps the two decisions separable and auditable.

### Answers are anonymised
Candidates are relabelled A / B / C before judging. A judge never learns which model
wrote what — it cannot favour a sibling's style or a name it finds impressive.
Same principle as `ask_generator` having no third parameter: remove the opportunity
rather than trusting good behaviour.

### Optional, not yet built
Ask each judge for an overall ranking and check it against the ranking its own
scores imply. Disagreement = sloppy scoring, a cheap integrity signal.
Good "with another day" line if unbuilt.

---

## A tenth trap we found ourselves: the non-discriminating judge (2026-08-30)

First full run with judges. `cohere/north-mini-code` returned a perfectly valid
scorecard — and gave **every answer 5/5 on every criterion**:

```
cohere:  [A] 5 5 5 5 -> 5.00     minimax: [A] 5 4 4 5 -> 4.55
         [B] 5 5 5 5 -> 5.00              [B] 5 5 5 5 -> 5.00
         [C] 5 5 5 5 -> 5.00              [C] 5 4 5 5 -> 4.75
```

**Why this is more dangerous than a failed judge.** A failed judge is visibly
absent — you see FAILED, you know you have one judge, you cap confidence.
A judge that scores everything 5 *looks like it is working*: valid JSON, every
field filled, every check passed. And numerically it sits within 0.45 / 0.00 /
0.25 of the other judge — so `inter_judge_agreement` would read it as **high
agreement** and push confidence UP, when the judge expressed no opinion at all.

> **A judge with zero variance agrees with everybody. That is not agreement,
> it is abstention wearing agreement's clothes.**

Not one of the brief's nine traps. Found by running the system, not by reading.

### Fix (build in Phase 4)
Compute the spread of each judge's weighted totals across candidates. If the
spread is ~0, flag the judge `non_discriminating`: keep its scores in provenance
for the record, exclude it from the agreement signal, and treat the run as
effectively single-judge (with the single-judge cap).

### The subtlety — do NOT "fix" this in the prompt
Tempting to add *"do not give the same score to every answer"* to the judge
prompt. **No.** If a judge genuinely thinks three answers are equally good,
forcing a spread manufactures a signal that does not exist — and that invented
spread would flow straight into `score_margin`.

**Detect the absence of signal; never fabricate the presence of one.**
Same error as defaulting a missing score to 3, just harder to see.

### Fair defence for cohere
On this question all three answers *are* correct; they differ only in
completeness. "These are all fine" is not crazy. Harder to defend: it gave
`completeness = 5` to answer A, which plainly omits the contributors B and C
both name. One question is not enough to condemn it — retest on a question
where answer quality genuinely differs before swapping to `poolside`.

### Hypothesis confirmed
Both judges: `parsed=direct`, `repairs=0`. The code-tuned-judge bet (reliable
structured output) held. Cohere's problem is content, not format.

---

## The single-judge problem (design note for Phase 4)

With one usable judge, `inter_judge_agreement` is **undefined** — not 0, not 1.
Three distinct situations must stay distinguishable:

| Situation | Meaning |
|---|---|
| Two judges, agree | Strong evidence |
| Two judges, disagree | Weak evidence — real information, points at ambiguity |
| One judge | **No evidence either way** — there was no cross-check |

Scoring the third as `agreement = 1.0` would make a single unchecked opinion
produce the highest confidence in the system. Confidence theater with extra steps.

**Handling = two moves together:** (1) drop the term and renormalise the other
weights, and (2) cap confidence at ~0.5. Renormalising alone lets the remaining
signals push high; capping alone leaves a fabricated number in the average.

---

## Judges cannot verify citations (2026-08-30)

`dots-studio` produced `(Source: https://www.toureiffel.paris/en/history)`.
`minimax` scored it and wrote: *"cites a **plausible** source."*

A judge cannot open a URL. It can only assess whether a link *looks* right.
That gap is exactly what the citation verifier exists to fill — trap #3, named
by our own judge, in one word.

## Non-determinism, second measured instance
`dots-studio`, same question, `temperature: 0`:
- run 1: 647 tokens, no citation
- run 2: 857 tokens, includes a citation

Output *content* varies between runs, not just length.

## Quota: ~53 requests used on 2026-08-30.
