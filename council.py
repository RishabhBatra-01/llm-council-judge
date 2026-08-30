"""
council.py - the LLM Council.

Phase 2: three generators answer independently, two judges score them.
Aggregation, gates, citations and audit come later.

Run:  python council.py "your question here"
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

from client import call_model
from aggregate import (CAPS, MARGIN_FULL_SCALE, NO_DECISION_THRESHOLD,
                       RUBRIC, SCORE_MIN, SCORE_MAX, SIGNAL_WEIGHTS,
                       TIE_EPSILON, DISCRIMINATION_THRESHOLD,
                       aggregate, weighted_total)
from decision import build_decision
import audit


# ===========================================================================
# CONFIG  (moves to config.yaml in Phase 6)
# ===========================================================================

# Three distinct families. Same model three times is not a council -
# it is one model with correlated errors and a louder voice.
GENERATORS = [
    {"model_id": "nvidia/nemotron-3-super-120b-a12b:free", "family": "nvidia"},
    {"model_id": "inclusionai/ling-3.0-flash-fin:free",    "family": "inclusionai"},
    {"model_id": "dots-studio/dots-3-note-preview:free",   "family": "dots-studio"},
]

# Neither judge shares a family with any generator.
JUDGES = [
    {"model_id": "minimax/minimax-m3:free",      "family": "minimax"},
    {"model_id": "cohere/north-mini-code:free",  "family": "cohere"},
]

GENERATOR_MAX_TOKENS = 3000      # 1500 starved dots-studio on a hard question:
                                 # all 1500 spent reasoning, 0 chars produced
JUDGE_MAX_TOKENS = 6000          # judges read 3 answers and emit JSON.
                                 # 2000 truncated cohere: it spent ~1900
                                 # tokens reasoning and never reached the JSON.

PAUSE_BETWEEN_CALLS = 2.0        # ~20 requests/min free-tier ceiling

SAMPLES_DIR = Path("samples")
SCHEMA_PATH = Path("schema/decision.schema.json")


def config_snapshot():
    """
    Every setting that could change a decision, in one dict.

    This is what config_hash fingerprints. If a run's decision would differ
    under different settings, the setting belongs in here - otherwise the hash
    would claim two runs were comparable when they were not.
    """
    return {
        "generators": GENERATORS,
        "judges": JUDGES,
        "rubric": RUBRIC,
        "score_range": [SCORE_MIN, SCORE_MAX],
        "signal_weights": SIGNAL_WEIGHTS,
        "caps": CAPS,
        "thresholds": {
            "no_decision": NO_DECISION_THRESHOLD,
            "tie_epsilon": TIE_EPSILON,
            "discrimination": DISCRIMINATION_THRESHOLD,
            "margin_full_scale": MARGIN_FULL_SCALE,
        },
        "generation": {
            "generator_max_tokens": GENERATOR_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "temperature": 0,
        },
        "prompts": {
            "generator": GENERATOR_SYSTEM_PROMPT,
            "judge": JUDGE_SYSTEM_PROMPT,
        },
    }


def validate_decision(decision):
    """
    Check our own output against our own contract before emitting it.

    A schema we ship but never run against ourselves is documentation, not a
    guarantee. Returns a list of problems; empty means valid.
    """
    try:
        import jsonschema
    except ImportError:
        # "We could not check" is not "we checked and it failed". Conflating
        # them would be the same error this whole system exists to avoid, so
        # it is reported separately and does not fail the run.
        return None

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(decision)]

ABSTAIN_MARKER = "INSUFFICIENT_INFORMATION"

MAX_JUSTIFICATION_CHARS = 300

# A judge that starts answering the question has stopped being a judge.
CONTAMINATION_PHRASES = [
    "the correct answer is",
    "the answer is",
    "here is my answer",
    "my answer:",
    "i would answer",
    "the right answer",
]


GENERATOR_SYSTEM_PROMPT = f"""You are one member of an independent expert panel.
Answer the question directly and completely.

Rules:
- Answer only from what you actually know. Never invent facts, figures, or sources.
- If you cannot answer reliably, reply with exactly {ABSTAIN_MARKER} on the first
  line, then one sentence saying what is missing.
- If a factual claim rests on a source, put the full URL inline in that sentence.
- Do NOT rate your own confidence. No percentages, no "I am 90% sure".
- Keep your answer under 250 words."""


JUDGE_SYSTEM_PROMPT = """You are a judge on an evaluation panel.
Your ONLY job is to score candidate answers against a rubric.

Absolute rules:
- You must NOT write your own answer to the question, not even partially.
- You must NOT rewrite, improve, or correct any candidate answer.
- You score only what is written in front of you.
- Return ONLY a single JSON object. No markdown fences, no commentary."""


# ===========================================================================
# GENERATORS
# ===========================================================================

def ask_generator(generator, question):
    """
    Ask ONE generator the ORIGINAL question.

    Note what this function cannot do: there is no parameter for other
    candidates' answers, so no caller can leak one generator's output into
    another's input. The independence rule is enforced by the shape of this
    function, not by remembering to obey it.
    """
    reply = call_model(
        generator["model_id"],
        question,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        max_tokens=GENERATOR_MAX_TOKENS,
    )

    candidate = {
        "model_id": generator["model_id"],
        "family": generator["family"],
        "ok": reply["ok"],
        "detail": reply["detail"],
        "failure_stage": reply["failure_stage"],
        "latency_ms": reply["latency_ms"],
        "attempts": reply["attempts"],
        "tokens": reply["tokens"],
        "cost_usd": reply["cost_usd"],
        "served_by": reply["served_by"],
        "answer": reply["text"],
        "abstained": False,
        "label": None,          # set by label_candidates(); survives save/load
    }

    # A generator saying "I don't know" succeeded. That is a signal, not a failure.
    if candidate["ok"] and candidate["answer"].startswith(ABSTAIN_MARKER):
        candidate["abstained"] = True

    return candidate


def gather_candidates(question):
    """Run every generator, one at a time, and collect what comes back."""
    candidates = []

    for index, generator in enumerate(GENERATORS):
        print(f"  [gen {index + 1}/{len(GENERATORS)}] {generator['model_id']}",
              file=sys.stderr)
        candidates.append(ask_generator(generator, question))

        if index < len(GENERATORS) - 1:
            time.sleep(PAUSE_BETWEEN_CALLS)

    return candidates


# ===========================================================================
# ANONYMISATION
# ===========================================================================

def label_candidates(candidates):
    """
    Give every usable answer a neutral label: A, B, C.

    Judges see labels and text only. They never learn which model wrote what,
    so they cannot favour a sibling's style or a name they find impressive.
    The mapping stays here, in our code.

    The label is stored ON the candidate, not in a side table keyed by object
    identity - identity does not survive being written to disk and read back.
    """
    labelled = {}
    usable = [c for c in candidates if c["ok"] and not c["abstained"]]

    for index, candidate in enumerate(usable):
        label = chr(ord("A") + index)
        candidate["label"] = label
        labelled[label] = candidate

    return labelled


# ===========================================================================
# JSON REPAIR  (trap #6)
# ===========================================================================

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text):
    """
    Get a dict out of model output, trying increasingly desperate methods.

    Returns (dict, method_used) or (None, reason_it_failed).
    """
    if not text:
        return None, "empty text"

    # 1. It is already clean JSON.
    try:
        return json.loads(text), "direct"
    except ValueError:
        pass

    # 2. It is wrapped in a ```json fence.
    fenced = FENCE_PATTERN.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "unfenced"
        except ValueError:
            pass

    # 3. There is prose around it. Take the outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), "sliced"
        except ValueError:
            pass

    return None, "no parseable JSON object found"


# ===========================================================================
# JUDGES
# ===========================================================================

def build_judge_prompt(question, labelled):
    """Question + anonymised answers + rubric. No weights, no model names."""
    answers = "\n\n".join(
        f"[{label}]\n{candidate['answer']}"
        for label, candidate in sorted(labelled.items())
    )

    criteria_lines = "\n".join(f"- {name}" for name in RUBRIC)

    example_scores = {
        label: {name: 0 for name in RUBRIC} for label in sorted(labelled)
    }
    example_justifications = {
        label: "under 300 characters" for label in sorted(labelled)
    }
    shape = json.dumps(
        {"scores": example_scores, "justification": example_justifications},
        indent=2,
    )

    return f"""QUESTION:
{question}

CANDIDATE ANSWERS:

{answers}

RUBRIC - score every answer on every criterion, integers {SCORE_MIN} to {SCORE_MAX}:
{criteria_lines}

  accuracy      = are the factual claims correct? does it avoid invented facts,
                  figures, names or sources?
  calibration   = does its certainty match its evidence? does it flag what it
                  does not know, and separate fact from inference?
  completeness  = does it address every part of the question actually asked?
  reasoning     = is the logic sound, do conclusions follow from what precedes?

Anchors:
  0 = fails this criterion entirely
  3 = adequate, no serious problems
  5 = excellent, nothing you would change

Return exactly this JSON shape and nothing else:
{shape}"""


def sanitise_judgement(raw, labels):
    """
    Keep only what a judge is allowed to say, and report anything it smuggled in.

    Returns (clean_dict_or_None, problems_list).
    A missing score is NOT filled in with a default - inventing a score to
    paper over a gap is exactly the kind of quiet lie this project exists
    to avoid. A judge with holes in its scorecard has failed.
    """
    problems = []

    if not isinstance(raw, dict):
        return None, ["judge output was not a JSON object"]

    # --- key allowlist: a judge has nowhere to put a smuggled answer --------
    extra_keys = set(raw) - {"scores", "justification"}
    if extra_keys:
        problems.append(f"dropped unexpected keys: {sorted(extra_keys)}")

    raw_scores = raw.get("scores")
    if not isinstance(raw_scores, dict):
        return None, problems + ["missing or malformed 'scores'"]

    clean_scores = {}
    for label in labels:
        entry = raw_scores.get(label)
        if not isinstance(entry, dict):
            return None, problems + [f"no scores for candidate {label}"]

        clean_entry = {}
        for criterion in RUBRIC:
            value = entry.get(criterion)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None, problems + [
                    f"{label}.{criterion} is not a number: {value!r}"]
            if not SCORE_MIN <= value <= SCORE_MAX:
                return None, problems + [
                    f"{label}.{criterion} out of range: {value}"]
            clean_entry[criterion] = float(value)

        clean_scores[label] = clean_entry

    # --- justifications: truncated, and checked for smuggled answers -------
    raw_justifications = raw.get("justification") or {}
    clean_justifications = {}

    for label in labels:
        text = raw_justifications.get(label, "")
        if not isinstance(text, str):
            text = ""

        lowered = text.lower()
        for phrase in CONTAMINATION_PHRASES:
            if phrase in lowered:
                problems.append(
                    f"contamination: {label} justification contains {phrase!r}")
                break

        if len(text) > MAX_JUSTIFICATION_CHARS:
            problems.append(
                f"{label} justification truncated "
                f"({len(text)} -> {MAX_JUSTIFICATION_CHARS} chars)")
            text = text[:MAX_JUSTIFICATION_CHARS]

        clean_justifications[label] = text

    return {"scores": clean_scores, "justification": clean_justifications}, problems


def ask_judge(judge, question, labelled):
    """
    Ask ONE judge to score the anonymised answers.

    On unparseable JSON we retry once with a blunter instruction, then give up.
    Giving up cleanly is a feature: a judge we could not read is a judge that
    failed, and confidence must fall accordingly.
    """
    prompt = build_judge_prompt(question, labelled)
    labels = sorted(labelled)

    result = {
        "model_id": judge["model_id"],
        "family": judge["family"],
        "ok": False,
        "detail": "",
        "problems": [],
        "parse_method": None,
        "scores": None,
        "justification": None,
        "latency_ms": 0,
        "attempts": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "repair_calls": 0,
    }

    for repair in range(2):          # first go, then one repair attempt
        suffix = "" if repair == 0 else (
            "\n\nYour previous reply was not valid JSON. "
            "Return ONLY the JSON object. No prose, no fences.")

        reply = call_model(
            judge["model_id"],
            prompt + suffix,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=JUDGE_MAX_TOKENS,
        )

        result["latency_ms"] += reply["latency_ms"]
        result["attempts"] += reply["attempts"]
        result["tokens"] += reply["tokens"]
        result["cost_usd"] += reply["cost_usd"]
        result["repair_calls"] = repair

        if not reply["ok"]:
            result["detail"] = reply["detail"]
            return result            # a network/content failure is not repairable

        parsed, method = extract_json_object(reply["text"])
        if parsed is None:
            result["detail"] = f"unparseable JSON ({method})"
            if repair == 0:
                print(f"      judge JSON unreadable, retrying once",
                      file=sys.stderr)
                time.sleep(PAUSE_BETWEEN_CALLS)
                continue
            return result

        clean, problems = sanitise_judgement(parsed, labels)
        result["problems"] = problems
        result["parse_method"] = method

        if clean is None:
            result["detail"] = "; ".join(problems) or "unusable scorecard"
            return result

        result["ok"] = True
        result["scores"] = clean["scores"]
        result["justification"] = clean["justification"]
        return result

    return result


def gather_judgements(question, labelled):
    """Run every judge over the same anonymised answers."""
    judgements = []

    for index, judge in enumerate(JUDGES):
        print(f"  [judge {index + 1}/{len(JUDGES)}] {judge['model_id']}",
              file=sys.stderr)
        judgements.append(ask_judge(judge, question, labelled))

        if index < len(JUDGES) - 1:
            time.sleep(PAUSE_BETWEEN_CALLS)

    return judgements


# ===========================================================================
# REPORTING  (a real Decision Object replaces this in Phase 4)
# ===========================================================================

def print_report(question, candidates, labelled, judgements):
    print("=" * 74)
    print(f"QUESTION: {question}")
    print("=" * 74)

    for candidate in candidates:
        label = candidate.get("label") or "-"
        print()
        print(f"--- [{label}] {candidate['family']} / {candidate['model_id']}")

        if not candidate["ok"]:
            print(f"    FAILED [{candidate['failure_stage']}] {candidate['detail']}")
            continue

        status = "ABSTAINED" if candidate["abstained"] else "answered"
        print(f"    {status} | {candidate['latency_ms']} ms | "
              f"{candidate['tokens']} tokens")
        print()
        print(candidate["answer"])

    print()
    print("=" * 74)
    print("JUDGING")
    print("=" * 74)

    if not judgements:
        print("  skipped - fewer than 2 usable answers to compare")
        return

    for judgement in judgements:
        print()
        print(f"--- {judgement['family']} / {judgement['model_id']}")

        if not judgement["ok"]:
            print(f"    FAILED: {judgement['detail']}")
            for problem in judgement["problems"]:
                print(f"      ! {problem}")
            continue

        print(f"    ok | {judgement['latency_ms']} ms | "
              f"parsed={judgement['parse_method']} | "
              f"repairs={judgement['repair_calls']}")

        for problem in judgement["problems"]:
            print(f"      ! {problem}")

        for label in sorted(judgement["scores"]):
            row = judgement["scores"][label]
            detail = "  ".join(f"{name}={row[name]:.0f}" for name in RUBRIC)
            print(f"      [{label}] {detail}   weighted={weighted_total(row):.2f}")
            print(f"          {judgement['justification'][label]}")


def print_decision(decision):
    print()
    print("=" * 74)
    print("DECISION")
    print("=" * 74)

    confidence = decision["confidence"]
    print(f"  status     : {decision['status']}")
    print(f"  confidence : {confidence['score']:.3f}"
          f"   (raw blend {confidence['raw_score']:.3f})")

    print("  signals    :")
    for name, value in confidence["signals"].items():
        shown = "unavailable" if value is None else f"{value:.3f}"
        contribution = confidence["contributions"].get(name)
        extra = f"  -> contributes {contribution:.3f}" if contribution else ""
        print(f"      {name:<24} {shown}{extra}")

    if confidence["caps_triggered"]:
        print(f"  ceilings   : {', '.join(confidence['caps_triggered'])}")

    workings = decision["workings"]
    if workings.get("combined_totals"):
        totals = "  ".join(f"{k}={v:.2f}"
                           for k, v in sorted(workings["combined_totals"].items()))
        print(f"  combined   : {totals}")
    if workings.get("judge_picks"):
        picks = "  ".join(f"{k.split('/')[-1]}->{v}"
                          for k, v in workings["judge_picks"].items())
        print(f"  picks      : {picks}")
    if workings.get("non_discriminating_judges"):
        print(f"  no opinion : {', '.join(workings['non_discriminating_judges'])}")
    if workings.get("tie_broken_by"):
        print(f"  tie broken : {workings['tie_broken_by']}")

    if decision["risks"]:
        print("  risks      :")
        for risk in decision["risks"]:
            print(f"      [{risk['severity']:<4} {risk['type']}] {risk['detail']}")

    print()
    if decision["winning_answer"]:
        print(f"  WINNER: [{decision['winner_label']}]")
        print()
        print(decision["winning_answer"])
    else:
        print("  No winner. The council declined to decide.")


def run_council(question):
    """One live council run. Returns everything we learned, as plain data."""
    print(f"Asking {len(GENERATORS)} generators...", file=sys.stderr)
    candidates = gather_candidates(question)

    labelled = label_candidates(candidates)

    judgements = []
    if len(labelled) >= 2:
        print(f"Judging {len(labelled)} answers with {len(JUDGES)} judges...",
              file=sys.stderr)
        time.sleep(PAUSE_BETWEEN_CALLS)
        judgements = gather_judgements(question, labelled)
    else:
        print(f"Only {len(labelled)} usable answer(s) - nothing to compare.",
              file=sys.stderr)

    return {
        "question": question,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidates": candidates,
        "judgements": judgements,
    }


def save_run(run, name):
    """
    Freeze a run to samples/<name>.json.

    Why bother: the free tier allows ~200 calls/day and one run costs 5. Tuning
    the confidence formula takes dozens of iterations - far more than a day's
    quota. Saved runs let us build the aggregator offline, for free, instantly,
    against the exact same inputs every time (which also helps with trap #9).
    """
    SAMPLES_DIR.mkdir(exist_ok=True)
    path = SAMPLES_DIR / f"{name}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"saved -> {path}", file=sys.stderr)
    return path


def load_run(name):
    """Replay a frozen run. Makes no network calls at all."""
    path = SAMPLES_DIR / f"{name}.json"
    if not path.exists():
        print(f"no such sample: {path}", file=sys.stderr)
        sys.exit(1)

    run = json.loads(path.read_text())

    # Rebuild the label -> candidate map from labels stored on the candidates.
    labelled = {c["label"]: c for c in run["candidates"] if c.get("label")}
    return run, labelled


def main():
    parser = argparse.ArgumentParser(
        description="LLM Council - several models answer, two judges score.")
    parser.add_argument("question", nargs="?",
                        help="the question to put to the council")
    parser.add_argument("--save", metavar="NAME",
                        help="freeze this run to samples/NAME.json")
    parser.add_argument("--offline", metavar="NAME",
                        help="replay samples/NAME.json, making zero API calls")
    parser.add_argument("--json", action="store_true",
                        help="emit only the Decision Object as JSON")
    parser.add_argument("--no-audit", action="store_true",
                        help="do not write to the audit chain (for experiments)")
    args = parser.parse_args()

    if args.offline:
        run, labelled = load_run(args.offline)
        print(f"[offline] replaying samples/{args.offline}.json", file=sys.stderr)
    elif args.question:
        run = run_council(args.question)
        labelled = label_candidates(run["candidates"])
        if args.save:
            save_run(run, args.save)
    else:
        parser.error("give a question, or --offline NAME")

    aggregation = aggregate(run)
    decision = build_decision(run["question"], run, aggregation, config_snapshot())

    problems = validate_decision(decision)

    # Log BEFORE emitting. A decision that reached the user but not the log
    # would be a decision we cannot audit - which is a decision we do not ship.
    if not args.no_audit:
        decision["audit_ref"] = audit.append(decision)

    if args.json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    else:
        print_report(run["question"], run["candidates"], labelled,
                     run["judgements"])
        print_decision(aggregation)
        print()
        print(f"  decision_id: {decision['decision_id']}")
        print(f"  config_hash: {decision['provenance']['config_hash'][:23]}...")
        print(f"  cost_usd   : {decision['provenance']['cost_usd']}")
        print(f"  latency    : {decision['provenance']['total_latency_ms']} ms")
        if decision["audit_ref"]:
            print(f"  audit_ref  : {decision['audit_ref'][:30]}...")

    if problems is None:
        print("warning: jsonschema not installed - output NOT validated",
              file=sys.stderr)
    elif problems:
        print("\nSCHEMA VALIDATION FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
