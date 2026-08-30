"""
The LLM Council. Generators answer independently, judges score them,
decide() runs the whole pipeline.

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
from aggregate import RUBRIC, SCORE_MIN, SCORE_MAX, aggregate, weighted_total
from config import (ABSTAIN_MARKER, CONTAMINATION_PHRASES, GENERATORS,
                    GENERATOR_MAX_TOKENS, GENERATOR_SYSTEM_PROMPT, JUDGES,
                    JUDGE_MAX_TOKENS, JUDGE_SYSTEM_PROMPT,
                    MAX_JUSTIFICATION_CHARS, PAUSE_BETWEEN_CALLS)
from config import snapshot as config_snapshot
from decision import build_decision, refused_decision
import citations as citations_module
import audit

try:
    import gates
    GATES_AVAILABLE = True
except ImportError:
    gates = None
    GATES_AVAILABLE = False


SAMPLES_DIR = Path("samples")
SCHEMA_PATH = Path("schema/decision.schema.json")


def validate_decision(decision):
    """Check our own output against our own contract before emitting it."""
    try:
        import jsonschema
    except ImportError:
        return None

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(decision)]


# Note there is no parameter for other candidates' answers: generator
# independence is enforced by this signature, not by remembering it.
def ask_generator(generator, question):
    """Ask ONE generator the ORIGINAL question."""
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
        "label": None,
    }

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


def label_candidates(candidates):
    """Give every usable answer a neutral label: A, B, C."""
    labelled = {}
    usable = [c for c in candidates if c["ok"] and not c["abstained"]]

    for index, candidate in enumerate(usable):
        label = chr(ord("A") + index)
        candidate["label"] = label
        labelled[label] = candidate

    return labelled


FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text):
    """Get a dict out of model output, trying increasingly desperate methods."""
    if not text:
        return None, "empty text"

    try:
        return json.loads(text), "direct"
    except ValueError:
        pass

    fenced = FENCE_PATTERN.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "unfenced"
        except ValueError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), "sliced"
        except ValueError:
            pass

    return None, "no parseable JSON object found"


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
    """Keep only what a judge is allowed to say, and report anything it smuggled in."""
    problems = []

    if not isinstance(raw, dict):
        return None, ["judge output was not a JSON object"]

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
            # Reject the whole scorecard rather than invent a missing score.
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None, problems + [
                    f"{label}.{criterion} is not a number: {value!r}"]
            if not SCORE_MIN <= value <= SCORE_MAX:
                return None, problems + [
                    f"{label}.{criterion} out of range: {value}"]
            clean_entry[criterion] = float(value)

        clean_scores[label] = clean_entry

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
    """Ask ONE judge to score the anonymised answers."""
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

    for repair in range(2):
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
            return result

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
    """Freeze a run to samples/<name>.json."""
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

    labelled = {c["label"]: c for c in run["candidates"] if c.get("label")}
    return run, labelled


def decide(question, run=None, write_audit=True, save_as=None):
    """The whole pipeline, start to finish. One question in, one Decision Object out."""
    if run is None:
        if not GATES_AVAILABLE:
            print("WARNING: gates.py not found - the pre-gate is NOT running. "
                  "Unsafe prompts will reach the models.", file=sys.stderr)
        else:
            allowed, rule, reason = gates.pre_gate(question)
            if not allowed:
                decision = refused_decision(question, reason, rule,
                                            config_snapshot())
                if write_audit:
                    decision["audit_ref"] = audit.append(decision)
                return decision, None, None, None

        run = run_council(question)
        if save_as:
            save_run(run, save_as)

    labelled = label_candidates(run["candidates"])

    checked = citations_module.verify_all(run["candidates"])

    aggregation = aggregate(run, citations=checked)

    if GATES_AVAILABLE and aggregation.get("winning_answer"):
        safe, rule, reason = gates.post_gate(aggregation["winning_answer"])
        if not safe:
            aggregation["status"] = "refused"
            aggregation["winning_answer"] = None
            aggregation["runner_up_answer"] = None
            aggregation["winner_label"] = None
            aggregation["confidence"]["score"] = 0.0
            aggregation["confidence"]["method"] = (
                "Refused at the post-gate: the winning answer tripped a safety "
                "rule. Confidence is 0.0 - the decision was withdrawn.")
            aggregation["risks"].append({
                "type": "safety", "severity": "high",
                "detail": f"{reason} (post-gate rule: {rule})",
            })

    if not GATES_AVAILABLE:
        aggregation["risks"].append({
            "type": "safety", "severity": "high",
            "detail": ("gates.py is not installed: neither the pre-gate nor the "
                       "post-gate ran for this decision."),
        })

    decision = build_decision(run["question"], run, aggregation,
                              config_snapshot(), citations=checked)

    if write_audit:
        decision["audit_ref"] = audit.append(decision)

    return decision, aggregation, labelled, run


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
        run, _ = load_run(args.offline)
        print(f"[offline] replaying samples/{args.offline}.json", file=sys.stderr)
        question = run["question"]
    elif args.question:
        run, question = None, args.question
    else:
        parser.error("give a question, or --offline NAME")

    decision, aggregation, labelled, run = decide(
        question, run=run,
        write_audit=not args.no_audit,
        save_as=args.save,
    )

    problems = validate_decision(decision)

    if args.json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    elif aggregation is None:
        print("=" * 74)
        print(f"QUESTION: {question}")
        print("=" * 74)
        print(f"\n  status     : {decision['status']}")
        for risk in decision["risks"]:
            print(f"  risk       : [{risk['severity']} {risk['type']}] "
                  f"{risk['detail']}")
        print("\n  Refused before any model was called. 0 API calls.")
        print(f"\n  decision_id: {decision['decision_id']}")
        if decision["audit_ref"]:
            print(f"  audit_ref  : {decision['audit_ref'][:30]}...")
    else:
        print_report(run["question"], run["candidates"], labelled,
                     run["judgements"])
        if decision["citations"]:
            print()
            print("=" * 74)
            print("CITATIONS")
            print("=" * 74)
            for citation in decision["citations"]:
                print(f"  [{citation['status']:<10}] [{citation['label']}] "
                      f"{citation['source'][:60]}")
                print(f"               {citation['detail']}")
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
