"""
Run the eval set and save a report.

Run from the repo root:  python evals/run_evals.py [--offline]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

import gates
from council import decide, load_run

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
REPORT_PATH = Path(__file__).with_name("report.json")

OFFLINE_SAMPLES = {
    "factual": "easy",
    "ambiguous": "contested",
    "unknowable": "unknowable",
}


def load_questions():
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_expectation(case, decision):
    """Did the system behave acceptably? Returns (matched, why_not)."""
    expected = case["expect"]
    acceptable = expected if isinstance(expected, list) else [expected]

    if decision["status"] not in acceptable:
        return False, f"status {decision['status']} not in {acceptable}"

    required = case.get("require_risk")
    if required:
        types = {risk["type"] for risk in decision["risks"]}
        if required not in types:
            return False, (f"status ok, but no '{required}' risk was recorded "
                           f"(got {sorted(types) or 'none'})")

    needed = case.get("require_abstentions")
    if needed:
        abstained = sum(1 for g in decision["provenance"]["generators"]
                        if g["abstained"])
        if abstained < needed:
            return False, (f"status ok, but only {abstained} generator(s) "
                           f"abstained (need {needed}). The council did not "
                           "decline - something else went wrong.")

    return True, ""


def summarise(case, decision):
    """One row of the report. Everything a reviewer needs to retrace it."""
    confidence = decision["confidence"]
    citations = decision.get("citations", [])
    matched, why_not = check_expectation(case, decision)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"].strip(),
        "expected": case["expect"],
        "require_risk": case.get("require_risk"),
        "actual": decision["status"],
        "match": matched,
        "mismatch_reason": why_not,
        "confidence": confidence["score"],
        "signals": confidence["signals"],
        "caps_triggered": confidence["caps_triggered"],
        "risks": [f"[{r['severity']} {r['type']}] {r['detail']}"
                  for r in decision["risks"]],
        "citations": [{"status": c["status"], "source": c["source"]}
                      for c in citations],
        "citation_summary": {
            "verified": sum(1 for c in citations if c["status"] == "verified"),
            "unverified": sum(1 for c in citations if c["status"] == "unverified"),
            "failed": sum(1 for c in citations if c["status"] == "failed"),
        },
        "generators_usable": sum(
            1 for g in decision["provenance"]["generators"]
            if g["ok"] and not g["abstained"]),
        "generators_abstained": sum(
            1 for g in decision["provenance"]["generators"] if g["abstained"]),
        "generators_failed": sum(
            1 for g in decision["provenance"]["generators"] if not g["ok"]),
        "judges_ok": sum(1 for j in decision["provenance"]["judges"] if j["ok"]),
        "cost_usd": decision["provenance"]["cost_usd"],
        "latency_ms": decision["provenance"]["total_latency_ms"],
        "decision_id": decision["decision_id"],
        "audit_ref": decision["audit_ref"],
        "winning_answer": decision["winning_answer"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the LLM Council eval set.")
    parser.add_argument("--offline", action="store_true",
                        help="replay frozen samples where available; 0 API calls")
    parser.add_argument("--no-audit", action="store_true",
                        help="do not write these decisions to the audit chain")
    args = parser.parse_args()

    cases = load_questions()
    rows = []

    for case in cases:
        question = case["question"].strip()
        print(f"\n=== {case['id']} ({case['category']})", file=sys.stderr)

        run = None
        if args.offline:
            sample = OFFLINE_SAMPLES.get(case["id"])
            if sample:
                run, _ = load_run(sample)
                question = run["question"]
            elif not gates.pre_gate(question)[0]:
                pass
            else:
                print(f"    skipped - no frozen sample for '{case['id']}'; "
                      "running it would make live calls", file=sys.stderr)
                continue

        decision, _, _, _ = decide(question, run=run,
                                   write_audit=not args.no_audit)
        row = summarise(case, decision)
        rows.append(row)

        mark = "ok  " if row["match"] else "MISS"
        expected = row["expected"]
        shown = "|".join(expected) if isinstance(expected, list) else expected
        print(f"    {mark} expected={shown:<22} "
              f"actual={row['actual']:<12} conf={row['confidence']:.3f}",
              file=sys.stderr)
        if not row["match"]:
            print(f"         {row['mismatch_reason']}", file=sys.stderr)

    matched = sum(1 for row in rows if row["match"])

    starved = [row["id"] for row in rows
               if row["generators_failed"] and not row["generators_usable"]
               and row["actual"] != "refused"]

    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "offline" if args.offline else "live",
        "degraded": bool(starved),
        "degraded_cases": starved,
        "degraded_note": (
            "Every generator call failed on these cases (free-tier rate limit). "
            "Their results measure provider availability, not council logic. "
            "Note the system still emitted a valid Decision Object for each and "
            "declined rather than fabricating - but this report should not be "
            "read as a measurement of the council." if starved else ""),
        "cases_run": len(rows),
        "cases_expected": len(cases),
        "matched_expectation": matched,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "total_latency_ms": sum(r["latency_ms"] for r in rows),
        "results": rows,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print("\n" + "=" * 70)
    for row in rows:
        mark = "ok  " if row["match"] else "MISS"
        citation_note = ""
        if row["citations"]:
            summary = row["citation_summary"]
            citation_note = (f"  cites: {summary['verified']}v "
                             f"{summary['unverified']}u {summary['failed']}f")
        print(f"  {mark} {row['id']:<12} {row['actual']:<12} "
              f"conf={row['confidence']:.3f}{citation_note}")

    print("=" * 70)
    print(f"  {matched}/{len(rows)} matched expectation | "
          f"${report['total_cost_usd']} | {report['total_latency_ms']} ms")
    if starved:
        print(f"  DEGRADED RUN - every generator call failed on: "
              f"{', '.join(starved)}")
        print("  These results measure provider availability, not the council. "
              "Re-run when the rate limit clears.")
    print(f"  report -> {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
