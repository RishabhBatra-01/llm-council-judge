"""
run_evals.py - run the eval set and save a report.

Run from the repo root:

    python evals/run_evals.py              # live, ~20 API calls
    python evals/run_evals.py --offline    # replay samples/, 0 API calls

Costs roughly 20 calls, not 25: the unsafe question is refused at the pre-gate
before any model is contacted.

What this measures is NOT "did the council get the right answer". For three of
the five questions the correct behaviour is declining. It measures whether the
system knows when it should not answer.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

# Run from anywhere: put the repo root on the import path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml                                                   # noqa: E402

from council import decide, load_run                          # noqa: E402

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
REPORT_PATH = Path(__file__).with_name("report.json")

# Offline replay uses these frozen runs where one exists. Questions without a
# sample are skipped in offline mode rather than quietly run live - a report
# labelled "offline" that made network calls would be a lie about its own cost.
OFFLINE_SAMPLES = {
    "factual": "easy",
    "ambiguous": "contested",
    "unknowable": "unknowable",
}


def load_questions():
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_expectation(case, decision):
    """
    Did the system behave acceptably? Returns (matched, why_not).

    `expect` may be a single status or a list of acceptable ones. `require_risk`
    additionally demands a risk of that type - used where the brief allows more
    than one correct outcome but only if the reason is surfaced. On the
    ambiguous case, deciding is fine; deciding SILENTLY is not.
    """
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
            if not sample:
                print(f"    skipped - no frozen sample for '{case['id']}'; "
                      "running it would make live calls", file=sys.stderr)
                continue
            run, _ = load_run(sample)
            question = run["question"]

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

    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "offline" if args.offline else "live",
        "cases_run": len(rows),
        "cases_expected": len(cases),
        "matched_expectation": matched,
        # Deliberately not called a pass rate. A mismatch is a prompt to read
        # the Decision Object and decide whether the system or the expectation
        # was wrong - not automatically a failure.
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
    print(f"  report -> {REPORT_PATH}")

    # Exit 0 either way. A mismatch is information, not a build failure - and
    # a harness that exits non-zero invites someone to "fix" it by editing the
    # expectation until it passes.
    return 0


if __name__ == "__main__":
    sys.exit(main())
