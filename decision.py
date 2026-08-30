"""
Build the Decision Object. Every run produces one, including refusals.
"""

import datetime
import hashlib
import json
import uuid

SCHEMA_VERSION = "1.0.0"


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def config_hash(snapshot):
    """Fingerprint the exact configuration used for this run."""
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_decision(question, run, aggregation, config_snapshot,
                   citations=None, audit_ref=None):
    """Assemble the Decision Object from what actually happened."""
    citations = citations or []
    candidates = run.get("candidates", [])
    judgements = run.get("judgements", []) or []

    generators = [
        {
            "model_id": c["model_id"],
            "family": c["family"],
            "served_by": c.get("served_by"),
            "label": c.get("label"),
            "ok": c["ok"],
            "abstained": c["abstained"],
            "failure_stage": c.get("failure_stage"),
            "detail": c.get("detail", ""),
            "latency_ms": c.get("latency_ms", 0),
            "tokens": c.get("tokens", 0),
            "attempts": c.get("attempts", 0),
            "cost_usd": c.get("cost_usd", 0.0),
        }
        for c in candidates
    ]

    judges = [
        {
            "model_id": j["model_id"],
            "family": j["family"],
            "ok": j["ok"],
            "detail": j.get("detail", ""),
            "parse_method": j.get("parse_method"),
            "repair_calls": j.get("repair_calls", 0),
            "problems": j.get("problems", []),
            "latency_ms": j.get("latency_ms", 0),
            "tokens": j.get("tokens", 0),
            "cost_usd": j.get("cost_usd", 0.0),
            "rubric_scores": j.get("scores") or {},
            "justification": j.get("justification") or {},
        }
        for j in judgements
    ]

    total_latency = (sum(g["latency_ms"] for g in generators)
                     + sum(j["latency_ms"] for j in judges))
    total_cost = (sum(g["cost_usd"] for g in generators)
                  + sum(j["cost_usd"] for j in judges))

    confidence = aggregation["confidence"]

    return {
        "decision_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "status": aggregation["status"],
        "question": question,
        "winning_answer": aggregation.get("winning_answer"),
        "runner_up_answer": aggregation.get("runner_up_answer"),
        "confidence": {
            "score": confidence["score"],
            "method": confidence["method"],
            "signals": confidence["signals"],
            "raw_score": confidence.get("raw_score", 0.0),
            "contributions": confidence.get("contributions", {}),
            "caps_triggered": confidence.get("caps_triggered", []),
        },
        "risks": aggregation.get("risks", []),
        "citations": citations,
        "provenance": {
            "generators": generators,
            "judges": judges,
            "config_hash": config_hash(config_snapshot),
            "cost_usd": round(total_cost, 6),
            "total_latency_ms": total_latency,
            "workings": aggregation.get("workings", {}),
        },
        "audit_ref": audit_ref,
    }


def refused_decision(question, reason, rule, config_snapshot, audit_ref=None):
    """A pre-gate refusal. Zero API calls were made."""
    return {
        "decision_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "status": "refused",
        "question": question,
        "winning_answer": None,
        "runner_up_answer": None,
        "confidence": {
            "score": 0.0,
            "method": ("Refused at the pre-gate before any model was called. "
                       "Confidence is 0.0 by definition - no council ran, so "
                       "there is no decision to be confident about."),
            "signals": {
                "inter_judge_agreement": None,
                "score_margin": None,
                "winner_quality": None,
                "agent_agreement": None,
                "verification_pass_rate": None,
            },
            "raw_score": 0.0,
            "contributions": {},
            "caps_triggered": [],
        },
        "risks": [{"type": "safety", "severity": "high",
                   "detail": f"{reason} (pre-gate rule: {rule})"}],
        "citations": [],
        "provenance": {
            "generators": [],
            "judges": [],
            "config_hash": config_hash(config_snapshot),
            "cost_usd": 0.0,
            "total_latency_ms": 0,
            "workings": {"refused_by": rule},
        },
        "audit_ref": audit_ref,
    }
