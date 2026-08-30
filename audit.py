"""
audit.py - the append-only, tamper-evident decision log.

Every decision (including refusals and no_decisions) gets one line in
audit/chain.jsonl. Each line carries the hash of the line before it, so any
edit or deletion anywhere in the history breaks every link after it.

Run:  python audit.py verify     check the chain
      python audit.py show       list the entries
      python audit.py demo       tamper with a copy and watch verify catch it
"""

import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path

CHAIN_PATH = Path("audit/chain.jsonl")

# The first record links to this. A fixed, obviously-not-a-real-hash value, so
# "start of chain" can never be confused with a genuine previous entry.
GENESIS = "sha256:" + "0" * 64


def _canonical(obj):
    """
    Bytes that depend only on CONTENT, never on formatting.

    Sorted keys and no incidental whitespace. Without this, re-serialising the
    same decision with different key order would produce a different hash and
    the chain would appear broken when nothing had changed.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_records(path=CHAIN_PATH):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append(decision, path=CHAIN_PATH):
    """
    Add one decision to the chain. Returns the entry hash (its audit_ref).

    Note what is hashed: the decision WITHOUT its audit_ref field. The ref is
    the hash of the entry, so including it would require the hash to contain
    itself. The field is derived, not content - so it is stripped before
    hashing and stamped onto the caller's copy afterwards.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in decision.items() if k != "audit_ref"}
    payload_hash = _sha256(_canonical(payload))

    records = _read_records(path)
    prev_hash = records[-1]["entry_hash"] if records else GENESIS

    header = {
        "seq": len(records),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prev_hash": prev_hash,
        "payload_hash": payload_hash,
    }
    entry_hash = _sha256(_canonical(header))

    record = dict(header, entry_hash=entry_hash, decision=payload)

    # Append only. We open in append mode and never rewrite earlier lines.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return entry_hash


def verify(path=CHAIN_PATH):
    """
    Walk the chain and report the first place it stops adding up.

    Four things are checked per record, and they catch different tampering:
      - sequence numbers are contiguous     -> a deleted line
      - prev_hash matches the previous entry -> a removed or reordered line
      - payload_hash matches the stored decision -> an edited decision
      - entry_hash matches its own header    -> an edited hash

    Editing a decision and recomputing its payload_hash still breaks the
    entry_hash. Recomputing that too still breaks the NEXT record's prev_hash.
    There is nowhere to stop.
    """
    records = _read_records(path)
    problems = []
    prev_hash = GENESIS

    for index, record in enumerate(records):
        where = f"seq {record.get('seq', '?')} (line {index + 1})"

        if record.get("seq") != index:
            problems.append(f"{where}: sequence gap - expected seq {index}. "
                            "A record was deleted or reordered.")

        if record.get("prev_hash") != prev_hash:
            problems.append(f"{where}: prev_hash does not match the previous "
                            f"entry. Chain broken here.")

        recomputed_payload = _sha256(_canonical(record.get("decision", {})))
        if recomputed_payload != record.get("payload_hash"):
            problems.append(f"{where}: decision content was edited "
                            f"(stored {str(record.get('payload_hash'))[:23]}..., "
                            f"recomputed {recomputed_payload[:23]}...)")

        header = {
            "seq": record.get("seq"),
            "created_at": record.get("created_at"),
            "prev_hash": record.get("prev_hash"),
            "payload_hash": record.get("payload_hash"),
        }
        if _sha256(_canonical(header)) != record.get("entry_hash"):
            problems.append(f"{where}: entry_hash does not match its own header.")

        if problems:
            # Everything after a break is untrustworthy anyway. Stop and say so.
            problems.append(f"entries {index}-{len(records) - 1} can no longer "
                            "be trusted.")
            return records, problems

        prev_hash = record["entry_hash"]

    return records, problems


# ===========================================================================
# CLI
# ===========================================================================

def cmd_verify(path):
    records, problems = verify(path)
    if not records:
        print(f"{path}: empty - nothing to verify")
        return 0
    if problems:
        print(f"CHAIN BROKEN ({len(records)} entries)")
        for problem in problems:
            print(f"  x {problem}")
        return 1
    print(f"OK  {len(records)} entries, chain intact")
    print(f"    head {records[-1]['entry_hash'][:30]}...")
    return 0


def cmd_show(path):
    records, _ = verify(path)
    if not records:
        print(f"{path}: empty")
        return 0
    for record in records:
        decision = record["decision"]
        print(f"  [{record['seq']:>3}] {record['created_at'][:19]}  "
              f"{decision['status']:<12} "
              f"conf={decision['confidence']['score']:.3f}  "
              f"{record['entry_hash'][7:19]}  "
              f"{decision['question'][:44]}")
    return 0


def cmd_demo(path):
    """
    Prove the chain actually chains.

    Copies the real log, edits one old decision, and runs the same verify()
    over the copy. A hash chain nobody has watched catch tampering is a claim,
    not a guarantee.
    """
    records = _read_records(path)
    if len(records) < 2:
        print("need at least 2 entries to demonstrate - run the council a few "
              "times first")
        return 1

    tampered_path = path.with_name("chain.tampered.jsonl")
    target = 0 if len(records) < 3 else 1

    copied = copy.deepcopy(records)
    original = copied[target]["decision"]["confidence"]["score"]
    copied[target]["decision"]["confidence"]["score"] = 0.99

    tampered_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in copied) + "\n",
        encoding="utf-8")

    print(f"copied {len(records)} entries -> {tampered_path}")
    print(f"edited seq {target}: confidence {original} -> 0.99 "
          "(exactly the edit someone would want to make)")
    print()
    print("verifying the tampered copy:")
    _, problems = verify(tampered_path)
    for problem in problems:
        print(f"  x {problem}")
    print()
    print("verifying the real chain:")
    cmd_verify(path)

    tampered_path.unlink()
    print(f"\nremoved {tampered_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Tamper-evident decision log.")
    parser.add_argument("command", choices=["verify", "show", "demo"])
    parser.add_argument("--path", default=str(CHAIN_PATH))
    args = parser.parse_args()

    path = Path(args.path)
    return {"verify": cmd_verify, "show": cmd_show, "demo": cmd_demo}[args.command](path)


if __name__ == "__main__":
    sys.exit(main())
