"""
aggregate.py - turn judge scorecards into a winner and an EARNED confidence.

No model is consulted here. Every decision in this file is made by arithmetic
we wrote and can explain. That is the whole point: confidence must come from
things we observed, never from a model telling us how sure it feels.

Nothing here touches the network. It is pure functions over plain data, which
is why it can be developed and tested offline against saved runs.
"""

# ===========================================================================
# POLICY  (all of this moves to config.yaml in Step 6)
# ===========================================================================

RUBRIC = {
    "accuracy":     0.40,
    "calibration":  0.25,
    "completeness": 0.20,
    "reasoning":    0.15,
}

SCORE_MIN, SCORE_MAX = 0, 5

# How much each observable signal contributes, when it is available.
SIGNAL_WEIGHTS = {
    "inter_judge_agreement":  0.35,
    "score_margin":           0.20,
    # score_margin measures SEPARATION, not quality. Three candidates all
    # scoring 5.0 and three all scoring 2.0 both produce a margin near zero -
    # identical signal, opposite realities. Absolute winner quality is what
    # tells them apart, so it carries equal weight to the margin.
    "winner_quality":         0.20,
    # Deliberately LOW. Three models agreeing may be one wrong prior shared
    # three times (trap #2). Agreement is real evidence, but weak, and we can
    # never tell from the agreement itself which kind we have.
    "agent_agreement":        0.10,
    "verification_pass_rate": 0.15,
}

# A 1.0-point gap on the 0-5 weighted scale counts as a decisive margin.
# Judges in practice spread candidates by well under a point, so normalising
# by the full 0-5 range would make every margin look negligible.
MARGIN_FULL_SCALE = 1.0

# A judge whose weighted totals span less than this has expressed no preference.
DISCRIMINATION_THRESHOLD = 0.05

# Two candidates closer than this on the combined score are tied, not ranked.
TIE_EPSILON = 0.10

# Below this, we decline rather than decide.
NO_DECISION_THRESHOLD = 0.35

# Ceilings. Each says: regardless of every other signal, this specific failure
# means we cannot be this confident. The lowest triggered ceiling wins.
#
# Ceilings, not penalties, on purpose: an average lets one strong signal hide a
# serious problem. A ceiling cannot be averaged away.
CAPS = {
    "judges_disagree_on_winner": 0.45,
    "single_effective_judge":    0.50,
    "judge_failed":              0.50,
    "judge_contamination":       0.40,
    "citation_failed":           0.50,
    "reduced_generator_pool":    0.75,
}


# ===========================================================================
# PER-JUDGE READINGS
# ===========================================================================

def weighted_total(criterion_scores):
    """Apply OUR weights to one judge's raw criterion scores. Range 0-5."""
    return sum(criterion_scores[name] * weight
               for name, weight in RUBRIC.items())


def judge_totals(judgement):
    """{label: weighted total} for one judge."""
    return {label: weighted_total(scores)
            for label, scores in judgement["scores"].items()}


def is_discriminating(totals):
    """
    Did this judge express a preference at all?

    A judge that gives every candidate the same score sits numerically close to
    every other judge, so a naive agreement signal would read it as STRONG
    agreement and push confidence up - when it said nothing. Zero variance is
    abstention wearing agreement's clothes.

    Observed live: one judge returned a valid scorecard of 5/5/5/5 for all three
    candidates. On a harder question the same model discriminated sharply and a
    different judge went flat - so this is a per-RUN property, never a per-model
    one, and the check runs on every judge every time.
    """
    if len(totals) < 2:
        return False
    return (max(totals.values()) - min(totals.values())) >= DISCRIMINATION_THRESHOLD


def judge_pick(totals):
    """
    This judge's top candidate, or None if its own top two are tied.

    A judge that scores its leaders equally has not chosen. Reading a winner out
    of a coin-flip-sized gap would invent a preference it never expressed.
    """
    if not totals:
        return None

    ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] - ranked[1][1] < TIE_EPSILON:
        return None
    return ranked[0][0]


# ===========================================================================
# SIGNALS
# ===========================================================================

def _rank_agreement(first, second, labels):
    """
    Fraction of candidate PAIRS the two judges ordered the same way.

    Kendall-style concordance. Ties on either side count as half credit,
    because a tie is neither agreement nor contradiction.
    """
    pairs = [(labels[i], labels[j])
             for i in range(len(labels))
             for j in range(i + 1, len(labels))]
    if not pairs:
        return None

    score = 0.0
    for left, right in pairs:
        gap_first = first[left] - first[right]
        gap_second = second[left] - second[right]

        if gap_first == 0 or gap_second == 0:
            score += 0.5                      # one judge expressed no order
        elif (gap_first > 0) == (gap_second > 0):
            score += 1.0                      # same ordering
        # opposite ordering scores 0

    return score / len(pairs)


def signal_inter_judge_agreement(all_totals):
    """
    How closely two judges agreed - on BOTH the numbers and the ordering.

    Returns None when the question cannot be asked: fewer than two judges that
    expressed a preference. None is not 0.0 and not 1.0. "We have no evidence
    either way" is a third state, and collapsing it into either number is a lie
    about what we know.

    We take the MINIMUM of two readings, because agreement means agreeing on
    both counts. Observed live: two judges scored the same three answers within
    0.65 of each other on a 0-5 scale - proximity 0.89, which reads as near
    perfect agreement - while ranking them in opposite orders. One put candidate
    B first, the other put B last. Score proximity alone called that agreement.
    It is not.
    """
    if len(all_totals) < 2:
        return None

    first, second = all_totals[0], all_totals[1]
    labels = sorted(set(first) & set(second))
    if not labels:
        return None

    mean_gap = sum(abs(first[label] - second[label]) for label in labels) / len(labels)
    proximity = max(0.0, 1.0 - (mean_gap / SCORE_MAX))

    concordance = _rank_agreement(first, second, labels)
    if concordance is None:
        return proximity

    return min(proximity, concordance)


def signal_score_margin(combined):
    """How decisively the winner beat the runner-up. None if nothing to compare."""
    if len(combined) < 2:
        return None

    ranked = sorted(combined.values(), reverse=True)
    gap = ranked[0] - ranked[1]
    return min(1.0, gap / MARGIN_FULL_SCALE)


def signal_winner_quality(combined):
    """
    How good the best candidate actually is, on its own terms (0-1).

    The companion to score_margin. A narrow win among excellent answers is a
    low-risk choice; a narrow win among poor ones is not. Margin alone cannot
    distinguish those, and reading a thin margin as weak evidence caused the
    council to decline a question every generator answered correctly.
    """
    if not combined:
        return None
    return max(combined.values()) / SCORE_MAX


def _content_words(text):
    """Crude bag of meaningful words. Good enough for a rough overlap measure."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {word for word in cleaned.split() if len(word) > 3}


def signal_agent_agreement(candidates):
    """
    Mean pairwise word overlap (Jaccard) between usable answers.

    A rough lexical measure, not a semantic one: two answers can agree in
    substance and share few words, or share many words and contradict. It is
    weighted low partly for that reason and partly because of trap #2.
    """
    texts = [c["answer"] for c in candidates if c["ok"] and not c["abstained"]]
    if len(texts) < 2:
        return None

    bags = [_content_words(text) for text in texts]
    scores = []
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            union = bags[i] | bags[j]
            if union:
                scores.append(len(bags[i] & bags[j]) / len(union))

    return sum(scores) / len(scores) if scores else None


def signal_verification_pass_rate(citations):
    """
    Share of citations that actually checked out. None when nothing was cited.

    None, not 1.0: "made no claims needing a source" must not score the same as
    "made claims and every source held up". A free pass for silence would reward
    vagueness.
    """
    if not citations:
        return None
    verified = sum(1 for c in citations if c.get("status") == "verified")
    return verified / len(citations)


# ===========================================================================
# BLEND + CEILINGS
# ===========================================================================

def blend(signals):
    """
    Weighted average over the signals we actually have.

    Missing signals are DROPPED and the remaining weights renormalised - never
    filled with a default. Substituting 0.5 for a signal we could not measure
    would put an invented number into the output and present it as an
    observation. If nothing is measurable, the answer is 0.0, not a guess.
    """
    available = {name: value for name, value in signals.items() if value is not None}
    if not available:
        return 0.0, {}

    total_weight = sum(SIGNAL_WEIGHTS[name] for name in available)
    contributions = {
        name: (SIGNAL_WEIGHTS[name] / total_weight) * value
        for name, value in available.items()
    }
    return sum(contributions.values()), contributions


def apply_caps(raw_score, triggered):
    """Lowest triggered ceiling wins. Ceilings cannot be averaged away."""
    if not triggered:
        return raw_score, []
    ceiling = min(CAPS[name] for name in triggered)
    return min(raw_score, ceiling), sorted(triggered)


# ===========================================================================
# THE AGGREGATOR
# ===========================================================================

def aggregate(run, citations=None):
    """
    Decide. Returns a dict of winner, status, confidence, risks and workings.

    Status is one of: decided | no_decision. (refused comes from the pre-gate.)
    """
    citations = citations or []
    candidates = run["candidates"]
    judgements = run.get("judgements") or []

    usable = [c for c in candidates if c["ok"] and not c["abstained"]]
    abstained = [c for c in candidates if c["ok"] and c["abstained"]]
    failed = [c for c in candidates if not c["ok"]]

    risks = []
    caps_triggered = set()
    notes = []

    # --- not enough to compare -------------------------------------------
    if len(usable) < 2:
        if abstained and not usable:
            detail = (f"All {len(abstained)} responding generators abstained. "
                      "The council declined to answer rather than invent one.")
            risk_type = "data_gap"
        elif not usable:
            detail = (f"No usable candidate answers: {len(failed)} generator "
                      "call(s) failed.")
            risk_type = "data_gap"
        else:
            detail = ("Only one usable candidate answer; nothing to compare it "
                      "against, so no judging took place.")
            risk_type = "data_gap"

        risks.append({"type": risk_type, "severity": "high", "detail": detail})
        return _no_decision(run, usable, abstained, failed, judgements,
                            risks, detail)

    if len(usable) < len(candidates):
        caps_triggered.add("reduced_generator_pool")
        risks.append({
            "type": "data_gap", "severity": "low",
            "detail": (f"{len(usable)} of {len(candidates)} generators produced "
                       "a usable answer; council diversity reduced."),
        })

    # --- read the judges ---------------------------------------------------
    ok_judges = [j for j in judgements if j["ok"]]

    if len(ok_judges) < len(judgements):
        caps_triggered.add("judge_failed")
        for judgement in judgements:
            if not judgement["ok"]:
                risks.append({
                    "type": "data_gap", "severity": "med",
                    "detail": (f"Judge {judgement['model_id']} unusable: "
                               f"{judgement['detail']}"),
                })

    if any(j.get("problems") for j in ok_judges):
        for judgement in ok_judges:
            for problem in judgement["problems"]:
                if problem.startswith("contamination"):
                    caps_triggered.add("judge_contamination")
                    risks.append({
                        "type": "safety", "severity": "med",
                        "detail": f"{judgement['model_id']}: {problem}",
                    })

    if not ok_judges:
        detail = "No judge produced a usable scorecard; nothing to aggregate."
        risks.append({"type": "data_gap", "severity": "high", "detail": detail})
        return _no_decision(run, usable, abstained, failed, judgements,
                            risks, detail)

    # --- discrimination check (our own trap) -------------------------------
    discriminating = []
    for judgement in ok_judges:
        totals = judge_totals(judgement)
        judgement["_totals"] = totals
        if is_discriminating(totals):
            discriminating.append(judgement)
        else:
            notes.append(f"{judgement['model_id']} scored every candidate "
                         "identically - excluded from the agreement signal")
            risks.append({
                "type": "ambiguity", "severity": "med",
                "detail": (f"Judge {judgement['model_id']} gave every candidate "
                           "the same score. It expressed no preference, so it "
                           "contributes no agreement evidence."),
            })

    if len(discriminating) < 2:
        caps_triggered.add("single_effective_judge")

    # Only judges that expressed a preference get to choose the winner.
    ranking_judges = discriminating or ok_judges

    picks = [judge_pick(j["_totals"]) for j in ranking_judges]
    real_picks = {pick for pick in picks if pick is not None}
    if len(real_picks) > 1:
        caps_triggered.add("judges_disagree_on_winner")
        risks.append({
            "type": "ambiguity", "severity": "high",
            "detail": (f"Judges selected different winners: "
                       f"{sorted(real_picks)}."),
        })

    # --- combine -----------------------------------------------------------
    labels = sorted({label for j in ranking_judges for label in j["_totals"]})
    combined = {
        label: sum(j["_totals"].get(label, 0.0) for j in ranking_judges)
               / len(ranking_judges)
        for label in labels
    }

    ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)

    # --- measure everything we can, BEFORE deciding anything ---------------
    # A no_decision must still report what we observed. Returning "unavailable"
    # for signals we actually measured would throw away evidence and make the
    # object less honest, not more cautious.
    if any(c.get("status") == "failed" for c in citations):
        caps_triggered.add("citation_failed")
        risks.append({
            "type": "factual", "severity": "high",
            "detail": "At least one citation could not be verified as reachable.",
        })

    signals = {
        "inter_judge_agreement": signal_inter_judge_agreement(
            [j["_totals"] for j in discriminating]),
        "score_margin": signal_score_margin(combined),
        "winner_quality": signal_winner_quality(combined),
        "agent_agreement": signal_agent_agreement(candidates),
        "verification_pass_rate": signal_verification_pass_rate(citations),
    }

    # --- tie-break ---------------------------------------------------------
    tie_broken_by = None
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < TIE_EPSILON:
        contenders = [label for label, total in ranked
                      if ranked[0][1] - total < TIE_EPSILON]

        # Rule 1: prefer the candidate with more VERIFIED citations.
        verified_count = {
            label: sum(1 for c in citations
                       if c.get("label") == label and c.get("status") == "verified")
            for label in contenders
        }
        best = max(verified_count.values()) if verified_count else 0
        leaders = [label for label in contenders if verified_count[label] == best]

        if best > 0 and len(leaders) == 1:
            tie_broken_by = "verified_citations"
            ranked = ([(leaders[0], combined[leaders[0]])] +
                      [pair for pair in ranked if pair[0] != leaders[0]])
        else:
            # Rule 2: there is no rule 2. We decline rather than coin-flip.
            detail = (f"Top candidates {sorted(contenders)} are within "
                      f"{TIE_EPSILON} on the combined score and no verified "
                      "citation separates them.")
            risks.append({"type": "ambiguity", "severity": "high",
                          "detail": detail})
            return _no_decision(run, usable, abstained, failed, judgements,
                                risks, detail, combined=combined, signals=signals)

    winner_label, winner_total = ranked[0]
    runner_label, _ = ranked[1] if len(ranked) > 1 else (None, None)

    raw_score, contributions = blend(signals)
    score, applied_caps = apply_caps(raw_score, caps_triggered)

    dropped = [name for name, value in signals.items() if value is None]

    method = (
        "Weighted blend of observable signals, then hard ceilings. "
        f"Weights: {SIGNAL_WEIGHTS}. "
        f"Signals unavailable and dropped (weights renormalised): {dropped or 'none'}. "
        f"Raw blended score {raw_score:.3f}. "
        f"Ceilings triggered: {applied_caps or 'none'} -> final {score:.3f}. "
        "No model was asked how confident it felt."
    )

    by_label = {c["label"]: c for c in usable if c.get("label")}
    status = "decided" if score >= NO_DECISION_THRESHOLD else "no_decision"

    if status == "no_decision":
        risks.append({
            "type": "ambiguity", "severity": "high",
            "detail": (f"Earned confidence {score:.2f} is below the "
                       f"{NO_DECISION_THRESHOLD} threshold; declining to decide."),
        })

    return {
        "status": status,
        "winning_answer": by_label[winner_label]["answer"] if status == "decided" else None,
        "runner_up_answer": (by_label[runner_label]["answer"]
                             if status == "decided" and runner_label else None),
        "winner_label": winner_label if status == "decided" else None,
        "confidence": {
            "score": round(score, 3),
            "method": method,
            "signals": {name: (round(value, 3) if value is not None else None)
                        for name, value in signals.items()},
            "raw_score": round(raw_score, 3),
            "contributions": {k: round(v, 3) for k, v in contributions.items()},
            "caps_triggered": applied_caps,
        },
        "risks": risks,
        "workings": {
            "combined_totals": {k: round(v, 3) for k, v in combined.items()},
            "per_judge_totals": {j["model_id"]: {k: round(v, 3)
                                                 for k, v in j["_totals"].items()}
                                 for j in ok_judges},
            "judge_picks": {j["model_id"]: judge_pick(j["_totals"])
                            for j in ranking_judges},
            "non_discriminating_judges": [j["model_id"] for j in ok_judges
                                          if j not in discriminating],
            "tie_broken_by": tie_broken_by,
            "notes": notes,
        },
    }


def _no_decision(run, usable, abstained, failed, judgements, risks, detail,
                 combined=None, signals=None):
    """A well-formed 'we decline'. Not an error, not an empty response."""
    if signals is None:
        signals = {
            "inter_judge_agreement": None,
            "score_margin": None,
            "winner_quality": None,
            "agent_agreement": signal_agent_agreement(run["candidates"]),
            "verification_pass_rate": None,
        }

    return {
        "status": "no_decision",
        "winning_answer": None,
        "runner_up_answer": None,
        "winner_label": None,
        "confidence": {
            # 0.0 is not a measurement of doubt - it records that there is no
            # decision here to be confident about. The signals below say what
            # we actually observed on the way to declining.
            "score": 0.0,
            "method": f"No decision reached: {detail} Confidence is 0.0 by "
                      "definition - there is no decision to be confident about. "
                      "The signals recorded below are what we measured before "
                      "declining.",
            "signals": {name: (round(value, 3) if value is not None else None)
                        for name, value in signals.items()},
            "raw_score": 0.0,
            "contributions": {},
            "caps_triggered": [],
        },
        "risks": risks,
        "workings": {
            "combined_totals": {k: round(v, 3) for k, v in (combined or {}).items()},
            "usable": len(usable),
            "abstained": len(abstained),
            "failed": len(failed),
        },
    }
