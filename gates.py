"""
gates.py - the pre-gate and post-gate.

DELIBERATELY RULE-BASED. No model is consulted here, and that is a design
decision rather than a shortcut:

  - deterministic: the same question is always refused for the same reason. We
    measured our own models producing different output for identical prompts at
    temperature 0, so a model-based classifier cannot promise this.
  - free: costs nothing against a ~200 request/day ceiling, and a refusal
    therefore costs zero API calls and zero seconds.
  - not promptable: a classifier reading the user's text can be argued with.
    "Ignore previous instructions" does not work on a regular expression.
  - auditable: the exact rule that fired is recorded in the Decision Object and
    in the hash-chained log, so a refusal can be explained months later.

The cost is coverage. A rule list catches the blatant and misses the subtle, and
a determined person can rephrase around it. We take that trade knowingly: a gate
that fails predictably is more useful in an audit trail than one that fails
creatively. See the README for what we would add with more time.

Both functions return (ok, rule, reason) so the caller can put the rule name
straight into the Decision Object.
"""

import re

MAX_QUESTION_CHARS = 4000
MIN_QUESTION_CHARS = 3

# (rule name, human-readable reason, [patterns])
#
# Patterns pair an ACTION with a TARGET on purpose. "How does ransomware
# spread?" is a legitimate question a security engineer might ask; "write me
# ransomware" is not. Matching the target alone would refuse the researcher
# along with the attacker, and a gate that blocks ordinary learning gets
# switched off by the people it is meant to protect.
DENY_RULES = [
    (
        "malware_authoring",
        "Requests working malicious software.",
        [r"\b(write|create|build|generate|code|develop|make|give me)\b[\s\S]{0,60}"
         r"\b(keylogger|ransomware|trojan|rootkit|botnet|spyware|"
         r"credential stealer|infostealer|computer virus)\b"],
    ),
    (
        "intrusion",
        "Requests unauthorised access to systems or accounts.",
        [r"\b(hack|break)\s+into\b[\s\S]{0,40}"
         r"\b(account|server|network|database|system|wi-?fi|phone|email)\b",
         r"\bbypass\b[\s\S]{0,30}\b(authentication|2fa|mfa|login|password)\b",
         r"\b(crack|steal|dump|exfiltrate)\b[\s\S]{0,30}"
         r"\b(password|credential|token)s?\b"],
    ),
    (
        "weapons_manufacture",
        "Requests instructions for building weapons or explosives.",
        [r"\b(how (to|do i) (make|build|construct)|instructions for "
         r"(making|building)|step[- ]by[- ]step)\b[\s\S]{0,50}"
         r"\b(bomb|explosive|ied|napalm|thermite|nerve agent|"
         r"chemical weapon|biological weapon|untraceable (gun|firearm))\b"],
    ),
    (
        "illicit_synthesis",
        "Requests synthesis routes for controlled substances or poisons.",
        [r"\b(synthesi[sz]e|manufacture|cook|produce|extract|make)\b[\s\S]{0,40}"
         r"\b(methamphetamine|fentanyl|heroin|ricin|sarin|vx nerve)\b"],
    ),
    (
        "self_harm",
        "Requests methods of self-harm or suicide.",
        [r"\b(how to|best way to|easiest way to|painless(ly)?)\b[\s\S]{0,40}"
         r"\b(kill myself|commit suicide|end my life|hang myself|overdose)\b",
         r"\b(lethal|fatal)\s+(dose|amount)\b"],
    ),
    (
        "csam",
        "Requests sexual content involving minors.",
        [r"\b(child|children|minor|minors|underage|pre-?teen)\w*\b[\s\S]{0,30}"
         r"\b(porn|sexual|nude|explicit|erotic)\w*\b",
         r"\b(porn|sexual|nude|explicit|erotic)\w*\b[\s\S]{0,30}"
         r"\b(child|children|minor|minors|underage|pre-?teen)\w*\b"],
    ),
    (
        "targeted_harm",
        "Requests help harming, stalking, or doxxing a specific person.",
        [r"\b(how (to|do i)|help me)\b[\s\S]{0,40}"
         r"\b(stalk|dox|harass|poison|hurt|kill|track)\b[\s\S]{0,25}"
         r"\b(my|his|her|their|this)\s+\w+"],
    ),
    (
        "fraud",
        "Requests help committing fraud or identity theft.",
        [r"\b(how (to|do i)|help me)\b[\s\S]{0,40}"
         r"\b(launder money|commit fraud|commit identity theft|"
         r"forge (a|an|my)?\s*\w*(signature|document|passport|id)|"
         r"counterfeit)\b"],
    ),
]

COMPILED = [
    (name, reason, [re.compile(pattern, re.IGNORECASE) for pattern in patterns])
    for name, reason, patterns in DENY_RULES
]


def check_text(text):
    """
    Return (rule_name, reason) for the first rule that fires, else None.

    First match wins and we stop looking. Enumerating every rule a text violates
    would tell someone exactly which phrasings to avoid next time, which turns
    the refusal message into a map around the gate.
    """
    if not text:
        return None
    for name, reason, patterns in COMPILED:
        for pattern in patterns:
            if pattern.search(text):
                return name, reason
    return None


def pre_gate(question):
    """
    Screen the question BEFORE any model is called.

    Returns (allowed, rule, reason). A refusal here costs zero API calls and
    zero seconds, and still produces a complete, schema-valid Decision Object
    that goes into the audit chain like any other decision.
    """
    stripped = (question or "").strip()

    if len(stripped) < MIN_QUESTION_CHARS:
        return False, "scope_empty", "Question is empty or too short to answer."

    if len(stripped) > MAX_QUESTION_CHARS:
        return (False, "scope_too_long",
                f"Question exceeds {MAX_QUESTION_CHARS} characters.")

    hit = check_text(stripped)
    if hit:
        return False, hit[0], hit[1]

    return True, None, None


def post_gate(answer):
    """
    Screen the winning answer before we hand it over.

    The same rules, applied to output. A question can pass the pre-gate and
    still draw an unsafe answer - the gate that screens input is not the gate
    that screens output, even when the rules behind them are identical.
    """
    hit = check_text(answer or "")
    if hit:
        return False, hit[0], hit[1]
    return True, None, None
