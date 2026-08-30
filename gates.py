"""
Pre-gate and post-gate. Deterministic rules, no model call.
"""

import re

MAX_QUESTION_CHARS = 4000
MIN_QUESTION_CHARS = 3

# Each pattern pairs an ACTION with a TARGET: "how does ransomware spread?"
# must pass, while a request to write one must not.
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
    """Return (rule_name, reason) for the first rule that fires, else None."""
    if not text:
        return None
    for name, reason, patterns in COMPILED:
        for pattern in patterns:
            if pattern.search(text):
                return name, reason
    return None


def pre_gate(question):
    """Screen the question BEFORE any model is called."""
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
    """Screen the winning answer before we hand it over."""
    hit = check_text(answer or "")
    if hit:
        return False, hit[0], hit[1]
    return True, None, None
