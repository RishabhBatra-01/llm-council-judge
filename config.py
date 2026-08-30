"""
config.py - load config.yaml once and expose it as named constants.

Why a loader at all: the brief asks for pinned model IDs, the rubric and the
confidence formula to live in a config file rather than scattered through the
code. This module is the single place that reads it, so every other module
imports settings from here and nowhere else.

Loading fails LOUDLY. A missing or malformed config is our own broken setup,
not someone else's bad data, and there is no sensible way to continue - running
with silently-defaulted weights would produce Decision Objects whose
config_hash claimed settings we never actually used.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load(path=CONFIG_PATH):
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"config is not a mapping: {path}")
    return loaded


CONFIG = load()

# --- models ----------------------------------------------------------------
_models = CONFIG["models"]
GENERATORS = [{"model_id": g["model_id"], "family": g["family"]}
              for g in _models["generators"]]
JUDGES = [{"model_id": j["model_id"], "family": j["family"]}
          for j in _models["judges"]]
BACKUPS = [{"model_id": b["model_id"], "family": b["family"]}
           for b in _models.get("backups", [])]

# --- rubric ----------------------------------------------------------------
_rubric = CONFIG["rubric"]
RUBRIC = dict(_rubric["criteria"])
SCORE_MIN = _rubric["score_min"]
SCORE_MAX = _rubric["score_max"]

# --- confidence ------------------------------------------------------------
_confidence = CONFIG["confidence"]
SIGNAL_WEIGHTS = dict(_confidence["signal_weights"])
CAPS = dict(_confidence["caps"])

# --- thresholds ------------------------------------------------------------
_thresholds = CONFIG["thresholds"]
MARGIN_FULL_SCALE = _thresholds["margin_full_scale"]
DISCRIMINATION_THRESHOLD = _thresholds["discrimination"]
TIE_EPSILON = _thresholds["tie_epsilon"]
NO_DECISION_THRESHOLD = _thresholds["no_decision"]
LOW_JUDGE_AGREEMENT = _thresholds["low_judge_agreement"]

# --- generation ------------------------------------------------------------
_generation = CONFIG["generation"]
TEMPERATURE = _generation["temperature"]
GENERATOR_MAX_TOKENS = _generation["generator_max_tokens"]
JUDGE_MAX_TOKENS = _generation["judge_max_tokens"]
PAUSE_BETWEEN_CALLS = _generation["pause_between_calls"]

# --- client ----------------------------------------------------------------
_client = CONFIG["client"]
TIMEOUT_SECONDS = _client["timeout_seconds"]
MAX_ATTEMPTS = _client["max_attempts"]
BASE_BACKOFF_SECONDS = _client["base_backoff_seconds"]
RETRYABLE = set(_client["retryable_status"])
FATAL = set(_client["fatal_status"])

# --- citations -------------------------------------------------------------
_citations = CONFIG["citations"]
CITATION_TIMEOUT = _citations["timeout_seconds"]
CITATION_MAX_PER_RUN = _citations["max_per_run"]
CLAIM_OVERLAP_THRESHOLD = _citations["claim_overlap_threshold"]
CITATION_USER_AGENT = _citations["user_agent"]

# --- judging ---------------------------------------------------------------
_judging = CONFIG["judging"]
MAX_JUSTIFICATION_CHARS = _judging["max_justification_chars"]
CONTAMINATION_PHRASES = list(_judging["contamination_phrases"])

# --- prompts ---------------------------------------------------------------
ABSTAIN_MARKER = CONFIG["abstain_marker"]
GENERATOR_SYSTEM_PROMPT = CONFIG["prompts"]["generator_system"].strip()
JUDGE_SYSTEM_PROMPT = CONFIG["prompts"]["judge_system"].strip()


def snapshot():
    """
    Every setting that could change a decision, in one dict.

    This is what config_hash fingerprints. We return the parsed config rather
    than the raw file text so the hash tracks CONTENT, not comments or
    whitespace - reformatting the YAML must not make two identical runs look
    incomparable, and editing a weight must.
    """
    return {
        "models": {"generators": GENERATORS, "judges": JUDGES},
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
            "temperature": TEMPERATURE,
            "generator_max_tokens": GENERATOR_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
        "citations": {
            "claim_overlap_threshold": CLAIM_OVERLAP_THRESHOLD,
            "timeout_seconds": CITATION_TIMEOUT,
        },
        "prompts": {
            "generator": GENERATOR_SYSTEM_PROMPT,
            "judge": JUDGE_SYSTEM_PROMPT,
        },
    }
