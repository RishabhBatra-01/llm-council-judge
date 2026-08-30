"""
Load config.yaml once and expose it as named constants.
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

_models = CONFIG["models"]
GENERATORS = [{"model_id": g["model_id"], "family": g["family"]}
              for g in _models["generators"]]
JUDGES = [{"model_id": j["model_id"], "family": j["family"]}
          for j in _models["judges"]]
BACKUPS = [{"model_id": b["model_id"], "family": b["family"]}
           for b in _models.get("backups", [])]

_rubric = CONFIG["rubric"]
RUBRIC = dict(_rubric["criteria"])
SCORE_MIN = _rubric["score_min"]
SCORE_MAX = _rubric["score_max"]

_confidence = CONFIG["confidence"]
SIGNAL_WEIGHTS = dict(_confidence["signal_weights"])
CAPS = dict(_confidence["caps"])

_thresholds = CONFIG["thresholds"]
MARGIN_FULL_SCALE = _thresholds["margin_full_scale"]
DISCRIMINATION_THRESHOLD = _thresholds["discrimination"]
TIE_EPSILON = _thresholds["tie_epsilon"]
NO_DECISION_THRESHOLD = _thresholds["no_decision"]
LOW_JUDGE_AGREEMENT = _thresholds["low_judge_agreement"]

_generation = CONFIG["generation"]
TEMPERATURE = _generation["temperature"]
GENERATOR_MAX_TOKENS = _generation["generator_max_tokens"]
JUDGE_MAX_TOKENS = _generation["judge_max_tokens"]
PAUSE_BETWEEN_CALLS = _generation["pause_between_calls"]

_client = CONFIG["client"]
TIMEOUT_SECONDS = _client["timeout_seconds"]
MAX_ATTEMPTS = _client["max_attempts"]
BASE_BACKOFF_SECONDS = _client["base_backoff_seconds"]
RETRYABLE = set(_client["retryable_status"])
FATAL = set(_client["fatal_status"])

_citations = CONFIG["citations"]
CITATION_TIMEOUT = _citations["timeout_seconds"]
CITATION_MAX_PER_RUN = _citations["max_per_run"]
CLAIM_OVERLAP_THRESHOLD = _citations["claim_overlap_threshold"]
CITATION_USER_AGENT = _citations["user_agent"]

_judging = CONFIG["judging"]
MAX_JUSTIFICATION_CHARS = _judging["max_justification_chars"]
CONTAMINATION_PHRASES = list(_judging["contamination_phrases"])

ABSTAIN_MARKER = CONFIG["abstain_marker"]
GENERATOR_SYSTEM_PROMPT = CONFIG["prompts"]["generator_system"].strip()
JUDGE_SYSTEM_PROMPT = CONFIG["prompts"]["judge_system"].strip()


def snapshot():
    """Every setting that could change a decision, in one dict."""
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
