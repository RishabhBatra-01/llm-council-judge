"""
client.py - the only place in this project that talks to the network.

Everything else calls call_model() and gets back a dict. It never raises:
a dead model is data, not a crash.
"""

import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]      # crash now if the key is missing

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MAX_TOKENS = 1500
TEMPERATURE = 0
TIMEOUT_SECONDS = 60

# How we react to each HTTP status. Retrying a 403 forever burns quota on
# something that can never succeed; not retrying a 429 throws away a model
# that was fine.
RETRYABLE = {429, 500, 502, 503, 504}
FATAL = {400, 401, 403, 404}

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 2.0


def backoff_seconds(attempt):
    """2s, then 4s, then 8s. Doubling gives congestion room to clear."""
    return BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))


def retry_after_seconds(response):
    """Trust the server's number over our guess, when it gives one."""
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return min(float(header), 30.0)
    except ValueError:
        return None                      # it may be an HTTP date; ignore


def read_reply(response, elapsed_ms, attempts):
    """
    Read a 200 reply and decide whether it is actually USABLE.

    HTTP 200 only means the network call worked. A model can still return an
    empty string, or stop mid-sentence having run out of tokens.
    """
    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError) as error:
        return {"ok": False, "failure_stage": "parse",
                "detail": f"unreadable reply: {error}",
                "text": "", "latency_ms": elapsed_ms, "attempts": attempts,
                "tokens": 0, "cost_usd": 0.0, "served_by": "?",
                "finish_reason": "?", "reasoning_tokens": None}

    usage = data.get("usage", {})
    details = usage.get("completion_tokens_details") or {}

    finish_reason = choice.get("finish_reason", "?")
    content = (message.get("content") or "").strip()

    result = {
        "ok": True,
        "failure_stage": None,
        "detail": "",
        "text": content,
        "latency_ms": elapsed_ms,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens"),  # None = not reported
        "served_by": data.get("model", "?"),
        "cost_usd": usage.get("cost", 0.0),
    }

    if finish_reason == "length":
        result["ok"] = False
        result["failure_stage"] = "content"
        result["detail"] = f"truncated at max_tokens ({len(content)} chars)"
    elif not content:
        result["ok"] = False
        result["failure_stage"] = "content"
        result["detail"] = f"empty content (finish_reason={finish_reason})"

    return result


def failure(stage, detail, elapsed_ms=0, attempts=0):
    """A failed call, shaped exactly like a successful one."""
    return {"ok": False, "failure_stage": stage, "detail": detail,
            "text": "", "latency_ms": elapsed_ms, "attempts": attempts,
            "tokens": 0, "cost_usd": 0.0, "served_by": "?",
            "finish_reason": "?", "reasoning_tokens": None}


def call_model(model_id, prompt, system_prompt=None, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Send one prompt to one model, retrying only when retrying can help.

    Always returns a dict with the same keys, ok or not.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    detail = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()

        try:
            response = requests.post(
                CHAT_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": max_tokens,
                },
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            detail = f"network error: {error}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(backoff_seconds(attempt))
                continue
            break

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = response.status_code

        if status == 200:
            return read_reply(response, elapsed_ms, attempt)

        body = response.text[:120].replace("\n", " ")

        if status in FATAL:
            return failure("http", f"HTTP {status} (fatal): {body}",
                           elapsed_ms, attempt)

        detail = f"HTTP {status}: {body}"

        if status in RETRYABLE and attempt < MAX_ATTEMPTS:
            wait = retry_after_seconds(response) or backoff_seconds(attempt)
            print(f"      HTTP {status}, retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue

        return failure("http", detail, elapsed_ms, attempt)

    return failure("network", detail, 0, MAX_ATTEMPTS)
