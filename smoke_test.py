"""smoke_test.py - prove every model we plan to use is actually alive. Now with retry + backoff, because the free tier hands out 429s that"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

RETRYABLE = {429, 500, 502, 503, 504}
FATAL     = {400, 401, 403, 404}

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 2.0
PAUSE_BETWEEN_MODELS = 2.0

MAX_TOKENS = 1000

TEST_QUESTION = "Reply with exactly one word: ok"

CANDIDATES = [
    ("judge?", "cohere/north-mini-code:free"),
    ("judge?", "poolside/laguna-s-2.1:free"),
    ("judge?", "liquid/lfm-2.5-2.6b:free"),
    ("judge?", "z-ai/glm-5.2:free"),
    ("gen?",   "dots-studio/dots-3-note-preview:free"),
]


def backoff_seconds(attempt):
    """2s, then 4s, then 8s. Doubling gives congestion room to clear."""
    return BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))


def retry_after_seconds(response):
    """If the server told us exactly how long to wait, trust it over our guess."""
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return min(float(header), 30.0)
    except ValueError:
        return None


def describe_success(response, elapsed_ms, attempts):
    """Read a 200 reply and decide whether it is actually USABLE."""
    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError) as error:
        return {"ok": False, "failure_stage": "parse",
                "detail": f"unreadable reply: {error}",
                "ms": elapsed_ms, "attempts": attempts}

    usage = data.get("usage", {})
    details = usage.get("completion_tokens_details") or {}

    finish_reason = choice.get("finish_reason", "?")
    content = (message.get("content") or "").strip()

    result = {
        "ok": True,
        "failure_stage": None,
        "detail": "",
        "ms": elapsed_ms,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "served_by": data.get("model", "?"),
        "content_chars": len(content),
        "reply": content[:40],
    }

    if finish_reason == "length":
        result["ok"] = False
        result["failure_stage"] = "content"
        result["detail"] = f"truncated at max_tokens ({len(content)} chars of content)"
    elif not content:
        result["ok"] = False
        result["failure_stage"] = "content"
        result["detail"] = f"empty content (finish_reason={finish_reason})"

    return result


def call_model(model_id, question=TEST_QUESTION):
    """Send one question to one model, retrying when it makes sense to."""
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
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=60,
            )
        except requests.RequestException as error:
            detail = f"network error: {error}"
            if attempt < MAX_ATTEMPTS:
                wait = backoff_seconds(attempt)
                print(f"        network error, retrying in {wait:.0f}s "
                      f"(attempt {attempt}/{MAX_ATTEMPTS})")
                time.sleep(wait)
                continue
            break

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = response.status_code

        if status == 200:
            return describe_success(response, elapsed_ms, attempt)

        body = response.text[:120].replace("\n", " ")

        if status in FATAL:
            return {"ok": False, "failure_stage": "http",
                    "detail": f"HTTP {status} (fatal): {body}",
                    "ms": elapsed_ms, "attempts": attempt}

        detail = f"HTTP {status}: {body}"

        if status in RETRYABLE and attempt < MAX_ATTEMPTS:
            wait = retry_after_seconds(response) or backoff_seconds(attempt)
            print(f"        HTTP {status}, retrying in {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(wait)
            continue

        return {"ok": False, "failure_stage": "http", "detail": detail,
                "ms": elapsed_ms, "attempts": attempt}

    return {"ok": False, "failure_stage": "network", "detail": detail,
            "ms": 0, "attempts": MAX_ATTEMPTS}


def format_reasoning(reasoning_tokens):
    if reasoning_tokens is None:
        return "reasoning not reported"
    if reasoning_tokens == 0:
        return "no reasoning"
    return f"{reasoning_tokens} reasoning"


def main():
    print(f"Testing {len(CANDIDATES)} models\n")

    working, broken = [], []

    for role, model_id in CANDIDATES:
        result = call_model(model_id)

        if result["ok"]:
            print(f"  OK    {role:<10} {model_id:<48} {result['ms']:>6} ms  "
                  f"{result['tokens']:>4} tok  ({format_reasoning(result['reasoning_tokens'])})  "
                  f"finish={result['finish_reason']}  tries={result['attempts']}")
            print(f"        reply: {result['reply']!r}")

            if result["served_by"] != model_id:
                print(f"        NOTE: served by {result['served_by']}")

            working.append((role, model_id))
        else:
            stage = result.get("failure_stage", "?")
            print(f"  FAIL  {role:<10} {model_id:<48} [{stage}] {result['detail']}")
            broken.append((role, model_id))

        print()
        time.sleep(PAUSE_BETWEEN_MODELS)

    print("-" * 70)
    print(f"{len(working)} working, {len(broken)} broken")
    for role, model_id in broken:
        print(f"  unusable: {role:<10} {model_id}")


if __name__ == "__main__":
    main()
