"""
Check the sources a generator cited: is the URL reachable, and does the
claim's wording appear there. Reachability is not truth - see README section 7.
"""

import re

import requests

from config import (CITATION_MAX_PER_RUN, CITATION_TIMEOUT,
                    CITATION_USER_AGENT, CLAIM_OVERLAP_THRESHOLD)

URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)】」]+")

TRAILING_JUNK = ".,;:!?'\"»)]}】"

STOPWORDS = {
    "this", "that", "these", "those", "with", "from", "have", "been", "were",
    "what", "when", "where", "which", "while", "would", "could", "should",
    "there", "their", "them", "then", "than", "about", "into", "over", "some",
    "such", "also", "more", "most", "other", "your", "http", "https", "www",
    "source", "according",
}


def _clean_url(raw):
    return raw.rstrip(TRAILING_JUNK)


def _sentence_around(text, url):
    """The sentence containing the URL, used as the claim being sourced."""
    position = text.find(url)
    if position == -1:
        return text[:300].strip()

    start = max(text.rfind(".", 0, position),
                text.rfind("\n", 0, position),
                text.rfind("- ", 0, position))
    start = 0 if start == -1 else start + 1

    end = len(text)
    for mark in (". ", ".\n", "\n"):
        found = text.find(mark, position)
        if found != -1:
            end = min(end, found + 1)

    claim = text[start:end].strip()
    claim = claim.replace(url, "").strip(" ()[]【】<>「」,;:")
    return re.sub(r"\s+", " ", claim)[:300]


def _content_words(text):
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {w for w in cleaned.split() if len(w) > 3 and w not in STOPWORDS}


def extract(candidates):
    """Find every cited URL across the usable answers."""
    found = []
    seen = set()

    for candidate in candidates:
        if not candidate.get("ok") or candidate.get("abstained"):
            continue

        answer = candidate.get("answer", "")
        for raw in URL_PATTERN.findall(answer):
            url = _clean_url(raw)
            key = (candidate.get("label"), url)
            if not url or key in seen:
                continue
            seen.add(key)
            found.append({
                "label": candidate.get("label"),
                "claim": _sentence_around(answer, raw),
                "source": url,
            })

    return found[:CITATION_MAX_PER_RUN]


def check_one(citation):
    """Open one URL and decide what we can honestly say about it."""
    url = citation["source"]
    headers = {"User-Agent": CITATION_USER_AGENT}

    result = dict(citation, status="failed", http_status=None, detail="")

    try:
        # HEAD is enough for reachability; some servers reject it, so fall back.
        response = requests.head(url, timeout=CITATION_TIMEOUT,
                                 allow_redirects=True, headers=headers)
        if response.status_code >= 400 or not response.content:
            response = requests.get(url, timeout=CITATION_TIMEOUT,
                                    allow_redirects=True, headers=headers)
    except requests.RequestException as error:
        result["detail"] = f"unreachable: {type(error).__name__}"
        return result

    result["http_status"] = response.status_code

    if response.status_code >= 400:
        result["detail"] = f"HTTP {response.status_code}"
        return result

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        result["status"] = "unverified"
        result["detail"] = (f"reachable, but {content_type or 'unknown type'} "
                            "cannot be searched for the claim")
        return result

    try:
        body = response.text
    except Exception:
        result["status"] = "unverified"
        result["detail"] = "reachable, but body could not be decoded"
        return result

    page_words = _content_words(re.sub(r"<[^>]+>", " ", body))
    claim_words = _content_words(citation.get("claim", ""))

    if not claim_words:
        result["status"] = "unverified"
        result["detail"] = "reachable, but no claim text to check against"
        return result

    overlap = len(claim_words & page_words) / len(claim_words)

    if overlap >= CLAIM_OVERLAP_THRESHOLD:
        result["status"] = "verified"
        result["detail"] = (f"reachable; {overlap:.0%} of claim wording present. "
                            "Reachability and word overlap are not proof the "
                            "source supports the claim.")
    else:
        result["status"] = "unverified"
        result["detail"] = (f"reachable, but only {overlap:.0%} of claim wording "
                            f"found (threshold {CLAIM_OVERLAP_THRESHOLD:.0%})")

    return result


def verify_all(candidates):
    """Extract and check every citation. Returns the list for the Decision Object."""
    return [check_one(citation) for citation in extract(candidates)]
