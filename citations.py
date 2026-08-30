"""
citations.py - check the sources a generator cited.

What this does: pulls URLs out of answer text, opens each one, and reports
whether it resolved and whether the claim's wording plausibly appears on the
page.

What this does NOT do, and will not be extended to do without a human in the
loop: decide whether the source actually SUPPORTS the claim. See README §7.
A live URL proves a page exists. It does not prove the page says what the
model said it says.

Three statuses, and the distinction between the last two matters:

  verified    reachable AND enough of the claim's wording appears on the page
  unverified  reachable, but the claim could not be located there
  failed      unreachable: 404, timeout, DNS failure, connection refused

`unverified` is not a soft `failed`. It means "we looked and could not tell",
which is a different fact about the world from "this source does not exist",
and the Decision Object keeps them apart.
"""

import re

import requests

from config import (CITATION_MAX_PER_RUN, CITATION_TIMEOUT,
                    CITATION_USER_AGENT, CLAIM_OVERLAP_THRESHOLD)

# Deliberately permissive about surrounding punctuation: models wrap URLs in
# brackets, parentheses, angle brackets and CJK brackets more or less at random.
URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)】」]+")

# Trailing punctuation that belongs to the sentence, not the URL.
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
    """
    The sentence containing the URL, used as the claim being sourced.

    A heuristic, and a shaky one: models do not reliably put a claim and its
    citation in the same sentence. It is the best available approximation
    without asking generators for structured JSON, which we avoid on purpose
    (every model forced into JSON is another parse failure waiting to happen).
    Its weakness is why a citation we cannot confirm becomes `unverified`
    rather than `failed`.
    """
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
    # Strip the URL and any bracket it was wearing out of the claim itself.
    claim = claim.replace(url, "").strip(" ()[]【】<>「」,;:")
    return re.sub(r"\s+", " ", claim)[:300]


def _content_words(text):
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {w for w in cleaned.split() if len(w) > 3 and w not in STOPWORDS}


def extract(candidates):
    """
    Find every cited URL across the usable answers.

    Returns a list of {label, claim, source} - unchecked. Deduplicated by
    (label, url) so a model citing the same source three times does not get
    three chances to raise its own verification_pass_rate.
    """
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
    """
    Open one URL and decide what we can honestly say about it.

    Never raises. A dead link is data.
    """
    url = citation["source"]
    headers = {"User-Agent": CITATION_USER_AGENT}

    result = dict(citation, status="failed", http_status=None, detail="")

    try:
        # HEAD first: cheap, and enough to settle reachability. Some servers
        # reject HEAD outright, so fall through to GET rather than calling a
        # working page dead because of a method restriction.
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

    # Reachable. Now the weaker question: does the claim's wording appear?
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
