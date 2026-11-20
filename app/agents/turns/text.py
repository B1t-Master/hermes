"""Answer-text cleanup and citation extraction for passenger-facing replies."""

from __future__ import annotations

import re

from app.config import settings

# Passenger-facing answers must never expose the internal data-source origin
# (the docs are ingested from KQ URLs/file keys). Strip markdown links and bare
# URLs from the visible answer text; citations stay in the sources[] metadata.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*((?:https?://|www\.)[^)\s]*)\s*\)", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# A trailing "Sources: ..." block some models append despite instructions.
_SOURCES_TRAILER = re.compile(r"\n+\s*sources?\s*:.*$", re.IGNORECASE | re.DOTALL)


def _official_urls() -> tuple[str, ...]:
    """Official KQ links that stay in the answer text.

    Citation URLs are internal (they point at ingested documents), so they are
    stripped. The official Kenya Airways links are the opposite: they are the
    passenger's next step. Dropping them used to leave sentences like
    "you can re-book yourself online at " with nothing after them.
    """
    return tuple(
        url
        for url in (
            settings.self_service_rebooking_url,
            settings.kq_help_url,
            settings.kq_manage_booking_url,
            settings.worldtracer_url,
        )
        if url
    )


# Hosts whose links are the passenger's next step rather than a citation.
_OFFICIAL_HOSTS = ("kenya-airways.com", "worldtracer.aero")

# Words that only make sense because a link used to follow them. Removing the
# link without them leaves "go to and fill in the form".
_DANGLING_CONNECTORS = (
    "go to",
    "visit",
    "open",
    "browse",
    "check",
    "see",
    "read",
    "at",
    "on",
    "via",
    "using",
    "from",
)

# A bare link, optionally with the connector that introduces it, optionally
# followed by the conjunction that continues the sentence.
_BARE_LINK_RE = re.compile(
    r"(?:\b(" + "|".join(_DANGLING_CONNECTORS) + r")([ \t]+))?"
    r"((?:https?://|www\.)\S+)"
    r"(?:[ \t]+(and|or|for|to|then)\b)?",
    re.IGNORECASE,
)


def _is_official_url(url: str) -> bool:
    """Whether a link is an official KQ/partner page worth keeping in the text."""
    if url in _official_urls():
        return True
    host = url.lower().split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0]
    host = host.split("@")[-1].split(":")[0]
    return any(host == h or host.endswith("." + h) for h in _OFFICIAL_HOSTS)


def strip_source_urls(text: str) -> str:
    """Remove source URLs from answer text, keeping any link label.

    The UI renders ``sources[]`` separately (title-only), so the body can stay
    clean. Handles markdown links (``[title](url)`` -> ``[title]``) and bare
    URLs; collapses the whitespace/parens left behind. Official KQ links are
    kept as real links.
    """
    if not text:
        return text
    keep: list[str] = []

    def _link(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if _is_official_url(url):
            # Park the official link so the bare-URL sweep cannot eat it.
            keep.append(f"[{label}]({url})")
            return f"\ue000{len(keep) - 1}\ue001"
        return f"[{label}]"

    cleaned = _MD_LINK_RE.sub(_link, text)

    def _bare(match: re.Match) -> str:
        connector, space, url, conjunction = match.group(1), match.group(2), match.group(3), match.group(4)
        if _is_official_url(url):
            lead = f"{connector}{space}" if connector else ""
            trail = f" {conjunction}" if conjunction else ""
            return f"{lead}{url}{trail}"
        # The link goes, and so does the connector that introduced it: leaving
        # them behind produced "go to  and fill in the form".
        return ""

    cleaned = _BARE_LINK_RE.sub(_bare, cleaned)
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # A link that opened a clause leaves the rest of it hanging.
    cleaned = re.sub(r"^[\s,]*(?:for|and|or|to|then)\s+", "", cleaned, flags=re.IGNORECASE)
    for index, link in enumerate(keep):
        cleaned = cleaned.replace(f"\ue000{index}\ue001", link)
    return cleaned.strip()


def strip_source_section(text: str) -> str:
    """Drop a trailing "Sources: ..." block a model may append.

    Source credits belong to `message.sources` (rendered as chips by the UI),
    never to the answer body. Belt-and-braces with the prompt instruction.
    """
    if not text:
        return text
    return _SOURCES_TRAILER.sub("", text).rstrip()


def citations_from_chunks(chunks: list) -> list[dict]:
    """Deduplicated [{title, url}] pairs from the graph's retrieved chunks.

    The graph stores chunks as plain dicts (RetrievedChunk.__dict__), so keys
    are `title`, `source_url`, `content`, `score`, `topic`, `chunk_index`.
    """
    citations: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk in chunks:
        title = chunk.get("title") or chunk.get("source_url") or "Source"
        url = chunk.get("source_url")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        citation: dict = {"title": title, "url": url}
        score = chunk.get("score")
        if score is not None:
            citation["score"] = round(score, 4)
        citations.append(citation)
    return citations
