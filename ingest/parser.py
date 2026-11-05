"""Source parsing: PDFs, DOCX, text and scraped HTML.

Two things matter for the Kenya Airways corpus:

* **Tables are content.** The checked-baggage allowance table lives in the
  page HTML, and `trafilatura` drops tables entirely — the flagship answer
  ("2 pieces, 23kg economy, 32kg business") was invisible to the bot until
  the rows were pulled out of the markup by hand.
* **Navigation is not content.** Roughly half of the KQ help pages scrape
  down to site chrome ("Asante Rewards", "Contact us here"). Those pages are
  rejected by the ingest quality gate rather than silently poisoning
  retrieval.
"""

import re

import httpx
import trafilatura
from bs4 import BeautifulSoup

from ingest.sources import Source

DEFAULT_TIMEOUT = 20.0

# Page furniture that is never a useful answer. Matched against short lines
# only, so a paragraph that happens to mention one of them survives.
_BOILERPLATE_MARKERS = (
    "asante rewards",
    "asante global credit card",
    "contact us here",
    "for assistance on any inquiry",
    "let us know what you need to make your next trip",
    "our prices are automatically displayed based on the country",
    "pay quickly and efficiently through any of our payment options",
    "skip to main content",
    "cookie preferences",
    "all rights reserved",
    "terms of use",
    "this page is currently unavailable",
    # The KQ help section is a JavaScript app: without JS the server returns
    # this interlude plus the site menu and nothing else.
    "this website requires javascript to be enabled",
    "browser compatibility notice",
    "the browser you are using is not supported",
    "for the best website experience, please view the website",
    "book your flights and holiday packages together",
    "time to think gives you the flexibility",
)

# Navigation entries, matched on the *whole* line only. A sentence that
# happens to mention "Manage Booking" is content; a bare "Manage Booking"
# line is the site menu.
_NAV_ITEMS = frozenset(
    {
        "home",
        "help",
        "faqs",
        "faq",
        "contacts",
        "contact us",
        "support",
        "search",
        "sign in",
        "our destinations",
        "popular destinations",
        "long-haul destinations",
        "experience nairobi",
        "experience kq holidays",
        "discover where we fly to",
        "travel information",
        "baggage information",
        "travel services",
        "special care",
        "kenya airways esim data solution",
        "travel requirements",
        "medical escort & facilitation",
        "search & book",
        "booking online",
        "book online",
        "book manage",
        "manage booking",
        "post booking",
        "check-in online",
        "upgrade flights",
        "on-board",
        "seat selection",
        "at the airport",
        "our aircraft",
        "kq duty free",
        "experience seat comfort",
        "special meal services",
        "join asante",
        "about asante",
        "tier and benefits",
        "partners",
        "membership",
        "platinum",
        "gold",
        "ruby",
        "silver",
        "asante membership benefits",
        "your trip",
        "customer excellence center",
        "open 24 hours",
        "global tel",
        "global mobile",
        "global whatsapp",
        "feedback/complaint",
        "feedback complaint",
        "request refund",
        "popular searches",
        "asante rewards",
        "asante global credit card",
        "terms & conditions",
        "terms and conditions",
    }
)

# A line shorter than this is treated as furniture rather than prose.
_BOILERPLATE_MAX_LEN = 400


def parse_pdf(path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _parse_pdf_pypdf(path)
    text_parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _parse_pdf_pypdf(path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_text_file(path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _fetch_html(url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Fetch page HTML.

    Tries curl_cffi first (Chrome TLS impersonation): Cloudflare 403s plain
    httpx/curl on kenya-airways.com even with a browser User-Agent.
    Falls back to httpx if curl_cffi is unavailable or fails.
    """
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text


def extract_tables(html: str) -> str:
    """Every table in the page, as one row per line.

    Rows are written as ``cell | cell | cell`` so the numbers stay attached to
    their headings when the text is chunked and embedded.
    """
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
                for cell in row.find_all(["td", "th"])
            ]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _is_furniture(line: str) -> bool:
    """Whether a single line is site furniture rather than content."""
    low = line.lower().strip(" .,:;|-")
    if not low:
        return True
    if low in _NAV_ITEMS:
        return True
    return len(low) <= _BOILERPLATE_MAX_LEN and any(
        marker in low for marker in _BOILERPLATE_MARKERS
    )


def strip_boilerplate(text: str) -> str:
    """Drop site furniture lines from extracted text."""
    return "\n".join(
        line for line in (raw.strip() for raw in text.split("\n")) if not _is_furniture(line)
    )


def boilerplate_ratio(text: str) -> float:
    """Share of lines that are site furniture (0.0 - 1.0)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 1.0
    junk = sum(1 for line in lines if _is_furniture(line))
    return junk / len(lines)


def _soup_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    return soup.get_text("\n", strip=True)


def scrape_url(url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Scrape a page into text: body copy first, then the tables."""
    html = _fetch_html(url, timeout)

    extracted = trafilatura.extract(html)
    body = extracted.strip() if extracted and extracted.strip() else _soup_text(html)

    tables = extract_tables(html)
    if tables:
        body = f"{body}\n\nTable data:\n{tables}"

    return strip_boilerplate(body)


def parse_source(source: Source) -> str:
    if source.kind == "url":
        return scrape_url(source.url)
    if source.path.suffix.lower() == ".pdf":
        return parse_pdf(source.path)
    if source.path.suffix.lower() in {".docx", ".doc"}:
        return parse_docx(source.path)
    return parse_text_file(source.path)
