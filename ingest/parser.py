import httpx
import trafilatura
from bs4 import BeautifulSoup

from ingest.sources import Source

DEFAULT_TIMEOUT = 20.0


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


def scrape_url(url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    html = _fetch_html(url, timeout)

    extracted = trafilatura.extract(html)
    if extracted and extracted.strip():
        return extracted.strip()

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    text = soup.get_text("\n", strip=True)
    return text


def parse_source(source: Source) -> str:
    if source.kind == "url":
        return scrape_url(source.url)
    if source.path.suffix.lower() == ".pdf":
        return parse_pdf(source.path)
    if source.path.suffix.lower() in {".docx", ".doc"}:
        return parse_docx(source.path)
    return parse_text_file(source.path)