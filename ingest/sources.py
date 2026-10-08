from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data_sources")
SOURCES_LIST = DATA_DIR / "data_source1.txt"
LOCAL_EXTS = {".pdf", ".docx", ".doc", ".txt"}


@dataclass
class Source:
    title: str
    topic: str
    kind: str
    path: Path | None = None
    url: str | None = None

    @property
    def key(self) -> str:
        if self.kind == "url":
            return f"url::{self.url}"
        return f"file::{self.path.as_posix()}"

    @property
    def source_url(self) -> str | None:
        return self.url

    @property
    def doc_type(self) -> str:
        if self.kind == "url":
            return "html"
        ext = (self.path.suffix or "").lower()
        return {".pdf": "pdf", ".docx": "docx", ".doc": "docx", ".txt": "text"}.get(ext, "text")


_TOPIC_RULES = [
    ("conditions", "Conditions of Carriage"),
    ("carriage", "Conditions of Carriage"),
    ("rebook", "Rebooking"),
    ("faq", "FAQ"),
    ("travel-information", "Travel Requirements"),
    ("baggage", "Baggage"),
    ("refund", "Refunds"),
    ("flight", "Flights"),
]


def _infer_topic(name: str) -> str:
    lowered = name.lower()
    for token, topic in _TOPIC_RULES:
        if token in lowered:
            return topic
    return "General"


def _title_from_url(url: str) -> str:
    stripped = url.rstrip("/")
    segment = stripped.rsplit("/", 1)[-1]
    return segment or stripped


def discover_local(directory: Path = DATA_DIR) -> list[Source]:
    if not directory.exists():
        return []
    sources: list[Source] = []
    for path in sorted(directory.iterdir()):
        if path.is_dir() or path.name == SOURCES_LIST.name:
            continue
        if path.suffix.lower() in LOCAL_EXTS:
            sources.append(
                Source(
                    title=path.stem,
                    topic=_infer_topic(path.stem),
                    kind="local",
                    path=path,
                )
            )
    return sources


def discover_urls(list_path: Path = SOURCES_LIST) -> list[Source]:
    if not list_path.exists():
        return []
    sources: list[Source] = []
    lines = (line.strip() for line in list_path.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if line.startswith(("http://", "https://")):
            sources.append(
                Source(
                    title=_title_from_url(line),
                    topic=_infer_topic(line),
                    kind="url",
                    url=line,
                )
            )
    return sources


def discover_all(extra_sources: list[str] | None = None) -> list[Source]:
    sources = discover_local() + discover_urls()
    for raw in extra_sources or []:
        if raw.startswith(("http://", "https://")):
            sources.append(
                Source(title=_title_from_url(raw), topic=_infer_topic(raw), kind="url", url=raw)
            )
        else:
            path = Path(raw)
            if path.exists():
                sources.append(
                    Source(title=path.stem, topic=_infer_topic(path.stem), kind="local", path=path)
                )
    return sources