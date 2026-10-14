from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.deps import get_current_agent
from app.models import Agent
from app.schemas import IngestReportOut, KnowledgeDocumentOut, UrlIngestIn

router = APIRouter(prefix="/kb", tags=["kb"])

ALLOWED_EXTS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file


def _infer_topic(name: str) -> str:
    lowered = name.lower()
    for token, topic in (
        ("conditions", "Conditions of Carriage"),
        ("carriage", "Conditions of Carriage"),
        ("rebook", "Rebooking"),
        ("faq", "FAQ"),
        ("travel-information", "Travel Requirements"),
        ("baggage", "Baggage"),
        ("refund", "Refunds"),
        ("flight", "Flights"),
    ):
        if token in lowered:
            return topic
    return "General"


def _parse_source_key(key: str) -> tuple[str, str | None]:
    if key.startswith("url::"):
        return ("url", key[5:])
    if key.startswith("file::"):
        return ("file", key[6:])
    return ("unknown", key)


@router.get("/sources", response_model=list[KnowledgeDocumentOut])
async def list_sources(agent: Agent = Depends(get_current_agent)):
    from ingest.store import get_store

    store = get_store()
    try:
        documents = await store.list_documents()
    finally:
        await store.close()
    return [KnowledgeDocumentOut(**doc) for doc in documents]


@router.post("/upload", response_model=IngestReportOut)
async def upload_files(
    files: list[UploadFile] = File(...),
    embed: bool = Query(True),
    force: bool = Query(False),
    agent: Agent = Depends(get_current_agent),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    from ingest.embedder import Embedder
    from ingest.pipeline import ingest_sources
    from ingest.sources import Source
    from ingest.store import get_store

    data_dir = Path("data_sources")
    data_dir.mkdir(exist_ok=True)

    sources: list[Source] = []
    for upload in files:
        name = upload.filename or "upload"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF and DOCX files are allowed. Got: {name}",
            )
        # avoid collisions
        stem = Path(name).stem
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-3]
        safe_name = f"{stem}_{ts}{ext}"
        dest = data_dir / safe_name
        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                ),
            )
        dest.write_bytes(content)

        sources.append(
            Source(
                title=stem,
                topic=_infer_topic(stem),
                kind="local",
                path=dest,
            )
        )

    store = get_store()
    embedder = Embedder() if embed else None
    try:
        await store.ensure_schema()
        report = await ingest_sources(
            sources,
            store,
            embedder=embedder,
            force=force,
            embed=embed,
        )
    finally:
        await store.close()

    return IngestReportOut(
        new=report.new,
        updated=report.updated,
        unchanged=report.unchanged,
        failed=report.failed,
        summary=report.summary(),
    )


@router.post("/urls", response_model=IngestReportOut)
async def ingest_urls(
    payload: UrlIngestIn,
    embed: bool = Query(True),
    force: bool = Query(False),
    agent: Agent = Depends(get_current_agent),
):
    from ingest.embedder import Embedder
    from ingest.pipeline import ingest_sources
    from ingest.sources import Source, _title_from_url
    from ingest.store import get_store

    sources: list[Source] = []
    for url in payload.urls:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
        sources.append(
            Source(
                title=_title_from_url(url),
                topic=_infer_topic(url),
                kind="url",
                url=url,
            )
        )

    store = get_store()
    embedder = Embedder() if embed else None
    try:
        await store.ensure_schema()
        report = await ingest_sources(
            sources,
            store,
            embedder=embedder,
            force=force,
            embed=embed,
        )
    finally:
        await store.close()

    return IngestReportOut(
        new=report.new,
        updated=report.updated,
        unchanged=report.unchanged,
        failed=report.failed,
        summary=report.summary(),
    )


@router.post("/sources/{source_key:path}/reingest", response_model=IngestReportOut)
async def reingest_source(
    source_key: str,
    embed: bool = Query(True),
    agent: Agent = Depends(get_current_agent),
):
    from ingest.embedder import Embedder
    from ingest.pipeline import ingest_sources
    from ingest.sources import Source, _title_from_url
    from ingest.store import get_store

    kind, rest = _parse_source_key(source_key)
    sources: list[Source] = []
    if kind == "url" and rest:
        sources.append(
            Source(
                title=_title_from_url(rest),
                topic=_infer_topic(rest),
                kind="url",
                url=rest,
            )
        )
    elif kind == "file" and rest:
        p = Path(rest)
        sources.append(
            Source(
                title=p.stem,
                topic=_infer_topic(p.stem),
                kind="local",
                path=p,
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid source key")

    store = get_store()
    embedder = Embedder() if embed else None
    try:
        await store.ensure_schema()
        report = await ingest_sources(
            sources,
            store,
            embedder=embedder,
            force=True,
            embed=embed,
        )
    finally:
        await store.close()

    return IngestReportOut(
        new=report.new,
        updated=report.updated,
        unchanged=report.unchanged,
        failed=report.failed,
        summary=report.summary(),
    )


@router.delete("/sources/{source_key:path}")
async def delete_source(
    source_key: str,
    agent: Agent = Depends(get_current_agent),
):
    from ingest.store import get_store

    store = get_store()
    try:
        found = await store.delete_source(source_key)
    finally:
        await store.close()
    if not found:
        raise HTTPException(status_code=404, detail="Source not found")

    # delete file if local
    kind, rest = _parse_source_key(source_key)
    if kind == "file" and rest:
        try:
            p = Path(rest)
            if p.exists():
                p.unlink()
        except Exception:
            pass  # ignore file deletion errors

    return {"ok": True}


@router.post("/reset")
async def reset_kb(
    agent: Agent = Depends(get_current_agent),
):
    from ingest.store import get_store

    store = get_store()
    try:
        await store.reset()
    finally:
        await store.close()
    return {"ok": True}
