"""One passenger turn = persist input -> run graph -> persist outputs.

Shared by the WebSocket handler (streaming) and the REST fallback
(non-streaming), so there is exactly one pipeline to reason about.

Ordering note: the passenger message is persisted *after* the graph run,
because the graph's sentiment-streak trigger reads prior passenger messages
from the DB; persisting first would make the current turn count as its own
history. The tradeoff: a crash mid-graph loses that one inbound message.
"""

import logging
import re
import time
import uuid

from sqlalchemy import func, or_, select, update

from app.agents.facts import (
    apply_booking_result,
    apply_rebook_result,
    detect_human_decline,
    extract_facts,
)
from app.agents.graph import get_graph
from app.config import settings
from app.db import SessionLocal
from app.metrics import ESCALATION, RESPONSE_TIME_MS, RETRIEVAL_HITS, add_metric
from app.models import (
    Booking,
    Conversation,
    ConversationStatus,
    Message,
    MessageSender,
    Passenger,
)

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 10

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


_ROLE = {MessageSender.PASSENGER.value: "user", MessageSender.BOT.value: "assistant", MessageSender.AGENT.value: "assistant"}


def message_to_dict(message: Message) -> dict:
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "sender": message.sender.value,
        "content": message.content,
        "sentiment_score": message.sentiment_score,
        "sentiment_label": message.sentiment_label,
        "sources": message.sources or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
        "actions": message.actions or None,
        "payment_id": str(message.payment_id) if message.payment_id else None,
    }
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
async def load_history(conversation_id: uuid.UUID, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Prior turns as chat messages, oldest first."""
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()
    rows = list(reversed(rows))
    return [{"role": _ROLE.get(m.sender.value, "assistant"), "content": m.content} for m in rows]
async def get_conversation(conversation_id: uuid.UUID) -> Conversation | None:
    async with SessionLocal() as db:
        return (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()


def booking_to_dict(booking: Booking) -> dict:
    """A booking as a small, prompt-friendly dict."""
    return {
        "pnr": booking.pnr,
        "flight_no": booking.flight_no,
        "origin": booking.origin,
        "destination": booking.destination,
        "date": booking.depart_date,
        "time": booking.depart_time,
        "status": booking.status.value if booking.status else None,
    }


async def load_profile(passenger_id: uuid.UUID) -> dict:
    """Identity + bookings for a passenger, for prompt inheritance.

    Logged-in passengers carry their name and email; their bookings are matched
    by ``passenger_id`` or by the profile email, so a booking made on another
    channel (or seeded in the prototype) still shows up here. Anonymous
    passengers have neither, and only conversation facts are used.
    """
    async with SessionLocal() as db:
        passenger = (
            await db.execute(select(Passenger).where(Passenger.id == passenger_id))
        ).scalar_one_or_none()
        if passenger is None:
            return {}

        conditions = [Booking.passenger_id == passenger_id]
        if passenger.email:
            conditions.append(
                func.lower(Booking.passenger_email) == passenger.email.lower()
            )
        bookings = (
            await db.execute(
                select(Booking)
                .where(or_(*conditions))
                .order_by(Booking.depart_date.asc().nulls_last())
            )
        ).scalars().all()

    return {
        "name": " ".join(
            part for part in (passenger.first_name, passenger.last_name) if part
        ),
        "email": passenger.email,
        "anonymous": passenger.is_anonymous,
        "bookings": [booking_to_dict(booking) for booking in bookings],
    }


async def run_turn(
    conversation_id: uuid.UUID,
    query: str,
    on_token=None,
) -> dict:
    """Run one graph turn. Returns the final graph state plus persisted messages.

    Expected shape of the return value:
      {**state, "passenger_message": {...}, "bot_message": {...} | None}
    """
    conversation = await get_conversation(conversation_id)
    if conversation is None:
        raise LookupError("conversation not found")
    if conversation.status == ConversationStatus.RESOLVED:
        raise PermissionError("conversation is closed")

    # Escalated conversations belong to the human agent: no bot turns.
    if conversation.status == ConversationStatus.ESCALATED:
        async with SessionLocal() as db:
            message = Message(
                conversation_id=conversation_id,
                sender=MessageSender.PASSENGER,
                content=query,
            )
            db.add(message)
            conversation.turn_count += 1
            await db.commit()
            await db.refresh(message)
        return {
            "escalate": True,
            "escalation_created": False,
            "escalation_reason": conversation.escalation_reason or "",
            "passenger_message": message_to_dict(message),
            "bot_message": None,
        }

    history = await load_history(conversation_id)
    # Merge this turn's slots into the conversation memory before the graph
    # runs, so the answer prompt can reuse them (and never re-ask).
    facts = extract_facts(query, conversation.facts or {})
    # Also extract from recent assistant messages to capture rebook results.
    # Assistant messages may only fill slots the passenger has not given: a
    # rebooking confirmation quotes a *confirmation* reference, and letting
    # that overwrite the passenger's ticket number broke "what is my ticket
    # number?" a few turns after a rebooking.
    for m in history[-6:]:
        if m.get("role") == "assistant":
            facts = extract_facts(m.get("content", ""), facts, source="assistant")
    # The passenger declining a hand-off ("no thanks, I'll continue") stops the
    # bot from pushing one on every later turn of the same conversation.
    if detect_human_decline(query):
        facts["human_declined"] = True
    profile = await load_profile(conversation.passenger_id)

    state = {
        "conversation_id": str(conversation_id),
        "passenger_id": str(conversation.passenger_id),
        "query": query,
        "history": history,
        # Pre-loaded conversation facts, so the graph's history triggers do
        # not need to re-fetch the row (one fewer DB round trip per turn).
        "conversation_status": conversation.status.value,
        "conversation_turn_count": conversation.turn_count,
        "soft_offers": conversation.soft_offers,
        "facts": facts,
        "profile": profile,
    }
    config = {"configurable": {"on_token": on_token}} if on_token is not None else None
    started = time.perf_counter()
    result = await get_graph().ainvoke(state, config=config)
    response_time_ms = (time.perf_counter() - started) * 1000.0
    result = dict(result)

    escalation_created = bool(result.get("escalation_created"))
    chunks = result.get("chunks") or []
    answer = strip_source_section(strip_source_urls(result.get("answer", "")))
    result["answer"] = answer

    # A successful rebooking is authoritative: the tool knows the ticket, the
    # confirmation reference, the flight and the date exactly. Write them into
    # the memory so the next question about the booking is answered from the
    # booking, not from whatever the model happened to repeat in prose.
    for tool_result in result.get("tool_results") or []:
        facts = apply_rebook_result(facts, tool_result)
        facts = apply_booking_result(facts, tool_result)

    # A booking tool result carries the payment card to render under the bot
    # message. The card collects the M-Pesa number directly.
    payment_uuid = None
    for tool_result in result.get("tool_results") or []:
        candidate = tool_result.get("payment_id")
        if not candidate:
            continue
        try:
            payment_uuid = uuid.UUID(str(candidate))
        except (ValueError, TypeError):
            continue

    async with SessionLocal() as db:
        # Only the first message carrying a payment id renders a card. If an
        # earlier message already holds this payment, drop the id from this
        # turn so a status/ticket reply cannot add a duplicate card.
        if payment_uuid is not None:
            existing = (
                await db.execute(
                    select(Message.id)
                    .where(
                        Message.conversation_id == conversation_id,
                        Message.payment_id == payment_uuid,
                    )
                    .limit(1)
                )
            ).first()
            if existing is not None:
                payment_uuid = None

        # A card needs a bot message to live under. If the tool produced a
        # payment but the model returned no text, synthesize a short cue.
        if payment_uuid is not None and not answer:
            answer = (
                "Your booking is held. Enter your M-Pesa number in the payment "
                "card below to receive the prompt."
            )

        passenger_message = Message(
            conversation_id=conversation_id,
            sender=MessageSender.PASSENGER,
            content=query,
            sentiment_score=result.get("sentiment_score"),
            sentiment_label=result.get("sentiment_label"),
        )
        db.add(passenger_message)
        bot_message = None
        # The bot always answers now, including on the escalation path (the
        # acknowledgement). A hand-off used to leave the passenger with no
        # reply at all.
        if answer:
            escalation_offer = result.get("escalation_offer") or {}
            bot_message = Message(
                conversation_id=conversation_id,
                sender=MessageSender.BOT,
                content=answer,
                sources=citations_from_chunks(result.get("chunks") or []),
                actions=escalation_offer.get("options") or None,
                payment_id=payment_uuid,
            )
            db.add(bot_message)
        add_metric(db, conversation_id, RESPONSE_TIME_MS, response_time_ms)
        add_metric(db, conversation_id, RETRIEVAL_HITS, len(chunks))
        if escalation_created:
            add_metric(db, conversation_id, ESCALATION, 1.0)
        # Increment the counter in-place; avoids a select-then-update.
        await db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(turn_count=Conversation.turn_count + 1, facts=facts)
        )
        await db.commit()
        await db.refresh(passenger_message)
        if bot_message is not None:
            await db.refresh(bot_message)

    result["passenger_message"] = message_to_dict(passenger_message)
    result["bot_message"] = message_to_dict(bot_message) if bot_message else None
    return result
