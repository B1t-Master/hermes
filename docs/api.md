# API reference

Hermes exposes a FastAPI REST API and two WebSocket channels. The interactive
schema is always available at `/docs` (Swagger UI) and `/openapi.json` when the
backend is running.

## Base URLs and conventions

| Context | Base URL |
|---|---|
| Local backend | `http://localhost:8000` |
| Docker frontend | `http://localhost:5173` |
| Frontend dev proxy | `/api` is rewritten to the backend root |
| Frontend container proxy | `/api/` is proxied to `http://backend:8000/` |

- Request and response bodies are JSON unless noted (file uploads use
  `multipart/form-data`).
- Authenticated REST requests use `Authorization: Bearer <access_token>`.
- WebSocket connections pass the token as `?token=<access_token>`.
- IDs are UUID strings. Timestamps are ISO-8601.
- Roles are `passenger` and `agent`. A token for one role is rejected by
  endpoints that require the other.

### Common response shapes

```json
// Token
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "role": "passenger"
}
```

```json
// MessageOut
{
  "id": "0f3c...",
  "conversation_id": "6a1e...",
  "sender": "bot",
  "content": "Your checked baggage allowance is ...",
  "sentiment_score": null,
  "sentiment_label": null,
  "sources": [{"title": "Baggage allowance", "url": "https://..."}],
  "actions": [{"label": "Connect me to a human", "value": "I'd like to speak to a human agent."}],
  "payment_id": null,
  "created_at": "2026-11-05T08:00:00Z"
}
```

```json
// PaymentOut
{
  "id": "b2a1...",
  "conversation_id": "6a1e...",
  "pnr": "KQUFBNK",
  "phone": null,
  "phone_masked": "0791 *** 504",
  "amount": 1.0,
  "currency": "KES",
  "status": "success",
  "receipt": "SGR7...",
  "checkout_request_id": "ws_CO_...",
  "flight_no": "KQ100",
  "origin": "NBO",
  "destination": "LHR",
  "depart_date": "2026-12-05",
  "depart_time": "21:40",
  "passenger_name": "Isaac",
  "passenger_email": null,
  "booking_status": "paid",
  "eticket_number": "074...",
  "message": null,
  "created_at": "2026-11-05T08:00:00Z",
  "updated_at": "2026-11-05T08:01:00Z"
}
```

## Authentication (`/auth`)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | none | Register a passenger. `201`. |
| `POST` | `/auth/login` | none | Passenger email/password login. |
| `POST` | `/auth/anonymous` | none | Create a guest passenger and token. |
| `POST` | `/auth/agent/register` | none | Create an agent account. `201`. |
| `POST` | `/auth/agent/login` | none | Agent login. |
| `POST` | `/auth/refresh` | bearer | Exchange a valid token for a fresh one. |
| `GET` | `/auth/me` | passenger | Current passenger profile. |
| `GET` | `/auth/agent/me` | agent | Current agent profile. |

### `POST /auth/register`

```json
{
  "first_name": "Isaac",
  "last_name": "Mwangi",
  "email": "isaac@example.com",
  "password": "minimum-eight-characters"
}
```

Returns `PassengerOut`:

```json
{
  "id": "0f3c...",
  "first_name": "Isaac",
  "last_name": "Mwangi",
  "email": "isaac@example.com",
  "is_anonymous": false
}
```

Errors: `409` email already registered, `422` validation error.

### `POST /auth/login`

```json
{"email": "isaac@example.com", "password": "minimum-eight-characters"}
```

Returns `Token`. Errors: `401` invalid email or password.

### `POST /auth/anonymous`

```json
{"first_name": "Guest"}
```

`first_name` is optional and defaults to `Guest`. Returns `Token`.

### `POST /auth/agent/register`

```json
{
  "name": "Support Agent",
  "email": "agent@example.com",
  "password": "minimum-eight-characters"
}
```

Returns `Token` with `role: "agent"`. Errors: `409` email already registered.

### `POST /auth/agent/login`

The body is a permissive object. `username` is the agent email for registered
agents, or the env-defined bootstrap username.

```json
{"username": "agent@example.com", "password": "minimum-eight-characters"}
```

Returns `Token` with `role: "agent"`. Errors: `401` invalid agent credentials.

### `POST /auth/refresh`

Send the current bearer token. Returns a new `Token` for the same subject and
role. Errors: `401` missing/invalid token or the account no longer exists.

### `GET /auth/me` and `GET /auth/agent/me`

Return `PassengerOut` / `AgentOut`.

```json
{"id": "0f3c...", "name": "Support Agent", "email": "agent@example.com"}
```

## Chat (`/chat`)

All chat endpoints require a passenger token. A passenger can only access their
own conversations and payments.

| Method | Path | Description |
|---|---|---|
| `GET` | `/chat/conversations` | List the passenger's conversations, newest first. |
| `POST` | `/chat/conversations` | Create a conversation. `201`. |
| `GET` | `/chat/conversations/{conversation_id}/messages` | List messages, oldest first. |
| `POST` | `/chat/conversations/{conversation_id}/messages` | Non-streaming turn fallback. |
| `GET` | `/chat/payments/{payment_id}` | Current payment-card state. |
| `POST` | `/chat/payments/{payment_id}/phone` | Submit an M-Pesa phone number. |
| `POST` | `/chat/payments/{payment_id}/email` | Save an e-ticket email after payment. |
| `POST` | `/chat/conversations/{conversation_id}/csat` | Submit a 1–5 rating. `204`. |

### `GET /chat/conversations`

Returns `list[ConversationOut]`:

```json
[
  {
    "id": "6a1e...",
    "status": "active",
    "turn_count": 3,
    "escalation_reason": null
  }
]
```

`status` is `active`, `escalated`, or `resolved`.

### `POST /chat/conversations`

No body. Returns a `ConversationOut` with `201`.

### `GET /chat/conversations/{conversation_id}/messages`

Returns `list[MessageOut]`, oldest first. Errors: `404` conversation not found
or not owned by the passenger.

### `POST /chat/conversations/{conversation_id}/messages`

This is the non-streaming fallback used by scripts and the evaluation harness.
The path id and body id must match in practice.

```json
{
  "conversation_id": "6a1e...",
  "content": "What is my checked baggage allowance?"
}
```

Returns `TurnOut`:

```json
{
  "passenger_message": { "...": "MessageOut" },
  "bot_message": { "...": "MessageOut" },
  "escalate": false,
  "escalation_reason": null,
  "offer_escalation": false,
  "escalation_offer": null,
  "intent": "routine",
  "sentiment_score": 0.12,
  "sentiment_label": "positive"
}
```

Notes:

- `bot_message` may be non-null on an escalation (the acknowledgement).
- `escalation_offer` is an object such as
  `{"reason": "...", "options": [{"label": "...", "value": "..."}]}` when a
  soft offer is attached.
- Errors: `400` conversation is closed or turn rejected, `404` conversation not
  found.

### `GET /chat/payments/{payment_id}`

Returns `PaymentOut`. Used by the payment card to poll state. Errors: `404`
payment not found or not owned through a conversation.

### `POST /chat/payments/{payment_id}/phone`

```json
{"phone": "0712345678"}
```

Normalises the number, starts the STK push, and schedules polling. The phone is
stored on the payment row and masked in responses; it never enters the LLM
prompt. Returns `PaymentOut`. Errors: `400` invalid number/provider error, `404`
payment not found.

### `POST /chat/payments/{payment_id}/email`

```json
{"email": "passenger@example.com"}
```

Stores the optional e-ticket address on the booking and, when the payment has
succeeded, issues the e-ticket and emails a copy immediately. Also posts a
confirmation message into the conversation and broadcasts a payment update.
Returns `PaymentOut`.

Errors: `409` payment has not succeeded, `404` booking/payment not found.

### `POST /chat/conversations/{conversation_id}/csat`

```json
{"rating": 5}
```

Rating must be an integer `1..5`. Resubmission replaces the previous rating.
Returns `204 No Content`.

## Agent dashboard (`/agents`)

All agent endpoints require an agent token.

| Method | Path | Description |
|---|---|---|
| `GET` | `/agents/escalations` | Queue listing. |
| `GET` | `/agents/escalations/{escalation_id}/messages` | Transcript for an escalation. |
| `POST` | `/agents/escalations/{escalation_id}/claim` | Atomically claim a pending escalation. |
| `POST` | `/agents/escalations/{escalation_id}/release` | Return your claimed escalation to the queue. |
| `POST` | `/agents/escalations/{escalation_id}/resolve` | Close the escalation and conversation. |
| `GET` | `/agents/payments/{payment_id}` | Read payment state from a transcript. |
| `POST` | `/agents/payments/{payment_id}/send-ticket` | Issue the e-ticket and send it in chat. |
| `GET` | `/agents/metrics` | Aggregated dashboard metrics. |

### `GET /agents/escalations?status=active`

`status` is one of `active` (default; pending + assigned), `pending`,
`assigned`, `resolved`, `all`. Results are oldest first.

Returns `list[QueueEscalationOut]`:

```json
[
  {
    "id": "e1...",
    "conversation_id": "6a1e...",
    "reason": "human_request",
    "summary": "passenger: give me a human agent\nbot: ...",
    "status": "pending",
    "created_at": "2026-11-05T08:00:00Z",
    "passenger_name": "Guest",
    "turn_count": 2,
    "conversation_status": "escalated",
    "agent_id": null,
    "agent_name": null,
    "last_message": "Payment received. An agent is preparing your e-ticket."
  }
]
```

### `GET /agents/escalations/{escalation_id}/messages`

Returns `list[MessageOut]` for the escalation's conversation. Any authenticated
agent may inspect a pending card before claiming it.

### `POST /agents/escalations/{escalation_id}/claim`

No body. Atomically moves `pending → assigned`. Exactly one concurrent claimant
can win; the loser gets `409`. Returns `EscalationOut`.

```json
{
  "id": "e1...",
  "conversation_id": "6a1e...",
  "reason": "human_request",
  "summary": "...",
  "status": "assigned",
  "created_at": "2026-11-05T08:00:00Z"
}
```

### `POST /agents/escalations/{escalation_id}/release`

No body. Moves `assigned → pending` and clears the agent. Only the assigned agent
may release. Returns `EscalationOut`. Errors: `409` not your escalation.

### `POST /agents/escalations/{escalation_id}/resolve`

No body. Requires the caller to be the assigned agent. Sets the escalation to
`resolved`, closes the conversation, notifies the passenger room, and broadcasts
a queue update. Returns `EscalationOut`. Errors: `409` claim first.

### `GET /agents/payments/{payment_id}`

Returns `PaymentOut`. Errors: `404` payment not found.

### `POST /agents/payments/{payment_id}/send-ticket`

No body. Requires:

- the caller to own the conversation's assigned escalation, and
- payment status `success`.

Issues the e-ticket, posts an `agent` message in the conversation, broadcasts a
payment update, and returns:

```json
{
  "payment": { "...": "PaymentOut" },
  "ticket": {
    "ticketed": true,
    "pnr": "KQUFBNK",
    "eticket_number": "074...",
    "email_sent": true,
    "email_note": "..."
  },
  "message": { "...": "MessageOut" }
}
```

Errors: `409` not claimed or payment not successful/ticket failed, `404` payment
not found.

### `GET /agents/metrics`

```json
{
  "avg_response_ms": 2500.0,
  "avg_retrieval_hits": 5.0,
  "escalations": 3,
  "csat_responses": 1,
  "avg_csat": 5.0
}
```

Fields may be `null` when no data exists.

## Knowledge base (`/kb`)

All knowledge-base endpoints require an agent token.

| Method | Path | Description |
|---|---|---|
| `GET` | `/kb/sources` | List ingested documents. |
| `POST` | `/kb/upload` | Upload PDF/DOCX files and ingest them. |
| `POST` | `/kb/urls` | Ingest one or more URLs. |
| `POST` | `/kb/sources/{source_key}/reingest` | Force re-ingest one source. |
| `DELETE` | `/kb/sources/{source_key}` | Delete a source and its fragments. |
| `POST` | `/kb/reset` | Delete the entire knowledge base. |

### `GET /kb/sources`

Returns `list[KnowledgeDocumentOut]`:

```json
[
  {
    "id": "d1...",
    "source_url": "https://www.kenya-airways.com/en-ke/contact/",
    "title": "contact",
    "doc_type": "html",
    "content_hash": "…",
    "last_fetched": "2026-11-05T08:00:00Z",
    "created_at": "2026-11-04T08:00:00Z",
    "fragment_count": 17
  }
]
```

`doc_type` is `pdf`, `docx`, `html`, or `text`.

### `POST /kb/upload`

`multipart/form-data` with one or more `files` fields. Query parameters:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `embed` | bool | `true` | Compute embeddings. |
| `force` | bool | `false` | Ignore content hashes. |

Allowed extensions: `.pdf`, `.docx`. Maximum 50 MB per file. Files are stored in
`data_sources/` with a timestamp suffix. Returns `IngestReportOut`:

```json
{
  "new": 1,
  "updated": 0,
  "unchanged": 0,
  "failed": [],
  "summary": "1 new, 0 updated, 0 unchanged, 0 failed"
}
```

Errors: `400` no files, unsupported type, or oversized file.

### `POST /kb/urls`

```json
{"urls": ["https://www.kenya-airways.com/en-ke/help/faqs/"]}
```

Query parameters `embed` (default `true`) and `force` (default `false`). URLs
must start with `http://` or `https://`. Returns `IngestReportOut`. Errors:
`400` invalid URL.

### `POST /kb/sources/{source_key}/reingest`

`source_key` is path-encoded and uses one of these forms:

- `url::https://www.kenya-airways.com/...`
- `file::data_sources/example.pdf`

Query parameter `embed` (default `true`). Forces re-ingestion. Returns
`IngestReportOut`.

### `DELETE /kb/sources/{source_key}`

Deletes the document, its fragments, and the local file when the key is a
`file::` source. Returns `{"ok": true}`. Errors: `404` source not found.

### `POST /kb/reset`

Deletes all knowledge documents and fragments. Returns `{"ok": true}`.

## M-Pesa callback

### `POST /daraja/callback`

Safaricom posts the STK result here. The path deliberately avoids the substring
`mpesa`, which Safaricom rejects in callback URLs.

- Optional query parameter `token` must match `MPESA_CALLBACK_SECRET` when that
  setting is configured; otherwise `403`.
- The endpoint always returns `200` with
  `{"ResultCode": 0, "ResultDesc": "Accepted"}` for accepted payloads, including
  unknown checkout ids, so Safaricom does not retry aggressively. The background
  poller covers late or missing callbacks.
- A late `success` callback overrides a stored `timeout`.

## Health

### `GET /health`

```json
{"status": "ok", "app": "hermes Agentic RAG"}
```

## WebSocket protocols

Every frame is a JSON object with a `type` field. Connection URL:

```text
ws://localhost:8000/ws/conversations/{conversation_id}?token=<jwt>
ws://localhost:8000/ws/dashboard?token=<jwt>
```

### Conversation channel

`/ws/conversations/{conversation_id}?token=...`

Authorisation:

- A passenger token must own the conversation.
- An agent token must own the conversation's **assigned** escalation.
- A resolved conversation is closed with code `4410`.

Close codes:

| Code | Meaning |
|---|---|
| `4401` | Unauthenticated / invalid token. |
| `4403` | Forbidden — not your conversation, or claim the escalation first. |
| `4410` | Conversation is closed. |
| `1011` | Internal server error. |

#### Client → server

| Type | Payload | Who |
|---|---|---|
| `user_message` | `{"type":"user_message","content":"..."}` | Passenger |
| `agent_message` | `{"type":"agent_message","content":"..."}` | Assigned agent |
| `typing` | `{"type":"typing"}` | Either; relayed to the room |
| `ping` | `{"type":"ping"}` | Either; server replies `pong` |
| `pong` | `{"type":"pong"}` | Either; heartbeat reply |

Limits: message content must be non-empty and at most 4000 characters.

#### Server → sender (unicast, turn-scoped)

| Type | Payload | Meaning |
|---|---|---|
| `typing` | `{"type":"typing","sender":"bot"}` | Bot is thinking. |
| `bot_token` | `{"type":"bot_token","content":"..."}` | One streamed token. |
| `bot_done` | `{"type":"bot_done","message":MessageOut\|null}` | Final persisted bot message. |
| `bot_error` | `{"type":"bot_error","detail":"..."}` | Turn failed; nothing persisted. |
| `error` | `{"type":"error","detail":"..."}` | Bad frame or permission problem. |
| `pong` | `{"type":"pong"}` | Heartbeat reply. |

#### Server → room (broadcast)

| Type | Payload | Meaning |
|---|---|---|
| `passenger_message` | `{"type":"passenger_message","message":MessageOut}` | Passenger turn; sender dedupes by id. |
| `agent_message` | `{"type":"agent_message","message":MessageOut}` | Agent reply. |
| `bot_message` | `{"type":"bot_message","message":MessageOut}` | Bot message for other room members. |
| `typing` | `{"type":"typing","sender":"passenger"\|"agent"}` | Typing indicator. |
| `escalated` | `{"type":"escalated","reason":"..."}` | Conversation entered the queue. |
| `resolved` | `{"type":"resolved","escalation_id":"..."}` | Agent closed the conversation. |

A soft-escalation offer is an ordinary bot message whose `message.actions`
contains `[{label, value}]`. Clicking a chip sends `value` back as a normal
`user_message`; there is no dedicated frame.

Heartbeat: the server sends `{"type":"ping"}` every 30 seconds. Clients should
reply with `pong`.

### Dashboard channel

`/ws/dashboard?token=<agent-jwt>`

The dashboard socket is receive-only apart from heartbeats.

#### Client → server

| Type | Payload |
|---|---|
| `ping` | `{"type":"ping"}` |
| `pong` | `{"type":"pong"}` |

#### Server → dashboard

| Type | Payload | Meaning |
|---|---|---|
| `queue_update` | `{"type":"queue_update","action":"created"\|"claimed"\|"released"\|"resolved","escalation_id":"...","conversation_id":"..."}` | Refresh the queue. |
| `ping` | `{"type":"ping"}` | Heartbeat. |
| `pong` | `{"type":"pong"}` | Heartbeat reply. |

Invalid or missing agent tokens close with `4401`.

## Payment flow summary

```text
POST /chat/conversations
POST /chat/conversations/{id}/messages        -> create_booking returns payment_id
   bot message renders PaymentCard(payment_id)
POST /chat/payments/{id}/phone                -> STK push + poller
   Daraja -> POST /daraja/callback            (or background polling)
   success -> booking paid + queue escalation
POST /agents/escalations/{id}/claim           (agent)
POST /agents/payments/{id}/send-ticket        (agent) -> ticketed + e-ticket
POST /chat/payments/{id}/email                (optional passenger email)
```

## MCP

The MCP servers are **stdio**, not HTTP, and therefore are not part of this REST
API. See [system-architecture.md](system-architecture.md#6-mcp-servers-and-tools)
for the tool list and launch commands.
