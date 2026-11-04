# hermes: Agentic RAG Support Bot (Kenya Airways prototype)

A RAG-based passenger support assistant with sentiment routing, human
escalation, and function calling for flight operations. Ships as a FastAPI
backend, a React frontend (passenger chat widget + agent dashboard), and a
Telegram channel. All local inference runs on CPU: self-hosted embeddings and
sentiment, with a streaming LLM via Groq or DeepSeek.

## Stack

- **Backend:** FastAPI, LangGraph, SQLAlchemy (async) + PostgreSQL/pgvector (local)
- **Inference:** `sentence-transformers` (`bge-small-en-v1.5`, CPU) · sentiment
  `twitter-roberta-base-sentiment-latest` (CPU)
- **LLM:** Groq `openai/gpt-oss-120b` (primary) / `openai/gpt-oss-20b` (fallback)
  via OpenAI-compatible API; DeepSeek configured as backup
- **Auth:** JWT (`PyJWT` + `bcrypt`): passengers (email+password or anonymous) and agents
- **Frontend:** React (Vite): passenger chat widget + human-agent dashboard
- **Channels:** web chat widget (WebSocket streaming) + Telegram (long polling)
- **Email:** Resend for rebooking confirmations, with a local outbox fallback
- **Flight data:** SQLite schedule (`.flights.sqlite`) served through a mock MCP server

## Docker quick start

With Docker Engine and the Docker Compose plugin installed:

```bash
cp .env.example .env  # Skip if you already have a configured .env
# Set GROQ_API_KEY (or DEEPSEEK_API_KEY) in .env.
docker compose up --build -d
```

Open http://localhost:5173 for the frontend. The backend API documentation is
at http://localhost:8000/docs. Compose starts PostgreSQL 16 with pgvector,
waits for the database and backend health checks, and serves the built React
app through Nginx, including API and WebSocket forwarding.

The first build installs the CPU ML dependencies; the first backend startup
downloads the embedding and sentiment models and can take several minutes.
Follow progress with `docker compose logs -f backend`. The containers need
internet access for model downloads and external LLM requests.

The knowledge base is populated automatically on every start by the one-shot
`kb-init` service. It syncs the bundled documents from the image into the
writable `knowledge_sources` volume, then parses and embeds every source.
Unchanged documents are skipped by content hash, so subsequent starts are
cheap. If a remote URL is unavailable, it logs a warning and the API still
starts with whatever was ingested. Follow its progress with:

```bash
docker compose logs -f kb-init
```

To force a full re-embed, or rebuild the corpus from scratch:

```bash
docker compose run --rm kb-init python -m ingest.run --embed --force
docker compose run --rm kb-init python -m ingest.run --embed --reset
```

The backend reads your existing `.env`, with `DATABASE_URL` overridden to use
the Compose database. Secrets are supplied at runtime and excluded from image
builds. PostgreSQL data, the flight SQLite database, email outbox, model cache,
and knowledge source uploads persist in named volumes. Existing host database
contents are not imported automatically. Bundled source documents are refreshed
from the image on every start; files uploaded through the dashboard stay in the
volume.

```bash
docker compose down       # Stop services; keep stored data
docker compose up -d      # Start again
```

Ports 5173 and 8000 are bound to localhost. These defaults are intended for
local use; change the example credentials in `.env` before sharing access.

## Quick start

1. **Prerequisites**: Local PostgreSQL 16+ with `pgvector` extension
   ```bash
   # Ubuntu 24.04+
   sudo apt-get update && sudo apt-get install -y postgresql postgresql-16-pgvector
   sudo -u postgres psql -c "CREATE DATABASE hermes;"
   sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
   ```

2. **Configure**:
   ```bash
   cp .env.example .env
   # Edit .env: fill in GROQ_API_KEY (or DEEPSEEK_API_KEY)
   # DATABASE_URL defaults to: postgresql+asyncpg://postgres:postgres@localhost:5432/hermes
   # Optional: TELEGRAM_BOT_TOKEN, RESEND_API_KEY
   ```

3. **Backend**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

4. **Ingest data** (PDFs in `data_sources/` + KQ FAQ URLs):
   ```bash
   python -m ingest.run --embed   # CPU embeddings (torch + sentence-transformers required)
   ```

5. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open http://localhost:5173

The flight schedule lands in `.flights.sqlite` on first use; no extra setup.

## Architecture

<img src="docs/architecture.png" alt="hermes architecture" width="720" />

Passenger query -> **sentiment + intent** analysis -> **Router agent** ->

- routine -> retrieve relevant chunks from pgvector -> **Q&A agent** generates a
  grounded, streaming answer; or
- escalate -> conversation summary routed to the **human-agent dashboard**; or
- rebook -> **rebook_flight** function calling (see below); or
- flight status -> enriched with live (mock) data from the flight service
  before answering; or
- lost/delayed baggage -> in-process guidance (baggage claim, Lost & Found
  desk, WorldTracer tracking) followed by a human-agent offer.

Escalation triggers: sentiment below -0.75 on 2 consecutive turns, high-risk
intent (compensation/safety/emergency), or more than 3 unresolved turns.
Safety and an explicit "talk to a human" escalate immediately — and the bot
still sends an acknowledgement, so the passenger is never left staring at
silence. Every other trigger is soft: the bot answers the question first and
attaches a "Connect me to a human" quick reply *beside* the answer, and only
hands off after `SOFT_ESCALATION_MAX_OFFERS` unanswered offers.
`ESCALATE_ON_COMPENSATION` (default `false`) decides whether a compensation
or refund request is answered-with-an-offer or escalated outright.

**Multi-passenger isolation:** every conversation, message, and escalation is
scoped to a `passenger_id`; the Q&A agent builds context only from that
passenger's history, so no context leaks between sessions.

## Passenger memory

Slots the passenger has given (`app/agents/facts.py`) persist on
`Conversation.facts` and are injected into every later turn as the *known
passenger context*: booking reference, contact email and phone, travel date,
flight numbers, route, bag-tracking reference, and the rebooking confirmation
reference.

Two rules keep the memory honest:

- **Only the passenger's own words fill a slot.** The bot may fill an empty
  slot from its own messages, but it never overwrites something the passenger
  said, and a *confirmation* reference (`HRM-…`) is kept apart from the
  passenger's *booking* reference.
- **A successful rebooking is authoritative.** The tool result (ticket, new
  confirmation reference, flight, date, route) is written straight into
  memory, so "what's my ticket number?" after a rebooking is answered from
  the booking instead of being re-parsed out of the bot's prose.

A booking reference is what most passengers call their ticket number, so
"what's my ticket number?" is answered with the reference on file rather than
asking for a 13-digit e-ticket number the passenger already gave.

## Booking & M-Pesa payment (card-first)

The booking flow never asks for an M-Pesa number in chat. The bot collects the
route, date and flight, then `create_booking` creates a **held booking plus a
pending payment row** and returns a `payment_id`. It does **not** contact
M-Pesa. The bot reply carries that id, and the chat renders an **M-Pesa payment
card** under the first message that references it.

The card collects the number and posts it browser → backend at
`POST /chat/payments/{id}/phone`. That endpoint calls
`app.agents.payments.start_payment()`, which normalises the number, sends the
STK push and schedules a background poller. Keeping every push on this path
means the poller and the success handler always run; the number never enters
the LLM prompt.

After payment succeeds, the same card offers an optional e-ticket email field.
`POST /chat/payments/{id}/email` stores it on the booking without issuing the
ticket; the agent's ticket send emails a copy when it is set. If the ticket was
already issued before the address was supplied, the endpoint emails the existing
e-ticket as a copy. The address never enters the chat or the LLM prompt.

Payment resolution has two paths:

- **Callback** — `POST /daraja/callback` (the path must not contain "mpesa")
  resolves the payment by Daraja checkout id.
- **Polling** — the background poller queries Daraja every
  `MPESA_STK_QUERY_INTERVAL_SECONDS` for `MPESA_STK_QUERY_MAX_ATTEMPTS`
  (defaults: 2s × 60 = 120s). This is the reliable path when no public callback
  URL is configured. If the poll budget expires first, a **late `success`
  callback still overrides the stored `timeout`**.

A success marks the booking paid, writes a bot "Payment received" notice, and
escalates the conversation to the human queue for ticketing. The notice carries
no `payment_id`, so it does not add a second card. Only the initial booking
message owns the card in both the passenger and agent transcripts. The agent
issues the e-ticket from the dashboard; that flips the booking to `ticketed`
and a `payment_update` refreshes the open card in place.

`MPESA_PROVIDER=simulated` exercises the whole flow without a phone: a prompt
resolves after `MPESA_SIMULATE_DELAY_SECONDS`, numbers ending `0000` fail and
`1111` cancel. `auto` uses Daraja when the four credentials are set.

## Rebooking via function calling

Rebooking is offered two ways. The bot surfaces the official KQ self-service
page (`SELF_SERVICE_REBOOKING_URL`) **and** offers to do it in chat, then
rebooks through a short conversation and never asks for a flight number (it
picks one):

1. Passenger asks to rebook. The bot asks for the booking reference, PNR, or
   13-digit e-ticket number, once.
2. If the passenger named a date, that date is used: natural language is
   understood (`tomorrow`, `tomorrow afternoon`, `day after tomorrow`, `next
   Friday`, or a calendar date like `2025-04-20`). If they did not, the bot
   rebooks them onto the **next available flight** — matching KQ's automatic
   rebooking of disrupted flights — and says which flight and date they were
   moved to, so they can ask for a different one.
3. With the reference in hand, the `rebook_flight` tool runs: it selects a
   flight from `.flights.sqlite`, records the rebooking in a `rebookings` table
   in the same database, and returns a confirmation reference (`HRM-…`).
4. If the booking reference has an email address on file (the
   `rebook_email_map` table), a confirmation is emailed via Resend. An unknown
   reference still rebooks, but no email is sent and the result says so.

The rebooking conversation stays on the tool path across turns: a follow-up
that contains only a booking reference (`HRM-9B4770`) or a date
(`tomorrow afternoon`) is recognised as part of the flow, not a new question.

## Baggage guidance

Lost, delayed, and unclaimed-bag questions are answered deterministically and
in-process (`app/agents/baggage.py`) — no external system, no LLM invention.
The bot lists the three places to look and then asks whether the passenger
wants a human agent:

1. **Arrival baggage claim** — JKIA Terminal 1A (`BAGGAGE_ARRIVAL_CLAIM`).
2. **Airport Lost & Found desk** — JKIA Terminal 1A (`BAGGAGE_LOST_AND_FOUND`).
3. **WorldTracer online tracking** — `WORLDTRACER_URL`, using the bag tag or
   baggage-receipt reference.

The answer carries two quick-reply chips ("Connect me to a human" /
"It's something else") so the next step is one click; choosing the human chip
routes through the normal hard escalation.

## Telegram channel

A Telegram adapter runs inside the backend when `TELEGRAM_BOT_TOKEN` is set. It
uses long polling (no public webhook needed). Each Telegram chat maps to one
passenger and one active conversation; agent replies from the dashboard are
relayed back to the chat through the in-process conversation hub.

## Email delivery

Rebooking confirmations go out through **Resend** (`services/email_real.py`).
With no `RESEND_API_KEY`, the sender falls back to a local outbox
(`EMAIL_OUTBOX_DIR`), so local runs and the test suite need no credentials.

## Flight schedule (mock) + MCP server

There is no real KQ flight data in this prototype, so flight questions are
answered from a simulated schedule (`services/flight_delays.py`): a small table
of invented routes and times, with delays rolled from a date-seeded RNG.
Results are **stable for a whole calendar day** and regenerate the next day.
Rebookings are recorded separately, so they never disturb the daily roll.

The same functions are exposed as a standalone **MCP server** so any MCP client
(Claude Desktop, IDEs, scripting) can use them:

```bash
.venv/bin/python -m services.flights_mcp     # stdio transport by default
```

Tools: `get_flight_status(flight_no, date?)` and
`list_delays(route?, date?, min_delay?)`.

The bot queries these tools through an MCP stdio client
(`app/agents/mcp_client.py`), so it is a real MCP consumer rather than a second
in-process caller. If the server is unreachable the client falls back to the
in-process library, so a flight question never takes the whole turn down. Set
`FLIGHTS_MCP_ENABLED=false` to skip MCP entirely. Smoke test:
`.venv/bin/python tests/mcp_flights_smoke.py` (also covered by
`tests/test_mcp_client.py` under pytest).

## Project layout

```
app/            FastAPI backend (routes, models, auth, agents)
app/agents/     LangGraph nodes, LLM client, tool calling, turn orchestration
app/channels/   Channel adapters (Telegram long polling)
services/       Flight-delay + rebooking mock libraries, email senders, MCP server
ingest/         Document ingestion + refresh (PDFs, KQ FAQ scrape)
evals/          Golden-set evaluation harness (dataset, runner, scoring)
frontend/       React app (chat widget + agent dashboard)
tests/          Unit tests (pytest) + live smoke harnesses (e2e WS, MCP flights)
data_sources/   Local documents fed to the vector store
```

## Data & refresh

Sources: SAA Conditions of Carriage, KQ self-rebooking FAQ (PDFs), and scraped
KQ FAQ web pages. When a source changes, re-run:

```bash
python -m ingest.run --embed --force
```

Refresh uses content hashing: only chunks whose source changed are
re-chunked/re-embedded; unchanged sources are left untouched.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q     # unit suite
.venv/bin/ruff check app services tests  # lint (the CI gate)
```

Live harnesses (need the backend running):
`.venv/bin/python tests/e2e_ws.py` and `.venv/bin/python tests/mcp_flights_smoke.py`.

## Evaluation

The golden-set harness drives the live backend over HTTP and scores each answer
deterministically (must-include / must-avoid terms, escalation, and citation
checks). There is no LLM judge and no A/B comparison: a run reflects the
current pipeline only.

```bash
.venv/bin/python -m evals.run                  # full golden set
.venv/bin/python -m evals.run --fast           # quick subset
.venv/bin/python -m evals.run --concurrency 8  # more parallel cases
.venv/bin/python -m evals.run --only rebook-full-flow
```

Results are written to `evals/report.json`; the process exits non-zero when any
case fails, so CI can gate on it. The backend must be running and the LLM quota
must be available (back-to-back runs can hit provider rate limits).

## Branches

- `main` is the stable line.
- `dev` is the integration branch for ongoing work.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit conventions,
and the pull-request checklist.

## Cost notes

- Embeddings + sentiment run on CPU inside the backend (free).
- Retain the managed LLM API route; self-hosting a GPU only breaks even
  around ~12M output tokens/month, far above prototype volume.
- pgvector replaces a dedicated vector DB (e.g. Pinecone) for free at this scale.

## Differentiation roadmap

Shipped: rebooking via function calling (with real email confirmations),
golden-set eval harness, Telegram adapter. Remaining: agent assist drafts,
browser voice widget, auto-translate.
