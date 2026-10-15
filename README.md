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
  before answering.

Escalation triggers: sentiment below -0.75 on 2 consecutive turns, high-risk
intent (compensation/safety/emergency), or more than 3 unresolved turns.

**Multi-passenger isolation:** every conversation, message, and escalation is
scoped to a `passenger_id`; the Q&A agent builds context only from that
passenger's history, so no context leaks between sessions.

## Rebooking via function calling

The bot rebooks through a two-step conversation and never asks for a flight
number (it picks one):

1. Passenger asks to rebook. The bot asks for the booking reference or
   e-ticket number, once.
2. Once it has the reference, it asks for the new travel date. Natural
   language is understood: `tomorrow`, `tomorrow afternoon`, `day after
   tomorrow`, `next Friday`, or a calendar date like `2025-04-20`.
3. With both in hand, the `rebook_flight` tool runs: it selects a flight from
   `.flights.sqlite`, records the rebooking in a `rebookings` table in the same
   database, and returns a confirmation reference.
4. If the booking reference has an email address on file (the
   `rebook_email_map` table), a confirmation is emailed via Resend. An unknown
   reference still rebooks, but no email is sent and the result says so.

The rebooking conversation stays on the tool path across turns: a follow-up
that contains only a booking reference (`HRM-9B4770`) or a date
(`tomorrow afternoon`) is recognised as part of the flow, not a new question.

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
