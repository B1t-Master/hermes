# hermes: Agentic RAG Support Bot (Kenya Airways prototype)

A RAG-based passenger Q&A assistant with sentiment routing, human escalation,
and function calling for flight operations. FastAPI backend, React frontend,
LangGraph orchestration. All inference runs on CPU: self-hosted embeddings +
sentiment, streaming LLM via Groq/DeepSeek.

## Stack

- **Backend:** FastAPI, LangGraph, SQLAlchemy (async) + PostgreSQL/pgvector (local)
- **Inference:** `sentence-transformers` (`bge-small-en-v1.5`, CPU) · sentiment
  `twitter-roberta-base-sentiment-latest` (CPU)
- **LLM:** Groq `openai/gpt-oss-120b` (primary) / `openai/gpt-oss-20b` (fallback)
  via OpenAI-compatible API; DeepSeek configured as backup
- **Auth:** JWT (`PyJWT` + `bcrypt`): passengers (email+password or anonymous) and agents
- **Frontend:** React (Vite): passenger chat widget + human-agent dashboard

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

## Architecture

Passenger query → **sentiment + intent** analysis → **Router agent** →

- routine → retrieve relevant chunks from pgvector → **Q&A agent** generates a grounded, streaming answer; or
- escalate → conversation summary routed to the **human-agent dashboard**.
- function call → **rebook_flight** tool executes mock rebooking via KQ API stub.

Flight-status questions are detected deterministically and enriched with live
(mock) data from the MCP flight-delay service before answering.

Escalation triggers: sentiment < -0.75 on 2 consecutive turns, high-risk intent
(compensation/safety/emergency), or more than 3 unresolved turns.

**Multi-passenger isolation:** every conversation/message/escalation is scoped to
a `passenger_id`; the Q&A agent builds context only from that passenger's
history, so no context leaks between sessions.

## Project layout

```
app/            FastAPI backend (routes, models, auth, agents)
ingest/         Document ingestion + refresh (PDFs, KQ FAQ scrape)
services/       Mock flight-delay library + standalone MCP server
frontend/       React app (chat widget + agent dashboard)
tests/          Live integration/smoke harnesses (e2e WS, MCP flights)
scripts/        One-off tooling (eval, etc.)
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

## Mock flight delays (MCP server)

There is no real KQ flight data in this prototype, so flight-status questions
are answered from a simulated schedule (`services/flight_delays.py`): a small
table of invented routes and times, with delays rolled from a date-seeded RNG.
Results are **stable for a whole calendar day** and regenerate the next day.

The same functions are exposed as a standalone **MCP server** so any MCP client
(Claude Desktop, IDEs, scripting) can use them:

```bash
.venv/bin/python -m services.flights_mcp     # stdio transport by default
```

Tools: `get_flight_status(flight_no, date?)` and
`list_delays(route?, date?, min_delay?)`.

The QA pipeline calls the same underlying functions in-process (`flight_lookup`
graph node), so the bot and every MCP client see identical answers. Smoke test:
`.venv/bin/python tests/mcp_flights_smoke.py`

## Cost notes

- Embeddings + sentiment run on CPU inside the backend (free).
- Retain the managed LLM API route; self-hosting a GPU only breaks even
  around ~12M output tokens/month, far above prototype volume.
- pgvector replaces a dedicated vector DB (e.g. Pinecone) for free at this scale.

## Evaluation

```bash
python -m scripts.eval   # runs faithfulness, citation precision, latency benchmarks
```
Produces HTML report at `eval/report.html`.

## Differentiation roadmap

See `todo.md` for the sprint plan: rebook function calling, eval harness,
Telegram adapter, agent assist drafts, voice widget, auto-translate.
