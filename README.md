# hermes: Agentic RAG Support Bot (Kenya Airways prototype)

A RAG-based passenger Q&A assistant with sentiment routing and human escalation.
FastAPI backend, React frontend, LangGraph orchestration. All three inference
models are low-cost: self-hosted CPU embeddings + sentiment, streaming LLM via
Groq/DeepSeek.

## Stack

- **Backend:** FastAPI, LangGraph, SQLAlchemy (async) + PostgreSQL/**pgvector** on Neon
- **Inference:** `sentence-transformers` (`bge-small-en-v1.5`, CPU) · sentiment
  `twitter-roberta-base-sentiment-latest` (CPU)
- **LLM:** Groq `openai/gpt-oss-120b` (primary) / `openai/gpt-oss-20b` (fallback)
  via OpenAI-compatible API; DeepSeek configured as an emergency backup
- **Auth:** JWT (`PyJWT` + `bcrypt`): passengers (email+password or anonymous session) and agents (shared credentials)
- **Frontend:** React (Vite): passenger chat widget + human-agent dashboard

## Quick start

1. Copy `.env.example` → `.env` and fill in `GROQ_API_KEY` (or `DEEPSEEK_API_KEY`) and `DATABASE_URL` (Neon, `enable extension pgvector`).
2. Backend:
   ```bash
   python -m venv .venv && .\.venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
3. Ingest data (PDFs in `data_sources/` + KQ FAQ URLs). By default the CLI writes
   to a local `SqliteStore` (`.hermes_store.sqlite`) so the pipeline runs without
   the ML stack or a network route to Neon:
   ```bash
   python -m ingest.run                      # local sqlite, no embeddings
   python -m ingest.run --embed              # add CPU embeddings (torch + sentence-transformers required)
   python -m ingest.run --store pg --embed   # Neon pgvector (creates schema; needs outbound :5432)
   ```
4. Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Architecture

Passenger query → **sentiment + intent** analysis → **Router agent** →

- routine → retrieve relevant chunks from pgvector → **Q&A agent** generates a grounded, streaming answer; or
- escalate → conversation summary routed to the **human-agent dashboard**.

Escalation triggers: sentiment < -0.75 on 2 consecutive turns, a high-risk
intent (compensation/safety/emergency), or more than 3 unresolved turns.

**Multi-passenger isolation:** every conversation/message/escalation is scoped to
a `passenger_id`; the Q&A agent builds context only from that passenger's
history, so no context leaks between sessions.

## Project layout

```
app/            FastAPI backend (routes, models, auth, agents)
ingest/         Document ingestion + refresh (PDFs, KQ FAQ scrape)
frontend/       React app (chat widget + agent dashboard)
data_sources/   Local documents fed to the vector store
```

## Data & refresh

Sources: SAA Conditions of Carriage, KQ self-rebooking FAQ (PDFs), and scraped
KQ FAQ web pages. When a source changes, re-run:

```bash
python -m ingest.refresh        # same --store / --embed switches as ingest.run
```

Refresh uses content hashing: only chunks whose source changed are
re-chunked/re-embedded; unchanged sources are left untouched.

**Store backends:** `--store auto` picks the Neon Postgres/pgvector store when
`DATABASE_URL` is set and reachable, otherwise falls back to a local `SqliteStore`
for offline/restricted-network development. Vectors are stored SQLite-side as
JSON; in-process cosine search is used during retrieval for the fallback.

## Cost notes

- Embeddings + sentiment run on CPU inside the backend (free).
- Retain the managed LLM API route; self-hosting a GPU only breaks even
  around ~12M output tokens/month, far above prototype volume.
- pgvector replaces a dedicated vector DB (e.g. Pinecone) for free at this scale.

## Future plans

- **WhatsApp integration via Turn.io**: same FastAPI backend gets a webhook
  adapter routing WhatsApp messages through the identical LangGraph pipeline;
  the agent dashboard then also serves Turn.io helpdesk handoffs.
- Scheduled re-ingestion / stale-data automation (currently manual refresh).
- Swap pgvector → Pinecone/Qdrant if the corpus grows past ~1M vectors.
- Real per-agent accounts and password reset for the dashboard.