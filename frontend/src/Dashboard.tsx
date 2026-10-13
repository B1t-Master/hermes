import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { api, ApiError } from "./api";
import { useAuth } from "./auth";
import RichText from "./RichText";
import { formatDate, formatTime } from "./format";
import type {
  AgentInfo,
  IngestReport,
  KnowledgeDocument,
  Message,
  QueueEscalation,
  ServerFrame,
} from "./types";
import { connectWs } from "./ws";
import type { WsHandle, WsStatus } from "./ws";

type Filter = "active" | "pending" | "assigned" | "resolved" | "all";
const FILTERS: Filter[] = ["active", "pending", "assigned", "resolved", "all"];

type View = "queue" | "kb";

function age(iso: string): string {
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  return `${Math.round(mins / 60)}h ago`;
}

export default function Dashboard() {
  const { token, logout } = useAuth();
  const [me, setMe] = useState<AgentInfo | null>(null);
  const [queue, setQueue] = useState<QueueEscalation[]>([]);
  const [filter, setFilter] = useState<Filter>("active");
  const [selected, setSelected] = useState<QueueEscalation | null>(null);
  const [transcript, setTranscript] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [status, setStatus] = useState<WsStatus>("connecting");
  const dashWs = useRef<WsHandle | null>(null);
  const convWs = useRef<WsHandle | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  /* ---- knowledge base tab state -------------------------------------------- */
  const [view, setView] = useState<View>("queue");
  const [kbDocs, setKbDocs] = useState<KnowledgeDocument[]>([]);
  const [kbReport, setKbReport] = useState<IngestReport | null>(null);
  const [kbBusy, setKbBusy] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  const [urlText, setUrlText] = useState("");
  const [embed, setEmbed] = useState(true);
  const [force, setForce] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  /* ---- queue reload (kept fresh via ref so WS callbacks see latest filter) -- */
  const reload = async () => {
    if (!token) return;
    try {
      const list = await api.queue(token, filter);
      setQueue(list);
      setSelected((prev) =>
        prev ? (list.find((i) => i.id === prev.id) ?? prev) : prev,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const reloadRef = useRef(reload);
  reloadRef.current = reload;

  /* ---- knowledge base: load + mutations (all agent-only) -------------------- */
  const loadKb = async () => {
    if (!token) return;
    try {
      setKbDocs(await api.kbList(token));
      setKbError(null);
    } catch (err) {
      setKbError(err instanceof Error ? err.message : String(err));
    }
  };
  const loadKbRef = useRef(loadKb);
  loadKbRef.current = loadKb;

  useEffect(() => {
    if (token && view === "kb") void loadKbRef.current();
  }, [token, view]);

  /** Run a KB mutation, surface its ingest report, then refresh the list. */
  const runKb = async (fn: () => Promise<IngestReport | { ok: boolean } | void>) => {
    setKbBusy(true);
    setKbError(null);
    setKbReport(null);
    try {
      const result = await fn();
      if (result && "summary" in result) setKbReport(result);
      await loadKbRef.current();
    } catch (err) {
      setKbError(err instanceof Error ? err.message : String(err));
    } finally {
      setKbBusy(false);
    }
  };

  const uploadDocs = (e: FormEvent) => {
    e.preventDefault();
    if (!token) return;
    const files = Array.from(fileRef.current?.files ?? []);
    if (files.length === 0) return;
    void runKb(async () => {
      const report = await api.kbUpload(token, files, embed, force);
      if (fileRef.current) fileRef.current.value = "";
      return report;
    });
  };

  const ingestUrls = (e: FormEvent) => {
    e.preventDefault();
    if (!token) return;
    const urls = urlText
      .split(/\s+/)
      .map((u) => u.trim())
      .filter(Boolean);
    if (urls.length === 0) return;
    void runKb(async () => {
      const report = await api.kbIngestUrls(token, urls);
      setUrlText("");
      return report;
    });
  };

  const reingest = (key: string) => {
    if (!token) return;
    void runKb(() => api.kbReingest(token, key));
  };

  const removeSource = (doc: KnowledgeDocument) => {
    if (!token || !doc.source_url) return;
    if (!window.confirm(`Remove "${doc.title}" and all of its chunks?`)) return;
    void runKb(() => api.kbDelete(token, doc.source_url as string));
  };

  const resetKb = () => {
    if (!token) return;
    if (!window.confirm("Delete ALL knowledge base documents and chunks?")) return;
    void runKb(() => api.kbReset(token));
  };

  /* ---- identity + queue fetch ---------------------------------------------- */
  useEffect(() => {
    if (token) api.agentMe(token).then(setMe).catch(() => undefined);
  }, [token]);

  useEffect(() => {
    void reloadRef.current();
  }, [token, filter]);

  /* ---- dashboard WS: live queue updates ------------------------------------ */
  useEffect(() => {
    if (!token) return;
    const handle = connectWs("/ws/dashboard", token, {
      onStatus: setStatus,
      onFrame: (f: ServerFrame) => {
        if (f.type === "queue_update") void reloadRef.current();
      },
    });
    dashWs.current = handle;
    return () => {
      handle.close();
      dashWs.current = null;
    };
  }, [token]);

  /* ---- transcript for the selected escalation ------------------------------- */
  useEffect(() => {
    if (!token || !selected) {
      setTranscript([]);
      return;
    }
    const eid = selected.id;
    api
      .escalationMessages(token, eid)
      .then(setTranscript)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, selected?.id]);

  /* ---- conversation WS: only once the claim is ours ------------------------- */
  const isMine =
    !!selected && !!me && selected.status === "assigned" && selected.agent_id === me.id;

  useEffect(() => {
    if (!token || !selected || !isMine) return;
    const cid = selected.conversation_id;

    const upsert = (msg: Message) => {
      setTranscript((prev) => {
        if (prev.some((m) => m.id === msg.id)) return prev;
        const pendingIdx = prev.findIndex(
          (m) => m.pending && m.sender === msg.sender && m.content === msg.content,
        );
        if (pendingIdx !== -1) {
          const next = [...prev];
          next[pendingIdx] = msg;
          return next;
        }
        return [...prev, msg];
      });
    };

    const handle = connectWs(`/ws/conversations/${cid}`, token, {
      onOpen: () => {
        if (token && selected) {
          api.escalationMessages(token, selected.id).then(setTranscript).catch(() => undefined);
        }
      },
      onFrame: (f: ServerFrame) => {
        switch (f.type) {
          case "passenger_message":
          case "bot_message":
          case "agent_message":
            upsert(f.message);
            // Keep the queue card's preview/turn count in sync while viewing.
            if (f.type === "passenger_message") void reloadRef.current();
            break;
          case "bot_done":
            if (f.message) upsert(f.message);
            break;
          case "resolved":
            void reloadRef.current();
            break;
          case "error":
            setError(f.detail);
            break;
          default:
            break;
        }
      },
    });
    convWs.current = handle;
    return () => {
      handle.close();
      convWs.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, isMine, selected?.id, selected?.conversation_id]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [transcript]);

  /* ---- actions -------------------------------------------------------------- */
  const claim = async (item: QueueEscalation) => {
    if (!token) return;
    setBusyId(item.id);
    setError(null);
    try {
      await api.claim(token, item.id);
      // Reflect the claim immediately (the queue filter may hide the card),
      // then reload so the visible list syncs too. agent_id comes from `me`
      // so the panel is "mine" even when the card drops out of the filter.
      setSelected((prev) =>
        prev && prev.id === item.id
          ? {
              ...prev,
              status: "assigned",
              agent_id: me?.id ?? prev.agent_id,
              agent_name: me?.name ?? prev.agent_name,
            }
          : prev,
      );
      await reload();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setError("Another agent claimed this conversation first.");
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
      await reload();
    } finally {
      setBusyId(null);
    }
  };

  const release = async (item: QueueEscalation) => {
    if (!token) return;
    setBusyId(item.id);
    setError(null);
    try {
      await api.release(token, item.id);
      // Keep the panel honest: we no longer own it, so drop the claim from
      // the selected copy before reloading (this also closes the conv WS,
      // which the server would otherwise reject on the next frame/reconnect).
      setSelected((prev) =>
        prev && prev.id === item.id
          ? { ...prev, status: "pending", agent_id: null, agent_name: null }
          : prev,
      );
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  const resolve = async (item: QueueEscalation) => {
    if (!token) return;
    setBusyId(item.id);
    setError(null);
    try {
      await api.resolve(token, item.id);
      await reload();
      setSelected(null);
      setTranscript([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  const sendReply = (e: FormEvent) => {
    e.preventDefault();
    const content = draft.trim();
    if (!content || !selected || !me) return;
    const optimistic: Message = {
      id: `pending-${crypto.randomUUID()}`,
      conversation_id: selected.conversation_id,
      sender: "agent",
      content,
      sentiment_score: null,
      sentiment_label: null,
      created_at: new Date().toISOString(),
      pending: true,
    };
    setTranscript((prev) => [...prev, optimistic]);
    convWs.current?.send({ type: "agent_message", content });
    setDraft("");
  };

  /* ---- render ---------------------------------------------------------------- */
  return (
    <main className="dashboard">
      <header>
        <h1>hermes ops</h1>
        <div className="header-meta">
          <span className={`badge conn-${status}`}>{status}</span>
          <span className="badge role">{me ? me.name : "agent"}</span>
          <button className="secondary" onClick={logout}>
            Sign out
          </button>
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab${view === "queue" ? " active" : ""}`}
          onClick={() => setView("queue")}
        >
          Support queue
          {queue.length > 0 && <span className="tab-count">{queue.length}</span>}
        </button>
        <button
          className={`tab${view === "kb" ? " active" : ""}`}
          onClick={() => setView("kb")}
        >
          Knowledge base
        </button>
      </nav>

      {view === "queue" && (
        <>
          <div className="filter-bar">
        {FILTERS.map((f) => (
          <button
            key={f}
            className={`chip${filter === f ? " active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
        <span className="queue-count">{queue.length} in view</span>
      </div>

      {error && <p className="notice error">{error}</p>}

      <section>
        {queue.length === 0 && <p className="placeholder">Queue is clear.</p>}
        <div className="queue-grid">
          {queue.map((item) => (
            <article
              key={item.id}
              className={`queue-card status-${item.status}${
                selected?.id === item.id ? " selected" : ""
              }`}
              onClick={() => setSelected(item)}
            >
              <header>
                <span className={`badge status-${item.status}`}>{item.status}</span>
                <strong>{item.passenger_name}</strong>
                <time>{age(item.created_at)}</time>
              </header>
              <p className="reason">{item.reason}</p>
              {item.last_message && <p className="preview">“{item.last_message}”</p>}
              <footer onClick={(e) => e.stopPropagation()}>
                <span className="turn-count">{item.turn_count} turns</span>
                {item.agent_name && (
                  <span className={`badge owner${item.agent_id === me?.id ? " mine" : ""}`}>
                    {item.agent_id === me?.id ? "you" : item.agent_name}
                  </span>
                )}
                <span className="card-actions">
                  {item.status === "pending" && (
                    <button
                      className="claim"
                      disabled={busyId === item.id}
                      onClick={() => claim(item)}
                    >
                      Claim
                    </button>
                  )}
                  {item.status === "assigned" && item.agent_id === me?.id && (
                    <>
                      <button
                        className="secondary"
                        disabled={busyId === item.id}
                        onClick={() => release(item)}
                      >
                        Release
                      </button>
                      <button
                        className="resolve"
                        disabled={busyId === item.id}
                        onClick={() => resolve(item)}
                      >
                        Resolve
                      </button>
                    </>
                  )}
                </span>
              </footer>
            </article>
          ))}
        </div>
      </section>

      {selected && (
        <section className="transcript-panel">
          <header>
            <h2>
              <span className={`badge status-${selected.status}`}>{selected.status}</span>{" "}
              {selected.passenger_name}
            </h2>
            <button className="secondary" onClick={() => setSelected(null)}>
              Close
            </button>
          </header>
          <p className="reason">{selected.reason}</p>

          <div className="chat-body" ref={scrollRef}>
            {transcript.length === 0 && <p className="placeholder">No messages.</p>}
            {transcript.map((m) => (
              <div key={m.id} className={`message ${m.sender}${m.pending ? " pending" : ""}`}>
                <div className="meta">
                  <strong>{m.sender === "bot" ? "hermes" : m.sender}</strong>
                  <span>{formatTime(m.created_at)}</span>
                </div>
                <div className="body">
                  <RichText text={m.content} />
                </div>
              </div>
            ))}
          </div>

          {isMine ? (
            <form className="composer" onSubmit={sendReply}>
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Reply as support agent…"
              />
              <button type="submit" disabled={!draft.trim()}>
                Send
              </button>
            </form>
          ) : (
            <p className="notice warn">
              {selected.status === "pending"
                ? "Claim this conversation to reply."
                : "Read-only: another agent owns this conversation."}
            </p>
          )}
        </section>
      )}
        </>
      )}

      {view === "kb" && (
        <section className="kb-panel">
          <header>
            <h2>Knowledge base</h2>
            <span className="queue-count">{kbDocs.length} sources</span>
            <button className="secondary" onClick={() => void loadKbRef.current()} disabled={kbBusy}>
              Refresh
            </button>
          </header>

          {kbError && <p className="notice error">{kbError}</p>}
          {kbReport && <p className="notice ok">Ingest complete: {kbReport.summary}</p>}

          <div className="kb-grid">
            <form className="kb-card" onSubmit={uploadDocs}>
              <h3>Upload documents</h3>
              <p className="hint">Accepted: PDF or DOCX. Files are stored in data_sources/.</p>
              <input ref={fileRef} type="file" multiple accept=".pdf,.docx" />
              <label className="check">
                <input type="checkbox" checked={embed} onChange={(e) => setEmbed(e.target.checked)} />
                Embed for semantic search
              </label>
              <label className="check">
                <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                Re-ingest even if unchanged
              </label>
              <button type="submit" disabled={kbBusy}>
                Upload &amp; ingest
              </button>
            </form>

            <form className="kb-card" onSubmit={ingestUrls}>
              <h3>Ingest from URLs</h3>
              <p className="hint">One URL per line. The page text is scraped and chunked.</p>
              <textarea
                value={urlText}
                onChange={(e) => setUrlText(e.target.value)}
                placeholder={"https://www.kenya-airways.com/...\nhttps://..."}
                rows={6}
              />
              <button type="submit" disabled={kbBusy || !urlText.trim()}>
                Ingest URLs
              </button>
            </form>
          </div>

          <div className="kb-list">
            {kbDocs.length === 0 && <p className="placeholder">No sources yet.</p>}
            {kbDocs.map((doc) => (
              <article key={doc.id} className="kb-item">
                <header>
                  <span className="badge">{doc.doc_type}</span>
                  <strong>{doc.title}</strong>
                  <time>added {formatDate(doc.created_at)}</time>
                </header>
                <p className="kb-meta">
                  {doc.fragment_count ?? 0} chunks · {doc.source_url ?? "unknown source"}
                </p>
                <p className="kb-meta">last fetched {formatDate(doc.last_fetched)}</p>
                <footer>
                  <button
                    className="secondary"
                    disabled={kbBusy || !doc.source_url}
                    onClick={() => doc.source_url && reingest(doc.source_url)}
                  >
                    Re-ingest
                  </button>
                  <button
                    className="danger"
                    disabled={kbBusy || !doc.source_url}
                    onClick={() => removeSource(doc)}
                  >
                    Delete
                  </button>
                </footer>
              </article>
            ))}
          </div>

          <div className="kb-danger">
            <button className="danger" onClick={resetKb} disabled={kbBusy || kbDocs.length === 0}>
              Reset all knowledge
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
