import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { api } from "./api";
import { useAuth } from "./auth";
import RichText from "./RichText";
import Wings from "./Wings";
import { formatTime } from "./format";
import { playAgentJoinedCue, playEscalationCue } from "./sound";
import type { Conversation, Message, ServerFrame } from "./types";
import { connectWs } from "./ws";
import type { WsHandle, WsStatus } from "./ws";

export default function ChatWidget() {
  const { token, role, logout } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState(false);
  const [agentTyping, setAgentTyping] = useState(false);
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const wsRef = useRef<WsHandle | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const agentTypingTimer = useRef<number | null>(null);
  const toastTimer = useRef<number | null>(null);
  /** Guard so the "agent joined" cue fires once per escalation. */
  const agentAnnouncedRef = useRef(false);
  /** Current conversation id, read by stale-socket frame handlers. */
  const activeIdRef = useRef<string | null>(null);
  const [csat, setCsat] = useState<Record<string, number>>({});

  const showToast = (text: string) => {
    setToast(text);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 8000);
  };

  /* ---- initial load: resume the newest open conversation or create one ---- */
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    (async () => {
      try {
        let list = await api.listConversations(token);
        if (cancelled) return;
        const open = list.find((c) => c.status !== "resolved");
        let target = open;
        if (!target) {
          target = await api.createConversation(token);
          list = [target, ...list];
        }
        if (cancelled) return;
        setConversations(list);
        setActiveId(target.id);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [token]);

  /* ---- history + websocket for the active conversation --------------------- */
  useEffect(() => {
    if (!token || !activeId) return;
    const cid = activeId;
    activeIdRef.current = cid;
    setMessages([]);
    setStreaming(null);
    setBotThinking(false);
    setAgentTyping(false);
    setError(null);
    agentAnnouncedRef.current = false;

    api.listMessages(token, cid).then(setMessages).catch(() => {
      /* resync on reconnect covers transient failures */
    });

    const refreshConversations = () => {
      if (!token) return;
      api.listConversations(token).then(setConversations).catch(() => undefined);
    };

    const flagAgentTyping = () => {
      setAgentTyping(true);
      if (agentTypingTimer.current) window.clearTimeout(agentTypingTimer.current);
      agentTypingTimer.current = window.setTimeout(() => setAgentTyping(false), 4000);
    };

    const upsert = (msg: Message) => {
      setMessages((prev) => {
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

    // Sidebar status/order syncs from the DB (turn counts included); the
    // local bounce gives instant feedback while the fetch is in flight.
    const setConvStatus = (id: string, status: Conversation["status"]) =>
      setConversations((list) =>
        list.map((c) => (c.id === id ? { ...c, status } : c)),
      );

    const onFrame = (f: ServerFrame) => {
      // Frames from a socket that was torn down after switching chats must
      // not leak into the newly active conversation.
      if (activeIdRef.current !== cid) return;
      switch (f.type) {
        case "typing":
          if (f.sender === "bot") setBotThinking(true);
          if (f.sender === "agent") flagAgentTyping();
          break;
        case "bot_token":
          setBotThinking(false);
          setStreaming((prev) => (prev ?? "") + f.content);
          break;
        case "bot_done":
          setBotThinking(false);
          setStreaming(null);
          if (f.message) upsert(f.message);
          refreshConversations();
          break;
        case "bot_error":
          setBotThinking(false);
          setStreaming(null);
          setError(f.detail);
          break;
        case "passenger_message":
          upsert(f.message);
          refreshConversations();
          break;
        case "bot_message":
          setBotThinking(false);
          setStreaming(null);
          upsert(f.message);
          break;
        case "agent_message":
          setAgentTyping(false);
          upsert(f.message);
          if (!agentAnnouncedRef.current) {
            agentAnnouncedRef.current = true;
            playAgentJoinedCue();
            showToast("A human agent has joined the chat.");
          }
          break;
        case "escalated":
          setConvStatus(cid, "escalated");
          refreshConversations();
          playEscalationCue();
          showToast("Connecting you to a human agent — they'll reply right here.");
          break;
        case "resolved":
          setConvStatus(cid, "resolved");
          refreshConversations();
          break;
        case "error":
          setError(f.detail);
          break;
        default:
          break;
      }
    };

    const handle = connectWs(`/ws/conversations/${cid}`, token, {
      onStatus: (s) => {
        setStatus(s);
        // A stalled/closed socket must not leave ghost "thinking" or a
        // half-built stream on screen.
        if (s === "reconnecting" || s === "closed") {
          setStreaming(null);
          setBotThinking(false);
        }
      },
      onOpen: () => {
        // After every (re)connect, resync from REST: the server does not
        // replay frames missed while disconnected.
        api.listMessages(token, cid).then(setMessages).catch(() => undefined);
        refreshConversations();
      },
      onFrame,
    });
    wsRef.current = handle;
    return () => {
      handle.close();
      wsRef.current = null;
      if (agentTypingTimer.current) window.clearTimeout(agentTypingTimer.current);
      if (toastTimer.current) window.clearTimeout(toastTimer.current);
    };
  }, [token, activeId]);

  /* ---- auto-scroll --------------------------------------------------------- */
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, botThinking]);

  const active = conversations.find((c) => c.id === activeId) ?? null;
  const resolved = active?.status === "resolved";

  const send = (e: FormEvent) => {
    e.preventDefault();
    const content = draft.trim();
    if (!content || !activeId || resolved) return;
    const optimistic: Message = {
      id: `pending-${crypto.randomUUID()}`,
      conversation_id: activeId,
      sender: "passenger",
      content,
      sentiment_score: null,
      sentiment_label: null,
      created_at: new Date().toISOString(),
      pending: true,
    };
    setMessages((prev) => [...prev, optimistic]);
    wsRef.current?.send({ type: "user_message", content });
    setDraft("");
    setError(null);
  };

  const newConversation = async () => {
    if (!token) return;
    try {
      const conv = await api.createConversation(token);
      setConversations((list) => [conv, ...list]);
      setActiveId(conv.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const rateActive = async (rating: number) => {
    if (!token || !activeId) return;
    try {
      await api.csat(token, activeId, rating);
      setCsat((prev) => ({ ...prev, [activeId]: rating }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <main className="chat">
      <header>
        <h1>
          <Wings /> hermes
        </h1>
        <div className="header-meta">
          <span className={`badge conn-${status}`}>{status}</span>
          <span className="badge role">{role}</span>
          <button className="secondary" onClick={logout}>
            Sign out
          </button>
        </div>
      </header>

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}

      <div className="chat-layout">
        <aside className="conv-list">
          <button onClick={newConversation}>+ New chat</button>
          {conversations.map((c) => (
            <button
              key={c.id}
              className={`conv-item${c.id === activeId ? " active" : ""}`}
              onClick={() => setActiveId(c.id)}
            >
              <span className={`badge status-${c.status}`}>{c.status}</span>
              <span className="conv-turns">{c.turn_count} turns</span>
            </button>
          ))}
        </aside>

        <section className="chat-thread">
          {active?.status === "escalated" && (
            <p className="notice warn">
              Escalated{active.escalation_reason ? `: ${active.escalation_reason}` : ""} — a
              human agent has been notified and will reply right here.
            </p>
          )}
          {resolved && (
            <p className="notice ok">
              This conversation was resolved. Start a new chat to continue.
            </p>
          )}
          {error && <p className="notice error">{error}</p>}

          {resolved && activeId && (
            <div className="csat">
              {csat[activeId] ? (
                <p className="placeholder">Thanks for the feedback: {csat[activeId]}/5.</p>
              ) : (
                <>
                  <p>How did we do?</p>
                  <div className="stars">
                    {[1, 2, 3, 4, 5].map((n) => (
                      <button
                        key={n}
                        type="button"
                        className="star"
                        aria-label={`${n} star${n > 1 ? "s" : ""}`}
                        onClick={() => rateActive(n)}
                      >
                        ★
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          <div className="chat-body" ref={scrollRef}>
            {messages.length === 0 && streaming === null && !botThinking && (
              <p className="placeholder">Ask about baggage, check-in, delays, refunds…</p>
            )}
            {messages.map((m) => (
              <div
                key={m.id}
                className={`message ${m.sender}${m.pending ? " pending" : ""}`}
              >
                <div className="meta">
                  <strong>
                    {m.sender === "bot" ? (
                      <>
                        <Wings size={15} /> hermes
                      </>
                    ) : (
                      m.sender
                    )}
                  </strong>
                  <span>{formatTime(m.created_at)}</span>
                  {m.sentiment_label && m.sender === "passenger" && (
                    <span className={`badge ${m.sentiment_label}`}>
                      {m.sentiment_label}
                    </span>
                  )}
                </div>
                <div className="body">
                  <RichText text={m.content} />
                </div>
                {m.sources && m.sources.length > 0 && (
                  <div className="sources">
                    {m.sources.map((s, i) => (
                      <span
                        key={`${s.title}-${i}`}
                        className="source"
                        title={s.url ?? undefined}
                      >
                        {s.title}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {streaming !== null && (
              <div className="message bot">
                <div className="meta">
                  <strong>hermes</strong>
                </div>
                <div className="body streaming">
                  <RichText text={streaming} />
                  <span className="caret">▌</span>
                </div>
              </div>
            )}
            {botThinking && streaming === null && (
              <div className="message bot thinking">
                <p>
                  hermes is thinking
                  <span className="dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                </p>
              </div>
            )}
            {agentTyping && <p className="notice small">Agent is typing…</p>}
          </div>

          <form className="composer" onSubmit={send}>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={
                resolved
                  ? "Conversation closed"
                  : status === "open"
                    ? "Type your message…"
                    : "Reconnecting…"
              }
              disabled={resolved || !activeId}
            />
            <button type="submit" disabled={!draft.trim() || resolved}>
              Send
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}
