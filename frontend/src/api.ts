import type {
  AgentInfo,
  Conversation,
  Escalation,
  IngestReport,
  KnowledgeDocument,
  Message,
  Metrics,
  Passenger,
  QueueEscalation,
  Role,
  Token,
  Turn,
} from "./types";

/** Backend error with HTTP status (FastAPI `detail`). */
export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  /** Multipart body; mutually exclusive with `body`. */
  formData?: FormData;
  token?: string;
}

/** All backend routes are proxied under /api (vite strips the prefix). */
async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {};
  let body: BodyInit | undefined;
  if (opts.formData) {
    // Let the browser set the multipart boundary.
    body = opts.formData;
  } else if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  if (opts.token) headers["Authorization"] = `Bearer ${opts.token}`;

  const res = await fetch(`/api${path}`, {
    method: opts.method ?? "GET",
    headers,
    body,
  });
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new ApiError(res.status, data.detail || res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  /* auth */
  register: (data: {
    first_name: string;
    last_name?: string;
    email: string;
    password: string;
  }) => request<Passenger>("/auth/register", { method: "POST", body: data }),
  login: (data: { email: string; password: string }) =>
    request<Token>("/auth/login", { method: "POST", body: data }),
  anonymous: (data: { first_name: string }) =>
    request<Token>("/auth/anonymous", { method: "POST", body: data }),
  agentLogin: (data: { username: string; password: string }) =>
    request<Token>("/auth/agent/login", { method: "POST", body: data }),
  agentMe: (token: string) => request<AgentInfo>("/auth/agent/me", { token }),

  /* conversations + messages (passenger) */
  listConversations: (token: string) =>
    request<Conversation[]>("/chat/conversations", { token }),
  createConversation: (token: string) =>
    request<Conversation>("/chat/conversations", { method: "POST", token }),
  listMessages: (token: string, conversationId: string) =>
    request<Message[]>(`/chat/conversations/${conversationId}/messages`, { token }),
  /* non-streaming REST fallback for one turn */
  sendTurn: (token: string, conversationId: string, content: string) =>
    request<Turn>(`/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      token,
      body: { conversation_id: conversationId, content },
    }),
  /* end-of-conversation star rating (1..5) */
  csat: (token: string, conversationId: string, rating: number) =>
    request<void>(`/chat/conversations/${conversationId}/csat`, {
      method: "POST",
      token,
      body: { rating },
    }),

  /* escalation queue (agent) */
  queue: (token: string, status = "active") =>
    request<QueueEscalation[]>(`/agents/escalations?status=${status}`, { token }),
  /* aggregated performance metrics (agent) */
  metrics: (token: string) => request<Metrics>("/agents/metrics", { token }),
  escalationMessages: (token: string, escalationId: string) =>
    request<Message[]>(`/agents/escalations/${escalationId}/messages`, { token }),
  claim: (token: string, escalationId: string) =>
    request<Escalation>(`/agents/escalations/${escalationId}/claim`, {
      method: "POST",
      token,
    }),
  release: (token: string, escalationId: string) =>
    request<Escalation>(`/agents/escalations/${escalationId}/release`, {
      method: "POST",
      token,
    }),
  resolve: (token: string, escalationId: string) =>
    request<Escalation>(`/agents/escalations/${escalationId}/resolve`, {
      method: "POST",
      token,
    }),

  /* knowledge base (agent) */
  kbList: (token: string) => request<KnowledgeDocument[]>("/kb/sources", { token }),
  kbUpload: (token: string, files: File[], embed = true, force = false) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<IngestReport>(
      `/kb/upload?embed=${embed}&force=${force}`,
      { method: "POST", token, formData: form },
    );
  },
  kbIngestUrls: (token: string, urls: string[]) =>
    request<IngestReport>("/kb/urls", { method: "POST", token, body: { urls } }),
  kbReingest: (token: string, sourceKey: string) =>
    request<IngestReport>(
      `/kb/sources/${encodeURIComponent(sourceKey)}/reingest`,
      { method: "POST", token },
    ),
  kbDelete: (token: string, sourceKey: string) =>
    request<{ ok: boolean }>(`/kb/sources/${encodeURIComponent(sourceKey)}`, {
      method: "DELETE",
      token,
    }),
  kbReset: (token: string) =>
    request<{ ok: boolean }>("/kb/reset", { method: "POST", token }),
};

export type { Role };
