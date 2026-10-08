import type {
  AgentInfo,
  Conversation,
  Escalation,
  Message,
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
  token?: string;
}

/** All backend routes are proxied under /api (vite strips the prefix). */
async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {};
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (opts.token) headers["Authorization"] = `Bearer ${opts.token}`;

  const res = await fetch(`/api${path}`, {
    method: opts.method ?? "GET",
    headers,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new ApiError(res.status, data.detail || res.statusText);
  }
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

  /* escalation queue (agent) */
  queue: (token: string, status = "active") =>
    request<QueueEscalation[]>(`/agents/escalations?status=${status}`, { token }),
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
};

export type { Role };
