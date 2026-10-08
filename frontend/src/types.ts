/* Shared frontend types, hand-synced with backend schemas (app/schemas.py)
 * and the WS envelope documented in app/api/ws.py. */

export type Role = "passenger" | "agent";

export interface Token {
  access_token: string;
  token_type: string;
  role: Role;
}

export interface Passenger {
  id: string;
  first_name: string;
  last_name: string | null;
  email: string | null;
  is_anonymous: boolean;
}

export interface AgentInfo {
  id: string;
  name: string;
  email: string;
}

export type ConversationStatus = "active" | "escalated" | "resolved";

export interface Conversation {
  id: string;
  status: ConversationStatus;
  turn_count: number;
  escalation_reason: string | null;
}

export type MessageSender = "passenger" | "bot" | "agent";

export interface MessageSource {
  title: string;
  url: string | null;
  score?: number;
}

export interface Message {
  id: string;
  conversation_id: string;
  sender: MessageSender;
  content: string;
  sentiment_score: number | null;
  sentiment_label: string | null;
  sources?: MessageSource[];
  created_at: string;
  /** Client-only marker for optimistically rendered (not yet echoed) messages. */
  pending?: boolean;
}

export interface Turn {
  passenger_message: Message;
  bot_message: Message | null;
  escalate: boolean;
  escalation_reason: string | null;
  intent: string | null;
  sentiment_score: number | null;
  sentiment_label: string | null;
}

export type EscalationStatus = "pending" | "assigned" | "resolved";

export interface Escalation {
  id: string;
  conversation_id: string;
  reason: string;
  summary: string;
  status: EscalationStatus;
  created_at: string;
}

export interface QueueEscalation extends Escalation {
  passenger_name: string;
  turn_count: number;
  conversation_status: ConversationStatus;
  agent_id: string | null;
  agent_name: string | null;
  last_message: string | null;
}

/* ------------------------------------------------------------------ */
/* WebSocket envelope                                                  */
/* ------------------------------------------------------------------ */

/** Client -> server frames. */
export type ClientFrame =
  | { type: "user_message"; content: string }
  | { type: "agent_message"; content: string }
  | { type: "typing" }
  | { type: "ping" }
  | { type: "pong" };

/** Server -> client frames (discriminated union on `type`). */
export type ServerFrame =
  /* heartbeat */
  | { type: "ping" }
  | { type: "pong" }
  /* addressed to the sender of the current turn */
  | { type: "typing"; sender: "bot" | Role }
  | { type: "bot_token"; content: string }
  | { type: "bot_done"; message: Message | null }
  | { type: "bot_error"; detail: string }
  /* room broadcasts */
  | { type: "passenger_message"; message: Message }
  | { type: "bot_message"; message: Message }
  | { type: "agent_message"; message: Message }
  | { type: "escalated"; reason: string }
  | { type: "resolved"; escalation_id: string }
  /* dashboard broadcasts */
  | {
      type: "queue_update";
      action: "created" | "claimed" | "released" | "resolved";
      escalation_id: string;
      conversation_id: string;
    }
  /* misc */
  | { type: "error"; detail: string };

export function isServerFrame(data: unknown): data is ServerFrame {
  return (
    typeof data === "object" &&
    data !== null &&
    typeof (data as { type?: unknown }).type === "string"
  );
}
