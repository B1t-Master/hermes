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

/** A quick-reply button on a bot message; clicking sends `value` as the reply. */
export interface MessageAction {
  label: string;
  value: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  sender: MessageSender;
  content: string;
  sentiment_score: number | null;
  sentiment_label: string | null;
  sources?: MessageSource[];
  actions?: MessageAction[];
  /** In-chat M-Pesa payment card attached to a bot/agent message. */
  payment_id?: string | null;
  created_at: string;
  /** Client-only marker for optimistically rendered (not yet echoed) messages. */
  pending?: boolean;
}

export type PaymentStatus = "pending" | "success" | "failed" | "cancelled" | "timeout";

/** In-chat M-Pesa payment card state (app/schemas.py PaymentOut). */
export interface Payment {
  id: string;
  conversation_id: string | null;
  pnr: string;
  phone: string | null;
  phone_masked: string | null;
  amount: number;
  currency: string;
  status: PaymentStatus;
  receipt: string | null;
  checkout_request_id: string | null;
  flight_no: string | null;
  origin: string | null;
  destination: string | null;
  depart_date: string | null;
  depart_time: string | null;
  passenger_name: string | null;
  passenger_email: string | null;
  booking_status: string | null;
  eticket_number: string | null;
  message: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface SendTicketResult {
  payment: Payment;
  ticket: Record<string, unknown>;
  message: Message;
}

export interface EscalationOffer {
  reason: string;
  options: MessageAction[];
}

export interface Turn {
  passenger_message: Message;
  bot_message: Message | null;
  escalate: boolean;
  escalation_reason: string | null;
  offer_escalation?: boolean;
  escalation_offer?: EscalationOffer | null;
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
  | { type: "payment_update"; payment: Payment }
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

export interface KnowledgeDocument {
  id: string;
  source_url: string | null;
  title: string;
  doc_type: string;
  content_hash: string;
  last_fetched: string;
  created_at: string;
  fragment_count: number | null;
}

export interface IngestReport {
  new: number;
  updated: number;
  unchanged: number;
  failed: [string, string][];
  summary: string;
}

export interface Metrics {
  avg_response_ms: number | null;
  avg_retrieval_hits: number | null;
  escalations: number;
  csat_responses: number;
  avg_csat: number | null;
}
