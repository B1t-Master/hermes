import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { api, ApiError } from "./api";
import type { Payment } from "./types";

interface PaymentCardProps {
  token: string;
  paymentId: string;
  mode: "passenger" | "agent";
  /** Latest server-pushed state, when a `payment_update` arrived over the WS. */
  live?: Payment;
  /** Agent only: issue the e-ticket and deliver it in the conversation. */
  onSendTicket?: (paymentId: string) => Promise<void>;
}

const POLL_MS = 3000;

const TERMINAL = new Set(["success", "failed", "cancelled", "timeout"]);

function prettyDate(value: string | null): string {
  if (!value) return "";
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return value;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function money(amount: number, currency: string): string {
  return `${currency} ${Math.round(amount).toLocaleString()}`;
}

function routeLabel(payment: Payment): string {
  if (!payment.origin || !payment.destination) return "";
  const city = (code: string) => code;
  return `${city(payment.origin)} → ${city(payment.destination)}`;
}

function statusLabel(status: Payment["status"]): string {
  switch (status) {
    case "pending":
      return "Awaiting payment";
    case "success":
      return "Paid";
    case "cancelled":
      return "Cancelled";
    case "timeout":
      return "Timed out";
    default:
      return "Failed";
  }
}

/** In-chat M-Pesa payment card.
 *
 * Passenger mode collects the number and posts it straight to the backend;
 * the number never passes through the chat/LLM. Agent mode is read-only and
 * offers the Send e-ticket action once the payment has succeeded.
 */
export default function PaymentCard({
  token,
  paymentId,
  mode,
  live,
  onSendTicket,
}: PaymentCardProps) {
  const [payment, setPayment] = useState<Payment | null>(null);
  const [phone, setPhone] = useState("");  const [email, setEmail] = useState("");  const [error, setError] = useState<string | null>(null);  const [submitting, setSubmitting] = useState(false);  const [emailSubmitting, setEmailSubmitting] = useState(false);  const [sending, setSending] = useState(false);  const [sent, setSent] = useState(false);
  const load = useCallback(async () => {
    try {
      const next =
        mode === "agent"
          ? await api.agentGetPayment(token, paymentId)
          : await api.getPayment(token, paymentId);
      setPayment(next);
      return next;
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the payment.");
      return null;
    }
  }, [mode, paymentId, token]);

  useEffect(() => {
    setPayment(null);
    setPhone("");
    setEmail("");
    setError(null);
    setSent(false);
    void load();
  }, [load]);

  useEffect(() => {
    if (live && live.id === paymentId) setPayment(live);
  }, [live, paymentId]);

  // Poll while the payment is waiting for the passenger's PIN. The WS
  // `payment_update` is the fast path; polling covers a dropped socket.
  useEffect(() => {
    if (!payment || TERMINAL.has(payment.status)) return;
    const timer = window.setInterval(() => {
      void load();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [payment, load]);

  async function submitPhone(event: FormEvent) {
    event.preventDefault();
    const value = phone.trim();
    if (!value) {
      setError("Enter the M-Pesa number that should receive the prompt.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const next = await api.submitPaymentPhone(token, paymentId, value);
      setPayment(next);
      setPhone("");
      if (next.status === "failed") {
        setError(next.message || "The payment could not be started.");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send the prompt.");
    } finally {
      setSubmitting(false);
    }
  }


  async function submitEmail(event: FormEvent) {
    event.preventDefault();
    const value = email.trim();
    if (!value) {
      setError("Enter the email address that should receive the e-ticket.");
      return;
    }
    setEmailSubmitting(true);
    setError(null);
    try {
      const next = await api.submitPaymentEmail(token, paymentId, value);
      setPayment(next);
      setEmail("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save the email.");
    } finally {
      setEmailSubmitting(false);
    }
  }

  async function sendTicket() {
    if (!onSendTicket) return;
    setSending(true);
    setError(null);
    try {
      await onSendTicket(paymentId);
      setSent(true);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send the e-ticket.");
    } finally {
      setSending(false);
    }
  }

  

  if (!payment) {
    return (
      <div className="pay-card pay-card-loading">
        <span className="pay-spinner" aria-hidden="true" />
        <span>{error || "Loading payment…"}</span>
      </div>
    );
  }

  const pending = payment.status === "pending";
  const started = pending && Boolean(payment.checkout_request_id);
  const paid = payment.status === "success";
  const ticketed = payment.booking_status === "ticketed" || sent;
  const canRetry = ["failed", "cancelled", "timeout"].includes(payment.status);

  return (
    <div className={`pay-card pay-${payment.status}`}>
      <div className="pay-head">
        <div>
          <p className="pay-kicker">M-Pesa payment</p>
          <p className="pay-route">{routeLabel(payment) || payment.pnr}</p>
          {payment.flight_no && (
            <p className="pay-meta">
              {payment.flight_no} · {prettyDate(payment.depart_date)}
              {payment.depart_time ? ` at ${payment.depart_time}` : ""}
            </p>
          )}
        </div>
        <div className="pay-amount">
          <strong>{money(payment.amount, payment.currency)}</strong>
          <span className={`pay-status pay-status-${payment.status}`}>
            {statusLabel(payment.status)}
          </span>
        </div>
      </div>

      <dl className="pay-facts">
        <div>
          <dt>Booking reference</dt>
          <dd>{payment.pnr}</dd>
        </div>
        {payment.phone_masked && (
          <div>
            <dt>M-Pesa number</dt>
            <dd>{payment.phone_masked}</dd>
          </div>
        )}
        {payment.receipt && (
          <div>
            <dt>M-Pesa receipt</dt>
            <dd>{payment.receipt}</dd>
          </div>
        )}
        {payment.passenger_email && (
          <div>
            <dt>Email</dt>
            <dd>{payment.passenger_email}</dd>
          </div>
        )}
        {payment.eticket_number && (
          <div>
            <dt>E-ticket</dt>
            <dd>{payment.eticket_number}</dd>
          </div>
        )}
      </dl>

      {mode === "passenger" && (
        <>
          {pending && started && (
            <div className="pay-wait" role="status">
              <span className="pay-spinner" aria-hidden="true" />
              <div>
                <strong>Check your phone</strong>
                <p>
                  Enter your M-Pesa PIN to authorise{" "}
                  {money(payment.amount, payment.currency)}. This card updates
                  automatically.
                </p>
              </div>
            </div>
          )}

          {(!pending || canRetry) && !paid && (
            <form className="pay-form" onSubmit={submitPhone}>
              <label htmlFor={`phone-${payment.id}`}>M-Pesa number</label>
              <div className="pay-row">
                <input
                  id={`phone-${payment.id}`}
                  type="tel"
                  inputMode="tel"
                  autoComplete="tel"
                  placeholder="0712 345 678"
                  value={phone}
                  onChange={(event) => setPhone(event.target.value)}
                  disabled={submitting}
                />
                <button type="submit" disabled={submitting}>
                  {submitting
                    ? "Sending…"
                    : canRetry
                      ? "Try again"
                      : "Send prompt"}
                </button>
              </div>
              <p className="pay-note">
                Your number goes straight to M-Pesa. It is never posted in the chat.
              </p>
            </form>
          )}

          {paid && (
            <div className="pay-success" role="status">
              <strong>Payment received</strong>
              <p>{payment.message || "An agent is preparing your e-ticket."}</p>
            </div>
          )}

          {paid && !payment.passenger_email && (
            <form className="pay-form" onSubmit={submitEmail} style={{ marginTop: 1.5 }}>
              <label htmlFor={`email-${payment.id}`}>
                Email for your e-ticket (optional)
              </label>
              <div className="pay-row">
                <input
                  id={`email-${payment.id}`}
                  type="email"
                  inputMode="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  disabled={emailSubmitting}
                />
                <button type="submit" disabled={emailSubmitting || !email.trim()}>
                  {emailSubmitting ? "Saving…" : "Email copy"}
                </button>
              </div>
              <p className="pay-note">
                We'll email a copy of the e-ticket. It never appears in the chat.
              </p>
            </form>
          )}

          {paid && payment.passenger_email && (
            <p className="pay-note">
              We'll email your e-ticket to {payment.passenger_email}.
            </p>
          )}
        </>
      )}

      {mode === "agent" && (
        <div className="pay-agent-actions">
          {paid && !ticketed && (
            <button type="button" onClick={sendTicket} disabled={sending}>
              {sending ? "Sending…" : "Send e-ticket"}
            </button>
          )}
          {(ticketed || payment.eticket_number) && (
            <p className="pay-success-line">
              E-ticket {payment.eticket_number ? `#${payment.eticket_number} ` : ""}
              sent to the passenger.
            </p>
          )}
          {!paid && <p className="pay-meta">Waiting for the passenger to pay.</p>}
        </div>
      )}

      {error && <p className="pay-error">{error}</p>}
    </div>
  );
}
