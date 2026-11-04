# Hermes: CSAT close button, remove notice.warn, fix pay-card UI, email-after-payment

## Context / constraints

- Do NOT add or remove anything the user did not explicitly ask for.
- Do not break existing working functionality (booking flow, starter options, 2024-date fix).
- Commit all previously-implemented fixes first, then work on these bugs.
- Plan agent only: no source edits in this turn.

## Issues to fix

### 1. CSAT close (X) button is gone

`frontend/src/styles.css:1345-1362` defines `.csat-close` CSS but `frontend/src/ChatWidget.tsx:361-389` never renders it. Once the CSAT prompt auto-shows after 20s, the passenger has no way to dismiss it.

Fix: render a `<button className="csat-close" onClick={() => setShowCsat(false)}>×</button>` inside the `.csat` div in `ChatWidget.tsx`.

### 2. Remove `notice.warn`

Two places render `className="notice warn"`:
- `ChatWidget.tsx:332-340` — escalated banner.
- `ChatWidget.tsx:354-359` — "Add your email" banner.

Also `styles.css:1627-1629` defines the `.notice.warn` pulse animation.

Fix:
- Remove `warn` from both `className` strings in `ChatWidget.tsx` (keep `notice` only, or replace with `notice ok` for escalated / `notice` plain for email).
- Remove the `.notice.warn` animation block from `styles.css`.

### 3. Pay-card success is elongated when successful

`PaymentCard.tsx:236-262` renders the phone-input `pay-form` when `{!pending || canRetry}`. When `paid` is true, `!pending` is also true, so the phone form still renders underneath the `pay-success` block, making the card tall/elongated.

Fix: change the condition so the phone form only shows when the payment is still pending (or retryable after failure), not when `paid`:
```
{pending && (!started || canRetry) && !paid && (
```
i.e. hide the form entirely once `paid` is true. The `pay-success` block already handles the paid state.

### 4. Agent should only ask for email AFTER payment succeeds

Currently `app/agents/graph.py` (`_booking_fallback`) sets `facts["needs_email"] = True` and asks for the email immediately after `create_booking()`, before the M-Pesa payment is confirmed. It should wait until `start_payment` reports success, then ask for email only after.

Fix:
- In `_booking_fallback`, when creating the booking, set `pending_booking` but do NOT set `needs_email` yet.
- The email prompt should be emitted only after the payment poller / `start_payment` confirms `status === "success"` (i.e. after `payments.start_payment` returns and the poller resolves).
- Simplest deterministic path: after `create_booking` + `start_payment`, return the "payment card below" message WITHOUT asking for email. On the next passenger turn, if `facts.pending_booking` exists and the payment is now success (checked via `paymentUpdates`/facts), set `needs_email = True` and ask for email.

Backend wiring: `graph.py` reads `state.facts` which is loaded from the DB per turn (`turns.py:335`). The payment status is reflected in `Booking.payment_id`/`Booking.booking_status`. The `_booking_fallback` can check `payment.status === "success"` before setting `needs_email`.

## Implementation order

1. Commit all previously-implemented fixes (booking/rebooking, starter options, 2024 date, placeholder phone, date highlighting, email-collection logic, simulated MPesa provider).
2. Frontend: CSAT close button, remove `notice.warn` (JSX + CSS), fix pay-card elongated condition.
3. Backend: defer email request until after payment success in `_booking_fallback`.
4. Validate: `ruff check`, `pytest`, `npm run typecheck`, end-to-end booking flow.

## Validation

- Ruff: `python -m ruff check app services tests`
- Tests: `python -m pytest tests/ -q`
- TypeScript: `npm run typecheck` (frontend)
- E2E: book flight → provide phone → wait for simulated payment → verify email is asked only after payment success → provide email → verify ticket issued + confirmation sent.

## Open questions

- Confirm `MPESA_PROVIDER=simulated` is set in `.env` so the simulated payment resolves (already done in prior turn).
- Confirm previous fixes are committed before starting this batch.
