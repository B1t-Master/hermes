/* WebSocket client: JWT via query param, typed envelope, auto-reconnect with
 * exponential backoff, client heartbeat + read-timeout watchdog.
 *
 * Server pings every 30s; if no frame arrives within READ_TIMEOUT_MS the
 * socket is considered dead and recycled (which triggers reconnect).
 * Outbound frames sent while offline are queued and flushed on reconnect.
 *
 * On every (re)open the consumer should refetch state over REST: the server
 * does not replay missed frames. */

import type { ClientFrame, ServerFrame } from "./types";
import { isServerFrame } from "./types";

export type WsStatus = "connecting" | "open" | "reconnecting" | "closed";

export interface WsHandlers {
  onFrame: (frame: ServerFrame) => void;
  /** Fires after every successful (re)connect: refetch missed state here. */
  onOpen?: () => void;
  onStatus?: (status: WsStatus) => void;
}

export interface WsHandle {
  send: (frame: ClientFrame) => void;
  /** Permanent close: cancels reconnect/heartbeat timers. */
  close: () => void;
}

const BACKOFF_MS = [0, 500, 1000, 2000, 4000, 8000, 15000];
const CLIENT_PING_MS = 25_000;
const READ_TIMEOUT_MS = 70_000;
const WATCHDOG_MS = 15_000;
const MAX_QUEUED = 50;

export function connectWs(path: string, token: string, handlers: WsHandlers): WsHandle {
  const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}` +
    `?token=${encodeURIComponent(token)}`;

  let ws: WebSocket | null = null;
  let stopped = false;
  let attempt = 0;
  let lastFrameAt = 0;
  let reconnectTimer: number | null = null;
  let pingTimer: number | null = null;
  let watchdogTimer: number | null = null;
  const queue: ClientFrame[] = [];

  const clearTimers = () => {
    for (const t of [reconnectTimer, pingTimer, watchdogTimer]) {
      if (t !== null) window.clearInterval(t);
    }
    reconnectTimer = null;
    pingTimer = null;
    watchdogTimer = null;
  };

  const send = (frame: ClientFrame) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(frame));
    } else if (queue.length < MAX_QUEUED) {
      queue.push(frame);
    }
  };

  const startHeartbeat = () => {
    pingTimer = window.setInterval(() => send({ type: "ping" }), CLIENT_PING_MS);
    watchdogTimer = window.setInterval(() => {
      if (ws && Date.now() - lastFrameAt > READ_TIMEOUT_MS) {
        ws.close(); // stale: onclose -> reconnect
      }
    }, WATCHDOG_MS);
  };

  const scheduleReconnect = () => {
    if (stopped) return;
    const delay = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
    attempt += 1;
    handlers.onStatus?.("reconnecting");
    reconnectTimer = window.setTimeout(open, delay);
  };

  const open = () => {
    if (stopped) return;
    handlers.onStatus?.(attempt === 0 ? "connecting" : "reconnecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      attempt = 0;
      lastFrameAt = Date.now();
      handlers.onStatus?.("open");
      // Flush anything queued while we were down.
      while (queue.length > 0) {
        const frame = queue.shift();
        if (frame) ws?.send(JSON.stringify(frame));
      }
      startHeartbeat();
      handlers.onOpen?.();
    };

    ws.onmessage = (ev) => {
      lastFrameAt = Date.now();
      if (typeof ev.data !== "string") return;
      let data: unknown;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return; // ignore non-JSON frames
      }
      if (!isServerFrame(data)) return;
      if (data.type === "ping") {
        send({ type: "pong" });
        return; // heartbeat is handled here, consumers don't care
      }
      handlers.onFrame(data);
    };

    ws.onclose = () => {
      if (pingTimer !== null) window.clearInterval(pingTimer);
      if (watchdogTimer !== null) window.clearInterval(watchdogTimer);
      pingTimer = null;
      watchdogTimer = null;
      ws = null;
      if (stopped) {
        handlers.onStatus?.("closed");
        return;
      }
      scheduleReconnect();
    };

    ws.onerror = () => {
      /* onclose always follows; reconnect handled there */
    };
  };

  open();

  return {
    send,
    close: () => {
      stopped = true;
      clearTimers();
      if (ws) {
        ws.close();
      } else {
        handlers.onStatus?.("closed");
      }
    },
  };
}
