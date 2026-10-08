"""End-to-end integration test: REST turn, WS streaming, escalation, claim, resolve.

Covers the full pipeline against a live server:

  anonymous login -> create conversation -> REST turn (retrieval + citations)
  -> WS streaming turn -> safety escalation -> dashboard queue events
  -> agent login -> transcript -> claim (double-claim rejected) -> agent reply
  -> resolve -> resolved events -> POST rejected on closed conversation

Run with:  .venv/bin/python tests/e2e_ws.py
Requires:  uvicorn running on :8000, .env with DATABASE_URL + agent creds,
           HF cache populated (the server runs with HF_HUB_OFFLINE=1).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

# Repo root on sys.path so `app.*` imports work from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings  # noqa: E402  (loads .env)

BASE = os.environ.get("E2E_BASE", "http://localhost:8000")
WS = os.environ.get("E2E_WS", "ws://localhost:8000")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""), flush=True)


async def recv_until(ws, wanted: set[str], timeout: float = 120):
    """Collect frames until one of `wanted` types arrives (skip noise frames)."""
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"wanted {wanted}, saw {[f.get('type') for f in seen]}")
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
        seen.append(frame)
        if frame.get("type") in wanted:
            return frame, seen


async def expect_rejected(url: str) -> bool:
    """True when the WS handshake/connect fails (server refuses before accept)."""
    try:
        async with connect(url, open_timeout=10) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
            except Exception:
                pass
            return False  # connection opened => not rejected
    except Exception:
        return True


async def wait_server() -> None:
    async with httpx.AsyncClient(timeout=5) as c:
        for _ in range(40):
            try:
                r = await c.get(f"{BASE}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise SystemExit("server not reachable")


async def main() -> None:
    t0 = time.monotonic()
    await wait_server()

    async with httpx.AsyncClient(timeout=180) as c:
        # ---- passenger setup ------------------------------------------------
        r = await c.post(f"{BASE}/auth/anonymous", json={"first_name": "E2EMary"})
        check("anonymous login", r.status_code == 200, str(r.status_code))
        ptoken = r.json()["access_token"]
        ph = {"Authorization": f"Bearer {ptoken}"}

        r = await c.post(f"{BASE}/chat/conversations", headers=ph)
        check("create conversation", r.status_code == 201, str(r.status_code))
        conv_id = r.json()["id"]

        # ---- REST fallback turn ---------------------------------------------
        r = await c.post(
            f"{BASE}/chat/conversations/{conv_id}/messages",
            headers=ph,
            json={
                "conversation_id": conv_id,
                "content": "What is the checked baggage allowance in economy?",
            },
        )
        ok = r.status_code == 200
        body = r.json() if ok else r.text
        check("REST turn 200", ok, str(body)[:180])
        if ok:
            check(
                "REST turn produced bot answer",
                bool(body.get("bot_message")),
                f"intent={body.get('intent')} sentiment={body.get('sentiment_label')}",
            )
            check(
                "REST turn cited sources",
                bool(body.get("bot_message", {}).get("sources")),
                str(body.get("bot_message", {}).get("sources"))[:120],
            )

        # ---- bad-token WS rejection -----------------------------------------
        check(
            "WS rejects garbage token",
            await expect_rejected(f"{WS}/ws/conversations/{conv_id}?token=not.a.jwt"),
        )

        # ---- agent login + dashboard (open BEFORE escalation to catch it) ----
        r = await c.post(
            f"{BASE}/auth/agent/login",
            json={
                "username": settings.agent_username,
                "password": settings.agent_password,
            },
        )
        check("agent login", r.status_code == 200, str(r.status_code))
        atoken = r.json()["access_token"]
        ah = {"Authorization": f"Bearer {atoken}"}

        async with connect(f"{WS}/ws/dashboard?token={atoken}", open_timeout=10) as dash:

            # ---- passenger WS: routine streamed turn --------------------------
            async with connect(
                f"{WS}/ws/conversations/{conv_id}?token={ptoken}", open_timeout=10
            ) as pws:
                await pws.send(
                    json.dumps({"type": "user_message", "content": "How do I check in online?"})
                )
                done, frames = await recv_until(pws, {"bot_done", "bot_error"})
                tokens = [f["content"] for f in frames if f.get("type") == "bot_token"]
                check("WS streamed tokens", len(tokens) > 0, f"{len(tokens)} tokens")
                check("WS bot_done has message", bool(done.get("message")))
                if done.get("message") and tokens:
                    streamed = "".join(tokens)
                    final = done["message"]["content"]
                    check(
                        "stream matches final answer",
                        final.startswith(streamed[:40]),
                        f"stream={streamed[:40]!r}",
                    )

                # ---- safety message => escalation ------------------------------
                await pws.send(
                    json.dumps(
                        {"type": "user_message", "content": "There is a bomb on board flight KQ101"}
                    )
                )
                esc, _ = await recv_until(pws, {"escalated"})
                check("safety message escalated", bool(esc.get("reason")), str(esc))

                # dashboard saw the queue_update created
                upd, _ = await recv_until(dash, {"queue_update"}, timeout=60)
                check(
                    "dashboard got queue_update(created)",
                    upd.get("action") == "created",
                    json.dumps(upd)[:140],
                )

                # ---- agent CANNOT enter before claiming -----------------------
                check(
                    "agent WS rejected before claim",
                    await expect_rejected(
                        f"{WS}/ws/conversations/{conv_id}?token={atoken}"
                    ),
                )

                # ---- queue + claim ---------------------------------------------
                r = await c.get(f"{BASE}/agents/escalations?status=pending", headers=ah)
                queue = r.json() if r.status_code == 200 else []
                item = next((e for e in queue if e["conversation_id"] == conv_id), None)
                check("escalation in pending queue", item is not None, f"{len(queue)} pending")
                check(
                    "queue card enriched",
                    bool(item and item.get("passenger_name") == "E2EMary" and item.get("last_message")),
                    f"card={json.dumps(item)[:160] if item else None}",
                )

                if item:
                    eid = item["id"]
                    r = await c.post(f"{BASE}/agents/escalations/{eid}/claim", headers=ah)
                    check("claim 200", r.status_code == 200, r.text[:140])
                    r2 = await c.post(f"{BASE}/agents/escalations/{eid}/claim", headers=ah)
                    check("double claim 409", r2.status_code == 409, str(r2.status_code))

                    r = await c.get(f"{BASE}/agents/escalations/{eid}/messages", headers=ah)
                    n_msgs = len(r.json()) if r.status_code == 200 else -1
                    # routine passenger+bot + safety passenger (escalation turns
                    # persist no bot message by design) + possibly the REST turn
                    check("transcript for agent", n_msgs >= 3, f"{n_msgs} messages")

                    # ---- agent replies over the conversation WS ---------------
                    async with connect(
                        f"{WS}/ws/conversations/{conv_id}?token={atoken}", open_timeout=10
                    ) as aws:
                        await aws.send(
                            json.dumps(
                                {
                                    "type": "agent_message",
                                    "content": "Mary, this is KQ support - we're on it now.",
                                }
                            )
                        )
                        frame, _ = await recv_until(pws, {"agent_message"}, timeout=30)
                        msg = frame.get("message") or {}
                        check(
                            "agent message relayed to passenger",
                            msg.get("sender") == "agent",
                            str(msg.get("content"))[:80],
                        )

                    # ---- resolve ------------------------------------------------
                    r = await c.post(
                        f"{BASE}/agents/escalations/{eid}/resolve", headers=ah
                    )
                    check("resolve 200", r.status_code == 200, r.text[:140])

                    upd = None
                    for _ in range(10):
                        upd, _ = await recv_until(dash, {"queue_update"}, timeout=30)
                        if upd.get("action") == "resolved":
                            break
                    check(
                        "dashboard got queue_update(resolved)",
                        bool(upd) and upd.get("action") == "resolved",
                        json.dumps(upd)[:140],
                    )
                    frame, _ = await recv_until(pws, {"resolved"}, timeout=30)
                    check("passenger got resolved event", frame.get("escalation_id") == eid)

                # resolve rejected from pending (workflow guard): re-check status
                r = await c.get(f"{BASE}/chat/conversations", headers=ph)
                convs = r.json() if r.status_code == 200 else []
                target = next((x for x in convs if x["id"] == conv_id), None)
                check(
                    "conversation status resolved",
                    bool(target and target["status"] == "resolved"),
                    str(target),
                )

        # passenger cannot post after resolve
        r = await c.post(
            f"{BASE}/chat/conversations/{conv_id}/messages",
            headers=ph,
            json={"conversation_id": conv_id, "content": "one more thing"},
        )
        check("POST rejected after resolve", r.status_code == 400, str(r.status_code))

    failed = [n for n, ok in results if not ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} checks passed in "
        f"{time.monotonic() - t0:.0f}s",
        flush=True,
    )
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        sys.exit(1)


asyncio.run(main())
