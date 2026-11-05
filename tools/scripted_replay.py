"""End-to-end check of the three bug classes this work was about.

1. Memory - the passenger's booking reference survives a rebook.
2. Behaviour - the first question is answered; a hand-off is offered, not forced.
3. Hand-off - an explicit "talk to a human" still escalates, with an ack.

Usage:  .venv/bin/python tools/scripted_replay.py [--base-url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx


async def _register(client: httpx.AsyncClient, base: str) -> tuple[str, str]:
    email = f"replay-{int(asyncio.get_event_loop().time() * 1000)}@example.com"
    password = "replay-password"
    res = await client.post(
        f"{base}/auth/register",
        json={"first_name": "Replay", "email": email, "password": password},
    )
    res.raise_for_status()
    res = await client.post(
        f"{base}/auth/login", json={"email": email, "password": password}
    )
    res.raise_for_status()
    return email, res.json()["access_token"]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=180) as client:
        email, token = await _register(client, base)
        headers = {"Authorization": f"Bearer {token}"}

        res = await client.post(
            f"{base}/chat/conversations", json={}, headers=headers
        )
        res.raise_for_status()
        conversation_id = res.json()["id"]

        script = [
            ("Hi", "greeting"),
            ("My flight KQ100 was cancelled, please rebook me.", "rebook start"),
            ("My booking reference is HRM-1234567.", "rebook complete"),
            ("What's my ticket number?", "memory check"),
            ("My bag didn't arrive.", "baggage"),
            ("I want to speak to a human agent.", "explicit hand-off"),
        ]

        for text, label in script:
            res = await client.post(
                f"{base}/chat/conversations/{conversation_id}/messages",
                json={"conversation_id": conversation_id, "content": text},
                headers=headers,
            )
            res.raise_for_status()
            data = res.json()
            offer = data.get("escalation_offer") or {}
            options = [opt.get("label") for opt in (offer.get("options") or [])]
            message = data.get("bot_message")
            if isinstance(message, dict):
                reply = message.get("content") or message.get("text") or ""
                chips = [
                    a.get("label")
                    for a in (message.get("actions") or message.get("message_actions") or [])
                ]
            else:
                reply = message or ""
                chips = []
            print("=" * 78)
            print(f"[{label}] passenger: {text}")
            print(f"  intent={data.get('intent')} escalated={data.get('escalate')} "
                  f"offer={data.get('offer_escalation')} options={options}")
            if chips:
                print(f"  chips: {chips}")
            print(f"  bot: {reply.strip()[:700]}")

        # The whole conversation, straight from the database-facing API.
        res = await client.get(
            f"{base}/chat/conversations/{conversation_id}/messages", headers=headers
        )
        if res.status_code == 200:
            print("=" * 78)
            print("stored messages:", len(res.json()))

    print("\n=== replay finished ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
