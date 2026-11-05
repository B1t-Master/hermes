"""Layout audit for the UI: measures what the CSS actually does.

The Memphis restyle has a few hard requirements that are easy to get wrong
and invisible in a screenshot (the agent drawer must dock to the *right* edge,
the composer must sit at the bottom of the panel, the frosted blur must
actually apply). This script measures them in headless Chrome.

Usage:  .venv/bin/python tools/audit_ui.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from dotenv import dotenv_values

from shot import Chrome  # noqa: E402

APP = "http://127.0.0.1:5173/"

CHECKS: list[tuple[str, str]] = []


def check(name: str, js: str) -> None:
    CHECKS.append((name, js))


# --- shared layout probes -------------------------------------------------

check(
    "body font is Poppins",
    "getComputedStyle(document.body).fontFamily",
)

check(
    "no horizontal overflow",
    "document.documentElement.scrollWidth <= window.innerWidth + 1",
)

check(
    "bot bubble has a 2px+ ink border",
    """
(() => {
  const m = document.querySelector('.message.bot');
  if (!m) return 'no bot message';
  const s = getComputedStyle(m);
  return [s.borderTopWidth, s.borderTopStyle, s.borderTopColor,
          s.boxShadow].join(' | ');
})()
""",
)

check(
    "action chips are pills with an ink border",
    """
(() => {
  const c = document.querySelector('.action-chip');
  if (!c) return 'no chip';
  const s = getComputedStyle(c);
  return [s.borderTopWidth, s.borderTopColor, s.borderRadius,
          s.backgroundColor].join(' | ');
})()
""",
)

check(
    "composer is a bordered block",
    """
(() => {
  const i = document.querySelector('.composer input');
  if (!i) return 'no composer';
  const s = getComputedStyle(i);
  return [s.borderTopWidth, s.borderTopColor, s.boxShadow].join(' | ');
})()
""",
)

# --- agent drawer ---------------------------------------------------------

check(
    "drawer overlay justifies to the end (right edge)",
    "getComputedStyle(document.querySelector('.transcript-overlay')).justifyContent",
)

check(
    "drawer panel is docked right and full height",
    """
(() => {
  const p = document.querySelector('.transcript-panel');
  if (!p) return 'no panel';
  const r = p.getBoundingClientRect();
  const overlay = document.querySelector('.transcript-overlay').getBoundingClientRect();
  return JSON.stringify({
    panelLeft: Math.round(r.left),
    panelRight: Math.round(r.right),
    overlayRight: Math.round(overlay.right),
    panelHeight: Math.round(r.height),
    overlayHeight: Math.round(overlay.height),
    dockedRight: Math.abs(r.right - overlay.right) <= 1,
    fullHeight: Math.abs(r.height - overlay.height) <= 1,
  });
})()
""",
)

check(
    "drawer panel scrolls its own body (composer pinned to bottom)",
    """
(() => {
  const p = document.querySelector('.transcript-panel');
  const body = p.querySelector('.chat-body');
  const composer = p.querySelector('.composer');
  if (!body || !composer) return 'no composer in panel';
  const pb = p.getBoundingClientRect();
  const cb = composer.getBoundingClientRect();
  const bb = body.getBoundingClientRect();
  return JSON.stringify({
    panelBottom: Math.round(pb.bottom),
    composerBottom: Math.round(cb.bottom),
    composerPinned: Math.abs(cb.bottom - pb.bottom) < 2,
    bodyScrollable: body.scrollHeight > body.clientHeight,
    bodyOverflowY: getComputedStyle(body).overflowY,
    bodyHeight: Math.round(bb.height),
  });
})()
""",
)

check(
    "frosted blur is applied to the overlay and the panel",
    """
(() => {
  const o = getComputedStyle(document.querySelector('.transcript-overlay'));
  const p = getComputedStyle(document.querySelector('.transcript-panel'));
  return JSON.stringify({
    overlayBackdrop: o.backdropFilter || o.webkitBackdropFilter,
    panelBackdrop: p.backdropFilter || p.webkitBackdropFilter,
  });
})()
""",
)

check(
    "queue cards are pastel stickers with hard shadows",
    """
(() => {
  const c = document.querySelector('.queue-card');
  if (!c) return 'no queue card';
  const s = getComputedStyle(c);
  return [s.backgroundColor, s.borderTopWidth, s.boxShadow].join(' | ');
})()
""",
)


async def run() -> int:
    env = dotenv_values(".env")
    user = env.get("AGENT_USERNAME", "agent")
    password = env.get("AGENT_PASSWORD", "change-me")

    chrome = Chrome()
    await chrome.start()
    results: list[tuple[str, object]] = []
    try:
        # --- passenger side ------------------------------------------------
        await chrome.goto(APP)
        # The Chrome profile is reused between runs: drop any stored session.
        await chrome.js("localStorage.clear(); location.reload();")
        await asyncio.sleep(2.0)
        await chrome.js(
            """
(() => {
  const btn = [...document.querySelectorAll('button')].find(b =>
    b.textContent.trim() === 'Continue as guest');
  btn.click();
  return 'ok';
})()
"""
        )
        await asyncio.sleep(2.5)
        await chrome.js(
            """
(async () => {
  const input = document.querySelector('.composer input');
  if (!input) return 'no composer input: ' + document.body.innerText.slice(0, 200);
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  setter.call(input, 'Hi');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.closest('form').requestSubmit();
  return 'sent';
})()
"""
        )
        await asyncio.sleep(3.0)
        print("passenger page:", await chrome.js("document.body.innerText.slice(0, 300)"))
        for name, js in CHECKS:
            if "drawer" in name or "overlay" in name or "queue" in name:
                continue
            results.append((name, await chrome.js(js)))

        # --- agent side ---------------------------------------------------
        await chrome.js("localStorage.clear(); location.reload();")
        await asyncio.sleep(2.0)
        await chrome.js(
            """
(() => {
  const btn = [...document.querySelectorAll('button')].find(b =>
    b.textContent.trim() === 'Agent sign in');
  btn.click();
  return 'ok';
})()
"""
        )
        await asyncio.sleep(0.6)
        await chrome.js(
            f"""
(async () => {{
  const inputs = [...document.querySelectorAll('.auth input')];
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  setter.call(inputs[0], {json.dumps(user)});
  inputs[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
  setter.call(inputs[1], {json.dumps(password)});
  inputs[1].dispatchEvent(new Event('input', {{ bubbles: true }}));
  inputs[1].closest('form').requestSubmit();
  return 'ok';
}})()
"""
        )
        await asyncio.sleep(3.0)
        await chrome.js(
            """
(() => {
  const card = document.querySelector('.queue-card');
  if (card) { card.click(); return 'opened'; }
  return 'queue empty';
})()
"""
        )
        await asyncio.sleep(2.0)
        for name, js in CHECKS:
            if "drawer" in name or "overlay" in name or "queue" in name:
                results.append((name, await chrome.js(js)))
    finally:
        await chrome.stop()

    print("\n=== UI layout audit ===")
    for name, value in results:
        print(f"  {name}:\n    {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
