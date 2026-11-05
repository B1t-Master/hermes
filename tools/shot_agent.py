"""Screenshot the agent dashboard with the transcript drawer open.

Companion to `tools/shot.py` (passenger side). Signs in as the agent, opens
the first queue card and writes the dashboard + drawer PNGs to
/tmp/opencode/shots.

Usage:  .venv/bin/python tools/shot_agent.py [name]
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).parent))

from shot import Chrome  # noqa: E402

APP = "http://127.0.0.1:5173/"


async def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "agent"
    env = dotenv_values(".env")
    user = env.get("AGENT_USERNAME", "agent")
    password = env.get("AGENT_PASSWORD", "change-me")

    chrome = Chrome()
    await chrome.start()
    try:
        await chrome.goto(APP)
        await chrome.js("localStorage.clear(); location.reload();")
        await asyncio.sleep(2.0)
        await chrome.js(
            """
(() => {
  const btn = [...document.querySelectorAll('button')].find(b =>
    b.textContent.trim() === 'Agent sign in');
  if (!btn) return 'no agent sign-in button';
  btn.click();
  return 'ok';
})()
"""
        )
        await asyncio.sleep(0.8)
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
        await chrome.shot(f"{name}-dashboard")
        print(
            await chrome.js(
                """
(() => {
  const card = document.querySelector('.queue-card');
  if (!card) return 'queue empty';
  card.click();
  return 'opened: ' + card.innerText.slice(0, 120).replace(/\\n/g, ' | ');
})()
"""
            )
        )
        await asyncio.sleep(2.0)
        await chrome.shot(f"{name}-drawer")
        print(await chrome.js("document.body.innerText.slice(0, 600)"))
    finally:
        await chrome.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
