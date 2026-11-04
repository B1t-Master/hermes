"""Drive the UI in headless Chrome and take screenshots.

Used to eyeball the Corporate Memphis restyle without a desktop browser:
launches Chrome with the DevTools protocol, clicks through the app and
writes PNGs to /tmp/opencode/shots.

Usage:  .venv/bin/python tools/shot.py <name> [--script "<js>"]
"""

import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

CHROME = "/usr/bin/google-chrome"
PORT = 9333
APP = "http://127.0.0.1:5173/"
OUT = Path("/tmp/opencode/shots")


class Chrome:
    def __init__(self, width: int = 1440, height: int = 980):
        self.width = width
        self.height = height
        self.proc: subprocess.Popen | None = None
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._id = 0

    async def start(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
                f"--remote-debugging-port={PORT}",
                f"--window-size={self.width},{self.height}",
                "--user-data-dir=/tmp/opencode/chrome-profile",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except Exception:
                await asyncio.sleep(0.25)
        tab = httpx.put(
            f"http://127.0.0.1:{PORT}/json/new?{APP}", timeout=10
        ).json()
        self.ws = await websockets.connect(tab["webSocketDebuggerUrl"], max_size=None)
        await self.send("Page.enable")
        await self.send("Runtime.enable")

    async def send(self, method: str, params: dict | None = None) -> dict:
        assert self.ws is not None
        self._id += 1
        msg_id = self._id
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = json.loads(await self.ws.recv())
            if raw.get("id") == msg_id:
                if "error" in raw:
                    raise RuntimeError(f"{method}: {raw['error']}")
                return raw.get("result", {})

    async def goto(self, url: str) -> None:
        await self.send("Page.navigate", {"url": url})
        await asyncio.sleep(1.5)

    async def js(self, expression: str) -> object:
        result = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        if result.get("exceptionDetails"):
            return {"error": result["exceptionDetails"].get("text")}
        return result.get("result", {}).get("value")

    async def shot(self, name: str) -> Path:
        data = await self.send("Page.captureScreenshot", {"format": "png"})
        path = OUT / f"{name}.png"
        path.write_bytes(base64.b64decode(data["data"]))
        print(f"wrote {path}")
        return path

    async def stop(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self.proc is not None:
            self.proc.terminate()


CLICK = """
(() => {
  const needle = %s;
  const btn = [...document.querySelectorAll('button')].find(b =>
    (b.textContent || '').trim().toLowerCase().includes(needle.toLowerCase())
  );
  if (!btn) return 'not found: ' + needle;
  btn.click();
  return 'clicked: ' + btn.textContent.trim();
})()
"""


async def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "shot"
    chrome = Chrome()
    await chrome.start()
    try:
        await chrome.goto(APP)
        await asyncio.sleep(1.2)
        await chrome.shot(f"{name}-auth")
        print(await chrome.js(CLICK % json.dumps("Continue as guest")))
        await asyncio.sleep(3.0)
        await chrome.shot(f"{name}-chat")
        # Ask a question so the thread has content.
        await chrome.js(
            """
(async () => {
  const input = document.querySelector('.composer input');
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(input, 'What is my checked baggage allowance?');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  const form = input.closest('form');
  form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
  return 'sent';
})()
"""
        )
        await asyncio.sleep(9.0)
        await chrome.shot(f"{name}-thread")
        print(await chrome.js("document.querySelector('.chat-body').innerText.slice(0, 900)"))
    finally:
        await chrome.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
