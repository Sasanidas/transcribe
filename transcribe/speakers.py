"""Receives who-is-speaking events from the Firefox add-on and labels segments.

The add-on POSTs {"events": [{"name", "t" (epoch ms), "self"}], "captions": bool}
to http://127.0.0.1:8765/events whenever a speaker's Meet caption grows.

Meet captions trail the audio by roughly 0.3-1.5 s, tile "speaking" animations by
much less, so a segment [start, end] is matched against caption activity in
[start, end + lag] and tile activity in [start, end + tile_lag].
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_ORIGINS = ("moz-extension://", "chrome-extension://")


class SpeakerTracker:
    def __init__(self, port: int = 8765, lag: float = 1.2, history_seconds: float = 600,
                 ignore_names: set[str] | None = None, debug: bool = False, tile_lag: float = 0.4):
        self.lag, self.tile_lag = lag, tile_lag
        self.history_seconds = history_seconds
        self.events: deque[tuple[float, str, str]] = deque()  # (epoch s, name, src), others only
        self.self_names: set[str] = set(ignore_names or ())  # your own Meet name: never a hint
        self.last_message = 0.0
        self.captions_on = False
        self.my_name: str | None = None  # your Meet display name, as read by the add-on
        self.debug = debug
        self._last_tiles: str | None = None
        self._announced = self._hinted = False
        self._first_message = 0.0
        self._lock = threading.Lock()
        tracker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                origin = self.headers.get("Origin", "")
                if self.path != "/events" or (origin and not origin.startswith(ALLOWED_ORIGINS)) \
                        or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.send_response(403)
                    self.end_headers()
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    tracker._ingest(body)
                    self.send_response(204)
                except (ValueError, KeyError, TypeError):
                    self.send_response(400)
                self.end_headers()

            def log_message(self, *args) -> None:  # silence per-request logging
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True, name="speaker-http").start()

    def _ingest(self, body: dict) -> None:
        now = time.time()
        with self._lock:
            self.last_message = now
            self.captions_on = bool(body.get("captions"))
            if not self._announced:
                self._announced, self._first_message = True, now
                print("  [add-on] Firefox add-on connected", flush=True)
            if body.get("self_name") and isinstance(body["self_name"], str):
                if body["self_name"][:100] != self.my_name:
                    print(f"  [add-on] your Meet name: {body['self_name'][:100]} "
                          f"(from {body.get('self_name_source')})", flush=True)
                self.my_name = body["self_name"][:100]
                self.self_names.add(self.my_name)
            elif not self._hinted and now - self._first_message > 15:
                self._hinted = True
                cands = [str(c)[:80] for c in body.get("self_candidates", [])][:8]
                print("  [add-on] can't find your Meet name. Open Meet's People panel once, or type "
                      "`me=Your Name` here.\n           Texts mentioning You/あなた on the page: "
                      f"{cands or 'none'}", flush=True)
            for e in body.get("events", []):
                name = str(e["name"])[:100]
                if e.get("self"):
                    self.self_names.add(name)
                if name in self.self_names:
                    continue
                self.events.append((float(e["t"]) / 1000.0, name, str(e.get("src", "caption"))))
            while self.events and self.events[0][0] < now - self.history_seconds:
                self.events.popleft()
        if self.debug and "tiles" in body:
            desc = ", ".join(f"{t.get('name')!r}[{t.get('source')}]{' SPEAKING' if t.get('speaking') else ''}"
                             f"({t.get('changes')})" for t in body["tiles"]) or "no tiles found"
            if desc != self._last_tiles:
                self._last_tiles = desc
                print(f"  [add-on] you={self.my_name!r} ({body.get('self_name_source')}) captions={'on' if self.captions_on else 'off'} tiles: {desc}", flush=True)

    @property
    def connected(self) -> bool:
        return time.time() - self.last_message < 5

    def who(self, start: float, end: float) -> str | None:
        """Most active speaker per the add-on during the segment (epoch seconds)."""
        with self._lock:
            counts = Counter(n for t, n, src in self.events
                             if start <= t <= end + (self.tile_lag if src == "tile" else self.lag))
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def stop(self) -> None:
        self.server.shutdown()
