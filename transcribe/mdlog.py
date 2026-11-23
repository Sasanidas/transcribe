"""Readable local transcript (Markdown), written live next to the JSONL file.

Lines are appended as they arrive; a rename (`2=Tanaka`, `me=...`) rewrites the
file so earlier lines get the new name too.
"""

from __future__ import annotations

import datetime as dt
import threading

from .gdoc import DocLine


class MarkdownTranscript:
    def __init__(self, path: str, title: str | None = None):
        self.path = path
        self.title = title or f"Meeting transcript {dt.datetime.now():%Y-%m-%d %H:%M}"
        self.lines: list[DocLine] = []
        self.lock = threading.Lock()
        self._rewrite()

    @staticmethod
    def _fmt(ln: DocLine) -> str:
        out = f"**{ln.time}  {ln.speaker}:** {ln.en or ''}\n"
        if ln.ja:
            out += f"> {ln.ja}\n"
        return out + "\n"

    def _rewrite(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n\n")
            f.writelines(self._fmt(ln) for ln in self.lines)

    def add(self, line: DocLine) -> None:
        with self.lock:
            self.lines.append(line)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self._fmt(line))

    def relabel(self, old: str, new: str) -> None:
        with self.lock:
            for ln in self.lines:
                if ln.speaker == old:
                    ln.speaker = new
            self._rewrite()
