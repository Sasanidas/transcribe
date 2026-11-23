"""Live Google Doc output.

Lines are queued and appended in batches every few seconds (well under the Docs
API's per-user write limit). The local JSONL transcript stays the source of truth;
if Google is unreachable, lines wait in the queue and are retried.

Auth: OAuth "Desktop app" client in credentials.json; the token is cached in
token.json. Scope drive.file = this app can only see/edit documents it created.

  uv run python -m transcribe.gdoc --login     # one-time browser login
  uv run python -m transcribe.gdoc --test      # create a doc with sample lines
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import threading
import time
from dataclasses import dataclass

from .translate import ROOT

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS = os.path.join(ROOT, "credentials.json")
TOKEN = os.path.join(ROOT, "token.json")
GREY = {"color": {"rgbColor": {"red": 0.45, "green": 0.45, "blue": 0.45}}}


def u16(s: str) -> int:
    """Length in UTF-16 code units — the unit Google Docs uses for indices."""
    return len(s.encode("utf-16-le")) // 2


def login(interactive: bool = True):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = Credentials.from_authorized_user_file(TOKEN, SCOPES) if os.path.exists(TOKEN) else None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError("Google login needed: run `uv run python -m transcribe.gdoc --login`")
        if not os.path.exists(CREDENTIALS):
            raise FileNotFoundError(f"{CREDENTIALS} not found — see docs/google-docs-setup.md")
        creds = InstalledAppFlow.from_client_secrets_file(CREDENTIALS, SCOPES).run_local_server(port=0)
    with open(TOKEN, "w") as f:
        f.write(creds.to_json())
    os.chmod(TOKEN, 0o600)
    return creds


def docs_service(interactive: bool = True):
    from googleapiclient.discovery import build

    return build("docs", "v1", credentials=login(interactive), cache_discovery=False)


@dataclass
class DocLine:
    time: str
    speaker: str
    en: str | None
    ja: str | None  # original Japanese (None for English speech)


class DocWriter:
    def __init__(self, service, doc_id: str | None = None, title: str | None = None, flush_seconds: float = 3.0):
        self.svc = service.documents()
        self.lock = threading.Lock()
        self.queue: list[DocLine] = []
        self.relabels: list[tuple[str, str]] = []
        self.flush_seconds = flush_seconds
        self.failures = 0
        if doc_id is None:
            title = title or f"Meeting transcript {dt.datetime.now():%Y-%m-%d %H:%M}"
            doc_id = self.svc.create(body={"title": title}).execute()["documentId"]
            self.doc_id = doc_id
            self._refresh_end()
            self._write_header(title)
        else:
            self.doc_id = doc_id
            self._refresh_end()
        self.url = f"https://docs.google.com/document/d/{self.doc_id}/edit"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gdoc")
        self._thread.start()

    # ---- public API (thread-safe) ----

    def add(self, line: DocLine) -> None:
        with self.lock:
            self.queue.append(line)

    def relabel(self, old: str, new: str) -> None:
        """Rename a speaker in lines already written (e.g. "Speaker 2" -> "Tanaka")."""
        with self.lock:
            self.relabels.append((old, new))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        self.flush()

    # ---- internals ----

    def _refresh_end(self) -> None:
        doc = self.svc.get(documentId=self.doc_id, fields="body/content/endIndex").execute()
        self.end = doc["body"]["content"][-1]["endIndex"] - 1  # insert before the final newline

    def _write_header(self, title: str) -> None:
        text = f"{title}\n"
        note = "Live transcript. Japanese is translated to English on this computer (PLaMo-3-translate); " \
               "the original Japanese is shown in grey under each line.\n\n"
        start = self.end
        reqs = [
            {"insertText": {"location": {"index": start}, "text": text + note}},
            {"updateParagraphStyle": {"range": {"startIndex": start, "endIndex": start + u16(text)},
                                      "paragraphStyle": {"namedStyleType": "HEADING_1"}, "fields": "namedStyleType"}},
            {"updateTextStyle": {"range": {"startIndex": start + u16(text), "endIndex": start + u16(text + note)},
                                 "textStyle": {"italic": True, "foregroundColor": GREY, "fontSize": {"magnitude": 9, "unit": "PT"}},
                                 "fields": "italic,foregroundColor,fontSize"}},
        ]
        self.svc.batchUpdate(documentId=self.doc_id, body={"requests": reqs}).execute()
        self.end += u16(text + note)

    def _requests_for(self, lines: list[DocLine], start: int) -> tuple[list[dict], int]:
        text, styles, pos = "", [], start
        for ln in lines:
            head = f"{ln.time}  {ln.speaker}: "
            body = (ln.en or "") + "\n"
            styles.append(("head", pos, pos + u16(head)))
            text += head + body
            pos += u16(head + body)
            if ln.ja:
                ja = ln.ja + "\n"
                styles.append(("ja", pos, pos + u16(ja) - 1))
                text += ja
                pos += u16(ja)
        reqs: list[dict] = [{"insertText": {"location": {"index": start}, "text": text}}]
        # Plain style for everything first (the inserted text inherits the previous style).
        reqs.append({"updateTextStyle": {"range": {"startIndex": start, "endIndex": pos},
                                         "textStyle": {}, "fields": "bold,italic,foregroundColor,fontSize"}})
        for kind, a, b in styles:
            if b <= a:
                continue
            style = ({"bold": True}, "bold") if kind == "head" else (
                {"foregroundColor": GREY, "fontSize": {"magnitude": 9, "unit": "PT"}}, "foregroundColor,fontSize")
            reqs.append({"updateTextStyle": {"range": {"startIndex": a, "endIndex": b},
                                             "textStyle": style[0], "fields": style[1]}})
        return reqs, pos - start

    def flush(self) -> None:
        with self.lock:
            lines, self.queue = self.queue, []
            relabels, self.relabels = self.relabels, []
        try:
            if relabels:
                reqs = [{"replaceAllText": {"containsText": {"text": f"  {old}: ", "matchCase": True},
                                            "replaceText": f"  {new}: "}} for old, new in relabels]
                self.svc.batchUpdate(documentId=self.doc_id, body={"requests": reqs}).execute()
                self._refresh_end()  # replacements changed the length
            if lines:
                reqs, added = self._requests_for(lines, self.end)
                self.svc.batchUpdate(documentId=self.doc_id, body={"requests": reqs}).execute()
                self.end += added
            self.failures = 0
        except Exception as e:  # network / quota: put everything back and retry next round
            with self.lock:
                self.queue[:0] = lines
                self.relabels[:0] = relabels
            self.failures += 1
            if self.failures in (1, 5, 20):
                print(f"  [google doc] write failed ({e}); will retry. Lines are safe in the local transcript.",
                      flush=True)
            try:
                self._refresh_end()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self.flush_seconds):
            self.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true", help="log in (opens the browser) and save token.json")
    ap.add_argument("--test", action="store_true", help="create a test document with sample lines")
    args = ap.parse_args()
    if args.login:
        login()
        print(f"Logged in; token saved to {TOKEN}")
    if args.test:
        w = DocWriter(docs_service(), title="Transcriber test")
        w.add(DocLine("00:00:05", "Speaker 1", "Thanks for your time, everyone.", "皆さんお疲れ様です"))
        w.add(DocLine("00:00:09", "Freyja", "Sounds good, let's start.", None))
        w.add(DocLine("00:00:14", "Speaker 2", "Testing isn't finished yet, so Friday may be difficult.",
                      "まだテストが終わっていないので金曜日のリリースは難しいかもしれません"))
        w.flush()
        w.relabel("Speaker 2", "Sato")
        w.flush()
        w.close()
        print(f"Test document: {w.url}")


if __name__ == "__main__":
    main()
