"""Japanese -> English translation with a local LLM served by llama.cpp.

llama-server runs as a subprocess on localhost and keeps the prompt's KV cache
between requests, so re-sending the (append-only) conversation history is cheap.

Backends:
  plamo   - PLaMo-3-translate: translation-specialised; history passed as
            user/assistant pairs, glossary via its "[plamo-translate]" system message.
  general - any instruct model (e.g. Qwen3.5): instructions + transcript context.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_SERVER = os.path.join(ROOT, "tools", "llama.cpp", "llama-b12521", "llama-server")
NVIDIA_LIBS = os.path.join(ROOT, ".venv", "lib", "python3.12", "site-packages", "nvidia")


class LlamaServer:
    def __init__(self, model: str, port: int = 8090, ctx: int = 8192, gpu_layers: int = 99,
                 extra: list[str] | None = None):
        self.url = f"http://127.0.0.1:{port}"
        libs = [os.path.join(NVIDIA_LIBS, d, "lib") for d in sorted(os.listdir(NVIDIA_LIBS))
                if os.path.isdir(os.path.join(NVIDIA_LIBS, d, "lib"))]
        env = {**os.environ, "LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])}
        cmd = [LLAMA_SERVER, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx),
               "-ngl", str(gpu_layers), "-fa", "on", "--jinja", "-np", "1", *(extra or [])]
        self.log = open(os.path.join(ROOT, "llama-server.log"), "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float = 180) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see llama-server.log")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become ready")

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


@dataclass
class Line:
    speaker: str
    ja: str
    en: str


class Translator:
    """Keeps a rolling history of translated lines and translates new ones with it."""

    def __init__(self, server: LlamaServer, backend: str = "plamo", history: int = 12,
                 glossary: dict[str, str] | None = None):
        self.server, self.backend = server, backend
        self.history: deque[Line] = deque(maxlen=history)
        self.glossary = glossary or {}

    def translate(self, speaker: str, ja: str) -> str:
        msgs = self._plamo_messages(ja) if self.backend == "plamo" else self._general_messages(speaker, ja)
        body = {"messages": msgs, "temperature": 0, "max_tokens": 256, "cache_prompt": True}
        if self.backend == "plamo":
            body["stop"] = ["<|plamo:tag|>", "<|plamo:bos|>"]
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        en = self.server.post("/v1/chat/completions", body)["choices"][0]["message"]["content"].strip()
        self.history.append(Line(speaker, ja, en))
        return en

    def _plamo_messages(self, ja: str) -> list[dict]:
        system = "[plamo-translate]\ntarget_lang=en\nsource_lang=ja"
        for src, dst in self.glossary.items():
            system += f"\nglossary={src} => {dst}"
        msgs = [{"role": "system", "content": system}]
        for line in self.history:
            msgs += [{"role": "user", "content": line.ja}, {"role": "assistant", "content": line.en}]
        return msgs + [{"role": "user", "content": ja}]

    def _general_messages(self, speaker: str, ja: str) -> list[dict]:
        system = (
            "You are a professional Japanese-to-English interpreter for a live business meeting. "
            "The Japanese comes from speech recognition: it has no punctuation, may write numbers in kanji, "
            "may be written in hiragana, and a sentence may be split across consecutive lines. "
            "Translate ONLY the final line into natural, accurate English. Use the earlier lines to resolve "
            "omitted subjects, pronouns and references, and keep the speaker's level of politeness and "
            "intent (e.g. soft refusals). Do not add information. Output only the English translation."
        )
        if self.glossary:
            system += "\nGlossary (always use these translations):\n" + "\n".join(
                f"- {s} => {d}" for s, d in self.glossary.items())
        context = "\n".join(f"{line.speaker}: {line.ja}\n  -> {line.en}" for line in self.history)
        user = (f"Conversation so far:\n{context}\n\n" if context else "") + f"Translate this line:\n{speaker}: {ja}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
