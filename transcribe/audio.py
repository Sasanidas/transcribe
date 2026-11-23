"""Audio capture from PipeWire/PulseAudio via `parec`.

Two sources run in parallel:
  - "mic":    your microphone (everything on it is you)
  - "system": the monitor of your output device (everyone else in the meeting)

PipeWire resamples to 16 kHz mono s16le for us. Timestamps are derived from the
number of samples received, so both streams share one sample-accurate clock that
starts at session start.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

SR = 16000
BLOCK = 512  # samples per chunk (32 ms) — the frame size Silero VAD expects


@dataclass
class Chunk:
    source: str        # "mic" | "system"
    t0: float          # seconds since session start (sample-accurate)
    samples: np.ndarray  # float32 in [-1, 1], length BLOCK


def _pactl(*args: str) -> str:
    return subprocess.run(["pactl", *args], check=True, capture_output=True, text=True).stdout.strip()


def default_devices() -> dict[str, str]:
    return {"mic": _pactl("get-default-source"), "system": _pactl("get-default-sink") + ".monitor"}


def list_sources() -> list[str]:
    return [line.split("\t")[1] for line in _pactl("list", "short", "sources").splitlines() if "\t" in line]


class ParecSource:
    def __init__(self, name: str, device: str, latency_ms: int = 20):
        self.name, self.device, self.latency_ms = name, device, latency_ms
        self.samples_read = 0
        self.wall_start = 0.0
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self, out: queue.Queue[Chunk]) -> None:
        self._proc = subprocess.Popen(
            ["parec", f"--device={self.device}", "--format=s16le", f"--rate={SR}", "--channels=1",
             f"--latency-msec={self.latency_ms}", "--raw", f"--client-name=transcribe-{self.name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.wall_start = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(out,), daemon=True, name=f"parec-{self.name}")
        self._thread.start()

    def _run(self, out: queue.Queue[Chunk]) -> None:
        assert self._proc and self._proc.stdout
        nbytes = BLOCK * 2
        buf = b""
        while True:
            data = self._proc.stdout.read(nbytes - len(buf))
            if not data:
                err = self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
                if self._proc.poll() not in (None, 0, -15):
                    self.error = err or f"parec exited with {self._proc.returncode}"
                return
            buf += data
            if len(buf) < nbytes:
                continue
            samples = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
            buf = b""
            out.put(Chunk(self.name, self.samples_read / SR, samples))
            self.samples_read += BLOCK

    @property
    def drift(self) -> float:
        """Wall-clock time minus audio time; should stay near zero (a few tens of ms)."""
        return (time.monotonic() - self.wall_start) - self.samples_read / SR

    done = False  # live sources never finish on their own

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)


class FileSource:
    """Replays a WAV file as if it were a live stream (for testing)."""

    def __init__(self, name: str, path: str, realtime: bool = True):
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            n = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        self.name, self.audio, self.realtime = name, audio, realtime
        self.samples_read = 0
        self.wall_start = 0.0
        self.error: str | None = None
        self.done = False
        self._stop = threading.Event()

    def start(self, out: queue.Queue[Chunk]) -> None:
        self.wall_start = time.monotonic()
        threading.Thread(target=self._run, args=(out,), daemon=True, name=f"file-{self.name}").start()

    def _run(self, out: queue.Queue[Chunk]) -> None:
        n = len(self.audio) - len(self.audio) % BLOCK
        for i in range(0, n, BLOCK):
            if self._stop.is_set():
                break
            if self.realtime:
                delay = self.wall_start + (i + BLOCK) / SR - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            out.put(Chunk(self.name, i / SR, self.audio[i:i + BLOCK]))
            self.samples_read = i + BLOCK
        self.done = True

    def stop(self) -> None:
        self._stop.set()
