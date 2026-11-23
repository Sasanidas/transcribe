"""Per-stream speech segmentation with Silero VAD (sherpa-onnx, CPU).

A segment is emitted once the speaker pauses for `min_silence` seconds, so each
segment is (usually) a whole sentence — important for Japanese, where the verb
and negation come at the end. Monologues are force-split at `max_speech`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

from .audio import SR

VAD_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "vad", "silero_vad.onnx")


@dataclass
class Segment:
    source: str
    start: float  # seconds since session start
    end: float
    samples: np.ndarray = field(repr=False)


class Segmenter:
    """Silero decides *where* speech is; the audio itself is cut from our own buffer
    with padding, because the VAD's own samples clip word onsets and endings."""

    def __init__(self, source: str, threshold: float = 0.4, min_silence: float = 0.8,
                 min_speech: float = 0.25, max_speech: float = 20.0, model: str = VAD_MODEL,
                 pad_before: float = 0.5, pad_after: float = 0.3, keep_seconds: float = 60.0):
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        cfg.silero_vad.max_speech_duration = max_speech
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = SR
        cfg.num_threads = 1
        self.source = source
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_speech + 10)
        self.pad_before, self.pad_after = int(pad_before * SR), int(pad_after * SR)
        self.keep = int(keep_seconds * SR)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_offset = 0  # absolute sample index of buf[0]
        self.total = 0       # absolute samples received
        self.pending: list[tuple[int, int]] = []  # (start, end) waiting for pad_after audio

    def feed(self, samples: np.ndarray) -> list[Segment]:
        self.buf = np.concatenate([self.buf, samples])
        self.total += len(samples)
        if len(self.buf) > self.keep:
            cut = len(self.buf) - self.keep
            self.buf = self.buf[cut:]
            self.buf_offset += cut
        self.vad.accept_waveform(samples)
        return self._drain()

    def flush(self) -> list[Segment]:
        self.vad.flush()
        return self._drain(final=True)

    def _drain(self, final: bool = False) -> list[Segment]:
        while not self.vad.empty():
            f = self.vad.front
            self.pending.append((f.start, f.start + len(f.samples)))
            self.vad.pop()
        out, still = [], []
        for start, end in self.pending:
            s = max(start - self.pad_before, self.buf_offset)
            e = end + self.pad_after
            if e > self.total and not final:
                still.append((start, end))  # wait for the trailing padding to arrive
                continue
            e = min(e, self.total)
            out.append(Segment(self.source, s / SR, e / SR, self.buf[s - self.buf_offset:e - self.buf_offset].copy()))
        self.pending = still
        return out
