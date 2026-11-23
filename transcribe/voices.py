"""Speaker identification by voice (no dependency on Meet captions).

Every meeting-audio segment gets a voiceprint (CAM++ embedding, CPU, ~8 ms per
second of audio). Segments are clustered online: a voice close enough to a known
speaker joins it, otherwise a new "Speaker N" is created.

Names come from "hints" (the Firefox add-on's caption speaker, when captions
work). Each hint is a vote for the cluster; the cluster shows its majority name,
so one mis-timed caption can't rename someone, and once a voice has a name it
keeps it when captions stop working.

Optionally, named voiceprints are saved to a local JSON file and loaded in later
meetings so regular participants are recognised from the first sentence.
Voiceprints are biometric data: the file stays on this machine.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

from .audio import SR

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "speaker",
                     "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")


@dataclass
class Voice:
    id: int
    total: np.ndarray            # sum of unit embeddings (centroid direction)
    count: int = 0
    votes: Counter = field(default_factory=Counter)
    fixed_name: str | None = None   # loaded from the voices file
    manual_name: str | None = None  # set by the user during the meeting; always wins

    @property
    def centroid(self) -> np.ndarray:
        return self.total / (np.linalg.norm(self.total) + 1e-9)

    @property
    def name(self) -> str:
        if self.manual_name:
            return self.manual_name
        if self.votes:
            return self.votes.most_common(1)[0][0]
        return self.fixed_name or f"Speaker {self.id}"

    @property
    def named(self) -> bool:
        return bool(self.manual_name or self.votes or self.fixed_name)


class VoiceID:
    def __init__(self, threshold: float = 0.55, min_seconds: float = 1.0, model: str = MODEL,
                 store: str | None = None):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=2)
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.threshold, self.min_seconds = threshold, min_seconds
        self.voices: list[Voice] = []
        self.store = store
        self._next_id = 1
        self._lock = threading.Lock()
        if store and os.path.exists(store):
            for name, vec in json.load(open(store, encoding="utf-8")).items():
                v = np.asarray(vec, dtype=np.float32)
                self.voices.append(Voice(self._new_id(), v / np.linalg.norm(v), 1, fixed_name=name))

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def embed(self, audio: np.ndarray) -> np.ndarray:
        s = self.ext.create_stream()
        s.accept_waveform(SR, audio)
        s.input_finished()
        e = np.asarray(self.ext.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def identify(self, audio: np.ndarray, hint: str | None = None) -> tuple[str | None, int | None]:
        """(name or "Speaker N", voice id) for this segment; (hint, None) if too short to judge."""
        if len(audio) / SR < self.min_seconds:
            return hint, None  # trust the caption (if any) for very short clips
        e = self.embed(audio)
        with self._lock:
            best, sim = None, -1.0
            for v in self.voices:
                s = float(e @ v.centroid)
                if s > sim:
                    best, sim = v, s
            if best is None or sim < self.threshold:
                best = Voice(self._new_id(), np.zeros_like(e))
                self.voices.append(best)
            best.total = best.total + e
            best.count += 1
            if hint:
                best.votes[hint] += 1
            return best.name, best.id

    def _find(self, key: str) -> Voice | None:
        key = key.strip()
        if key.lower().startswith("speaker "):
            key = key[8:].strip()
        if key.isdigit():
            return next((v for v in self.voices if v.id == int(key)), None)
        return next((v for v in self.voices if v.name == key), None)

    def rename(self, key: str, new: str) -> tuple[str, str | None, str | None]:
        """Rename a voice ("2", "Speaker 2" or its current name). Renaming to a name another
        voice already has merges them (same person split into two clusters).
        Returns (message, old name, new name); names are None if nothing changed."""
        new = new.strip()
        with self._lock:
            v = self._find(key)
            if v is None:
                return f"no speaker '{key}' (type 'list' to see them)", None, None
            old = v.name
            other = next((o for o in self.voices if o is not v and o.name == new), None)
            if other:
                other.total, other.count = other.total + v.total, other.count + v.count
                other.votes.update(v.votes)
                other.manual_name = new
                self.voices.remove(v)
                return f"{old} merged into {new}", old, new
            v.manual_name = new
            return f"{old} -> {new}", old, new

    def describe(self) -> str:
        with self._lock:
            return "\n".join(f"  {v.id}: {v.name}  ({v.count} segments)" for v in self.voices) or "  (none yet)"

    def save(self) -> None:
        """Persist voiceprints that have a real name (not "Speaker N")."""
        if not self.store:
            return
        named = {}
        for v in self.voices:
            if v.named and v.count:
                named[v.name] = v.centroid.round(5).tolist()
        os.makedirs(os.path.dirname(os.path.abspath(self.store)), exist_ok=True)
        json.dump(named, open(self.store, "w", encoding="utf-8"), ensure_ascii=False)
