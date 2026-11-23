"""Language routing + speech recognition.

  segment -> Whisper language ID (GPU, restricted to candidate languages)
          -> "ja": ReazonSpeech (CPU) or kotoba-whisper (GPU)
          -> other: Whisper transcribe (GPU)

Whisper's unrestricted language ID mislabels Japanese as Korean (see
bench/RESULTS.md), so we only compare the configured candidate languages.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass

import numpy as np

from .audio import SR
from .cuda_libs import preload
from .vad import Segment

preload()

from faster_whisper import WhisperModel  # noqa: E402  (needs CUDA libs preloaded)

REAZON_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "reazonspeech-k2")
KOTOBA = "kotoba-tech/kotoba-whisper-v2.0-faster"


@dataclass
class Utterance:
    source: str
    start: float
    end: float
    lang: str
    lang_conf: float
    text: str
    asr_seconds: float


class ReazonJa:
    def __init__(self, threads: int):
        import sherpa_onnx

        def find(prefix: str) -> str:
            hits = sorted(glob.glob(os.path.join(REAZON_DIR, "**", f"{prefix}*.int8.onnx"), recursive=True))
            if not hits:
                raise FileNotFoundError(f"ReazonSpeech model not found under {REAZON_DIR}; run scripts/fetch_models.sh")
            return hits[0]

        tokens = glob.glob(os.path.join(REAZON_DIR, "**", "tokens.txt"), recursive=True)[0]
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=find("encoder"), decoder=find("decoder"), joiner=find("joiner"), tokens=tokens,
            num_threads=threads, sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
        )

    def __call__(self, audio: np.ndarray) -> str:
        stream = self.rec.create_stream()
        stream.accept_waveform(SR, audio)
        self.rec.decode_stream(stream)
        return stream.result.text.strip()


def whisper_text(model: WhisperModel, audio: np.ndarray, lang: str) -> str:
    segs, _ = model.transcribe(
        audio, language=lang, task="transcribe", beam_size=5, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True,
    )
    # Drop segments Whisper itself thinks are non-speech (classic "Thank you." hallucinations).
    keep = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
    sep = "" if lang in ("ja", "zh") else " "
    return sep.join(s.text.strip() for s in keep).strip()


class Recognizer:
    def __init__(self, ja_engine: str = "reazon", whisper_model: str = "large-v3-turbo",
                 device: str = "cuda", cpu_threads: int = 6, languages: tuple[str, ...] = ("en", "ja")):
        compute = "int8_float16" if device == "cuda" else "int8"
        self.whisper = WhisperModel(whisper_model, device=device, compute_type=compute, cpu_threads=cpu_threads)
        if ja_engine == "reazon":
            self.ja = ReazonJa(cpu_threads)
        elif ja_engine == "kotoba":
            kotoba = WhisperModel(KOTOBA, device=device, compute_type=compute, cpu_threads=cpu_threads)
            self.ja = lambda a: whisper_text(kotoba, a, "ja")
        else:
            raise ValueError(f"unknown ja_engine {ja_engine!r}")
        self.languages = languages
        self.last_lang: dict[str, str] = {}

    def detect(self, audio: np.ndarray, source: str) -> tuple[str, float]:
        _, _, probs = self.whisper.detect_language(audio)
        p = {lang: x for lang, x in probs if lang in self.languages}
        total = sum(p.values()) or 1e-9
        lang = max(p, key=p.get)
        conf = p[lang] / total
        # Very short or ambiguous clips ("うん", "OK") -> trust this speaker's previous language.
        if (len(audio) / SR < 1.0 or conf < 0.7) and source in self.last_lang:
            lang = self.last_lang[source]
        self.last_lang[source] = lang
        return lang, conf

    def __call__(self, seg: Segment) -> Utterance:
        t = time.perf_counter()
        lang, conf = self.detect(seg.samples, seg.source)
        text = self.ja(seg.samples) if lang == "ja" else whisper_text(self.whisper, seg.samples, lang)
        return Utterance(seg.source, seg.start, seg.end, lang, conf, text, time.perf_counter() - t)
