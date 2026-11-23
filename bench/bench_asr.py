"""ASR speed benchmark: Whisper variants (CPU/GPU) vs ReazonSpeech-k2 (CPU).

Measures per-utterance latency (what matters for real-time), real-time factor,
and Whisper's language-ID cost (needed to route English vs Japanese speech).

Usage:
  python bench/bench_asr.py --wavs samples/*.wav
  python bench/bench_asr.py --wavs samples/*.wav --only reazon-cpu turbo-cpu
"""

import argparse
import glob
import os
import statistics
import time

import numpy as np
import soundfile as sf

def preload_cuda_libs() -> None:
    """Load pip-installed cuBLAS/cuDNN so CTranslate2 finds them without LD_LIBRARY_PATH."""
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    root = list(spec.submodule_search_locations)[0]
    for lib in ["cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
                *[f"cudnn/lib/{n}" for n in sorted(os.listdir(os.path.join(root, "cudnn", "lib")))]]:
        path = os.path.join(root, lib)
        if os.path.exists(path) and ".so" in path:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


preload_cuda_libs()

SR = 16000
REAZON_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "reazonspeech-k2")


def load_wav(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono
    if sr != SR:
        # Simple linear resample; good enough for a speed benchmark.
        n = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio).astype(np.float32)
    return audio


class WhisperEngine:
    def __init__(self, model: str, device: str, threads: int, language: str | None):
        from faster_whisper import WhisperModel

        compute = "int8" if device == "cpu" else "int8_float16"
        self.model = WhisperModel(model, device=device, compute_type=compute, cpu_threads=threads)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        segments, info = self.model.transcribe(
            audio, language=self.language, task="transcribe", beam_size=5, vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segments)  # consume generator so timing is real
        return text.strip(), f"{info.language}({info.language_probability:.2f})"

    def detect_language(self, audio: np.ndarray) -> str:
        lang, prob, _ = self.model.detect_language(audio)
        return f"{lang}({prob:.2f})"


class ReazonEngine:
    def __init__(self, threads: int):
        import sherpa_onnx

        def find(prefix: str) -> str:
            hits = sorted(glob.glob(os.path.join(REAZON_DIR, "**", f"{prefix}*.int8.onnx"), recursive=True)) or \
                   sorted(glob.glob(os.path.join(REAZON_DIR, "**", f"{prefix}*.onnx"), recursive=True))
            if not hits:
                raise FileNotFoundError(f"No {prefix}*.onnx under {REAZON_DIR}")
            return hits[0]

        tokens = glob.glob(os.path.join(REAZON_DIR, "**", "tokens.txt"), recursive=True)[0]
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=find("encoder"), decoder=find("decoder"), joiner=find("joiner"), tokens=tokens,
            num_threads=threads, sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
        )

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        stream = self.rec.create_stream()
        stream.accept_waveform(SR, audio)
        self.rec.decode_stream(stream)
        return stream.result.text.strip(), "ja(fixed)"


CONFIGS = {
    # name: (factory, supports language-ID)
    "reazon-cpu": lambda t: ReazonEngine(t),
    "kotoba-cpu": lambda t: WhisperEngine("kotoba-tech/kotoba-whisper-v2.0-faster", "cpu", t, "ja"),
    "kotoba-gpu": lambda t: WhisperEngine("kotoba-tech/kotoba-whisper-v2.0-faster", "cuda", t, "ja"),
    "turbo-cpu": lambda t: WhisperEngine("large-v3-turbo", "cpu", t, None),
    "turbo-gpu": lambda t: WhisperEngine("large-v3-turbo", "cuda", t, None),
    "large-v3-cpu": lambda t: WhisperEngine("large-v3", "cpu", t, None),
    "large-v3-gpu": lambda t: WhisperEngine("large-v3", "cuda", t, None),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS), help="subset of configs to run")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (8 physical cores here)")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    clips = [(os.path.basename(p), load_wav(p)) for p in args.wavs]
    print(f"{len(clips)} clips, durations: " + ", ".join(f"{n}={len(a)/SR:.1f}s" for n, a in clips))

    summary = []
    for name in args.only or CONFIGS:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        try:
            engine = CONFIGS[name](args.threads)
        except Exception as e:  # missing model / no CUDA etc.
            print(f"  SKIP: {e}")
            continue
        print(f"  load: {time.perf_counter() - t0:.1f}s")
        engine.transcribe(clips[0][1])  # warm-up

        lats, rtfs = [], []
        for clip_name, audio in clips:
            dur = len(audio) / SR
            times = []
            for _ in range(args.runs):
                t = time.perf_counter()
                text, lang = engine.transcribe(audio)
                times.append(time.perf_counter() - t)
            lat = statistics.median(times)
            lats.append(lat)
            rtfs.append(lat / dur)
            print(f"  {clip_name} ({dur:.1f}s): {lat:.2f}s  RTF={lat/dur:.2f}  lang={lang}")
            print(f"    -> {text[:120]}")

        lid = ""
        if isinstance(engine, WhisperEngine):
            t = time.perf_counter()
            for _, audio in clips:
                engine.detect_language(audio)
            lid = f"{(time.perf_counter() - t) / len(clips):.2f}s"
        summary.append((name, statistics.median(lats), max(lats), statistics.median(rtfs), lid))
        del engine

    print("\n| config | median latency | worst latency | median RTF | lang-ID cost |")
    print("|---|---|---|---|---|")
    for name, med, worst, rtf, lid in summary:
        print(f"| {name} | {med:.2f}s | {worst:.2f}s | {rtf:.2f} | {lid or 'n/a'} |")


if __name__ == "__main__":
    main()
