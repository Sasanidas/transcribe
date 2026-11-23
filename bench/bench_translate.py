"""Translation benchmark: runs a meeting-style dialogue (ReazonSpeech-style text: no
punctuation, kanji numerals, split sentences, one all-hiragana line) through each
candidate model, in order, with conversation history. Reports latency + VRAM and
prints outputs next to reference translations for review.

  uv run python bench/bench_translate.py
  uv run python bench/bench_translate.py --only plamo-2b
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transcribe.translate import LlamaServer, Translator  # noqa: E402

M = os.path.join(os.path.dirname(__file__), "..", "models", "llm")
CONFIGS = {
    "plamo-2b": (f"{M}/plamo-3-translate-2b-Q4_K_M.gguf", "plamo", ["--no-prefill-assistant"]),
    "plamo-8b-q3": (f"{M}/plamo-3-translate-8b-Q3_K_M.gguf", "plamo", ["--no-prefill-assistant"]),
    "qwen3.5-4b": (f"{M}/Qwen3.5-4B-Q4_K_M.gguf", "general", []),
}
GLOSSARY = {"先方": "the other party", "外注": "outsourcing"}


def vram() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS))
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "translation_cases.jsonl"))
    ap.add_argument("--gpu-layers", type=int, default=99)
    args = ap.parse_args()
    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8") if line.strip()]

    base = vram()
    results = {}
    for name in args.only or CONFIGS:
        path, backend, extra = CONFIGS[name]
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} missing")
            continue
        print(f"\n=== {name} ===", flush=True)
        server = LlamaServer(path, gpu_layers=args.gpu_layers, extra=extra)
        try:
            t = time.perf_counter()
            server.wait_ready()
            print(f"  ready in {time.perf_counter() - t:.1f}s, VRAM +{vram() - base} MiB", flush=True)
            tr = Translator(server, backend=backend, glossary=GLOSSARY)
            tr.translate("Tanaka", "よろしくお願いします")  # warm-up
            tr.history.clear()
            lats, outs = [], []
            for c in cases:
                t = time.perf_counter()
                en = tr.translate(c["speaker"], c["ja"])
                lats.append(time.perf_counter() - t)
                outs.append(en)
                print(f"  [{lats[-1]:.2f}s] {c['speaker']}: {c['ja']}\n      out: {en}\n      ref: {c['ref']}",
                      flush=True)
            results[name] = (statistics.median(lats), max(lats), vram() - base, outs)
        finally:
            server.stop()
            time.sleep(2)

    print("\n| model | median latency | worst | VRAM (LLM only) |\n|---|---|---|---|")
    for name, (med, worst, mem, _) in results.items():
        print(f"| {name} | {med:.2f}s | {worst:.2f}s | {mem} MiB |")
    out = os.path.join(os.path.dirname(__file__), "translation_outputs.json")
    json.dump({n: r[3] for n, r in results.items()}, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
