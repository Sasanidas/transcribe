"""Speaker-embedding benchmark.

Uses sherpa-onnx's labelled recordings (samples/speakers/<person>-*.wav) plus the
Japanese clips cut into halves (same speaker within a clip). Reports how well
same-speaker and different-speaker pairs separate, the best threshold, and speed.

  uv run python bench/bench_speaker.py
"""

import glob
import itertools
import os
import statistics
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = os.path.join(os.path.dirname(__file__), "..")
SR = 16000


def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    assert sr == SR, (path, sr)
    return a


def items():
    out = []
    for p in sorted(glob.glob(f"{ROOT}/samples/speakers/*.wav")):
        out.append((os.path.basename(p).split("-")[0], os.path.basename(p), load(p)))
    for p in sorted(glob.glob(f"{ROOT}/samples/[0-9].wav")):
        a = load(p)
        h = len(a) // 2
        name = "ja" + os.path.basename(p)[0]
        out += [(name, name + "a", a[:h]), (name, name + "b", a[h:])]
    return out


def main():
    data = items()
    print(f"{len(data)} clips, {len({d[0] for d in data})} speakers "
          f"(assumes each Japanese news clip is a different speaker)")
    for model in sorted(glob.glob(f"{ROOT}/models/speaker/*.onnx")):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=2)
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        embs, times = [], []
        for _, _, a in data:
            t = time.perf_counter()
            s = ext.create_stream()
            s.accept_waveform(SR, a)
            s.input_finished()
            e = np.array(ext.compute(s))
            times.append((time.perf_counter() - t) / (len(a) / SR))
            embs.append(e / np.linalg.norm(e))
        same, diff = [], []
        for i, j in itertools.combinations(range(len(data)), 2):
            sim = float(embs[i] @ embs[j])
            (same if data[i][0] == data[j][0] else diff).append(sim)
        # best accuracy threshold over all pairs
        best = max(((sum(s >= th for s in same) + sum(d < th for d in diff)) / (len(same) + len(diff)), th)
                   for th in np.arange(0.1, 0.9, 0.01))
        # nearest-neighbour identification: is each clip's most similar other clip the same person?
        nn_ok = 0
        for i in range(len(data)):
            sims = [(float(embs[i] @ embs[j]), j) for j in range(len(data)) if j != i]
            nn_ok += data[max(sims)[1]][0] == data[i][0]
        print(f"\n{os.path.basename(model)}  (dim {len(embs[0])})")
        print(f"  same-speaker sim: min {min(same):.2f} median {statistics.median(same):.2f}")
        print(f"  diff-speaker sim: median {statistics.median(diff):.2f} max {max(diff):.2f}")
        print(f"  gap (min same - max diff): {min(same) - max(diff):+.2f}")
        print(f"  best threshold {best[1]:.2f} -> pair accuracy {best[0]:.1%}")
        print(f"  nearest-neighbour ID: {nn_ok}/{len(data)}")
        print(f"  speed: {statistics.median(times) * 1000:.0f} ms per second of audio (CPU, 2 threads)")


if __name__ == "__main__":
    main()
