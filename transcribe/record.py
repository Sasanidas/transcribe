"""Record mic + meeting audio to WAV files with a live level meter.

  uv run python -m transcribe.record                 # record until Ctrl-C
  uv run python -m transcribe.record --duration 10   # record 10 s
  uv run python -m transcribe.record --list          # show available devices

Output: recordings/<YYYY-mm-dd_HH-MM-SS>/{mic,system}.wav (16 kHz mono)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import queue
import sys
import time

import numpy as np
import soundfile as sf

from .audio import SR, Chunk, ParecSource, default_devices, list_sources


def meter(level_db: float, width: int = 20) -> str:
    filled = int(np.clip((level_db + 60) / 60, 0, 1) * width)  # -60 dB .. 0 dB
    return "#" * filled + " " * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic", help="PulseAudio source for your microphone (default: system default)")
    ap.add_argument("--system", help="monitor source for meeting audio (default: default sink's monitor)")
    ap.add_argument("--out", default="recordings", help="output directory")
    ap.add_argument("--duration", type=float, help="stop after N seconds")
    ap.add_argument("--list", action="store_true", help="list audio sources and exit")
    args = ap.parse_args()

    if args.list:
        defaults = default_devices()
        for s in list_sources():
            tag = [k for k, v in defaults.items() if v == s]
            print(f"  {s}" + (f"   <- default {tag[0]}" if tag else ""))
        return

    devices = default_devices()
    devices["mic"] = args.mic or devices["mic"]
    devices["system"] = args.system or devices["system"]

    out_dir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    files = {n: sf.SoundFile(os.path.join(out_dir, f"{n}.wav"), "w", SR, 1, "PCM_16") for n in devices}
    sources = {n: ParecSource(n, d) for n, d in devices.items()}
    levels = {n: -90.0 for n in devices}

    q: queue.Queue[Chunk] = queue.Queue()
    for n, d in devices.items():
        print(f"{n:>6}: {d}")
    print(f"Recording to {out_dir}/  (Ctrl-C to stop)\n")
    for s in sources.values():
        s.start(q)

    start = time.monotonic()
    last_draw = 0.0
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            try:
                c = q.get(timeout=0.1)
            except queue.Empty:
                c = None
            if c is not None:
                files[c.source].write(c.samples)
                rms = float(np.sqrt(np.mean(c.samples ** 2)) + 1e-9)
                # fast attack, slow decay so speech is readable on the meter
                levels[c.source] = max(20 * np.log10(rms), levels[c.source] - 1.5)
            for s in sources.values():
                if s.error:
                    raise RuntimeError(f"{s.name}: {s.error}")
            now = time.monotonic()
            if now - last_draw > 0.1:
                last_draw = now
                el = int(now - start)
                line = "  ".join(f"{n} [{meter(levels[n])}] {levels[n]:6.1f} dB" for n in devices)
                sys.stdout.write(f"\r{el // 3600:02d}:{el % 3600 // 60:02d}:{el % 60:02d}  {line}")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources.values():
            s.stop()
        while not q.empty():  # flush what's left
            c = q.get_nowait()
            files[c.source].write(c.samples)
        print("\n")
        for n, s in sources.items():
            files[n].close()
            print(f"{n:>6}: {s.samples_read / SR:7.1f}s recorded, clock drift {s.drift * 1000:+.0f} ms")
        print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
