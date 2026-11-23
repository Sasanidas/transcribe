"""Live transcription to the terminal (and a JSONL transcript).

  uv run python -m transcribe.live                              # live: mic + meeting audio
  uv run python -m transcribe.live --files mic=a.wav system=b.wav   # replay recordings
  uv run python -m transcribe.live --files system=b.wav --fast       # replay as fast as possible

Sources named "mic" are labelled with --name (you); "system" with --others.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict

from .asr import Recognizer, Utterance
from .audio import Chunk, FileSource, ParecSource, default_devices
from .gdoc import DocLine, DocWriter, docs_service
from .mdlog import MarkdownTranscript
from .speakers import SpeakerTracker
from .translate import ROOT, LlamaServer, Translator
from .vad import Segment, Segmenter
from .voices import VoiceID

DEFAULT_LLM = os.path.join(ROOT, "models", "llm", "plamo-3-translate-2b-Q4_K_M.gguf")


def load_glossary(path: str | None) -> dict[str, str]:
    """One entry per line: `源語 => translation`. Lines starting with # are ignored."""
    if not path:
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=>" in line:
            src, dst = line.split("=>", 1)
            out[src.strip()] = dst.strip()
    return out


def hms(t: float) -> str:
    return f"{int(t) // 3600:02d}:{int(t) % 3600 // 60:02d}:{t % 60:04.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", metavar="NAME=PATH", help="replay WAV files instead of live capture")
    ap.add_argument("--fast", action="store_true", help="with --files: don't pace to real time")
    ap.add_argument("--mic", help="mic source (default: system default)")
    ap.add_argument("--system", help="meeting-audio monitor source (default: default sink's monitor)")
    ap.add_argument("--name", help="label for your mic (default: your Meet name from the add-on, else 'You')")
    ap.add_argument("--others", default="Others", help="label for the meeting-audio stream")
    ap.add_argument("--ja-engine", choices=["reazon", "kotoba"], default="reazon")
    ap.add_argument("--languages", default="en,ja", help="candidate languages for routing")
    ap.add_argument("--min-silence", type=float, default=0.5, help="pause (s) that ends a sentence")
    ap.add_argument("--out", default="transcripts", help="directory for the JSONL transcript")
    ap.add_argument("--no-translate", action="store_true", help="transcribe only")
    ap.add_argument("--llm", default=DEFAULT_LLM, help="GGUF translation model (PLaMo-3-translate)")
    ap.add_argument("--glossary", help="glossary file, one `源語 => translation` per line")
    ap.add_argument("--no-speakers", action="store_true", help="don't listen for the Firefox add-on")
    ap.add_argument("--meet-hints", action="store_true",
                    help="also use Meet tile/caption names from the add-on to name voices (off: Speaker 1, 2, ...)")
    ap.add_argument("--speaker-port", type=int, default=8765)
    ap.add_argument("--caption-lag", type=float, default=1.2,
                    help="how far (s) Meet captions trail the audio; also how long we wait for them")
    ap.add_argument("--meet-name", action="append", default=[],
                    help="your own display name in Meet (ignored as a hint); repeatable")
    ap.add_argument("--debug-speakers", action="store_true", help="print what the Firefox add-on sees")
    ap.add_argument("--doc", action="store_true", help="write the transcript live to a new Google Doc")
    ap.add_argument("--doc-id", help="append to this existing Google Doc instead (must be one this app created)")
    ap.add_argument("--doc-title", help="title for the new Google Doc")
    ap.add_argument("--no-voice-id", action="store_true", help="don't identify speakers by voice")
    ap.add_argument("--voice-threshold", type=float, default=0.55, help="similarity needed to match a voice")
    ap.add_argument("--voices", help="JSON file to load/save named voiceprints (stays local)")
    args = ap.parse_args()

    tracker = None if args.no_speakers else SpeakerTracker(
        args.speaker_port, lag=args.caption_lag, ignore_names=set(args.meet_name), debug=args.debug_speakers)
    voice_id = None if args.no_voice_id else VoiceID(threshold=args.voice_threshold, store=args.voices)

    doc = None
    if args.doc or args.doc_id:
        doc = DocWriter(docs_service(), doc_id=args.doc_id, title=args.doc_title)
        print(f"Google Doc: {doc.url}", flush=True)

    print("Loading models...", flush=True)
    t = time.perf_counter()
    server = translator = None
    if not args.no_translate:
        server = LlamaServer(args.llm, extra=["--no-prefill-assistant"])
    rec = Recognizer(ja_engine=args.ja_engine, languages=tuple(args.languages.split(",")))
    if server:
        server.wait_ready()
        translator = Translator(server, backend="plamo", glossary=load_glossary(args.glossary))
    print(f"Models loaded in {time.perf_counter() - t:.1f}s", flush=True)

    if args.files:
        sources = {}
        for spec in args.files:
            name, _, path = spec.partition("=")
            sources[name] = FileSource(name, path, realtime=not args.fast)
    else:
        dev = default_devices()
        sources = {"mic": ParecSource("mic", args.mic or dev["mic"]),
                   "system": ParecSource("system", args.system or dev["system"])}
    labels = {"mic": args.name, "system": args.others}
    segmenters = {n: Segmenter(n, min_silence=args.min_silence) for n in sources}

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jsonl")
    out_file = open(out_path, "a", encoding="utf-8")
    md = MarkdownTranscript(out_path[:-len(".jsonl")] + ".md", title=args.doc_title)
    outputs = [md] + ([doc] if doc else [])

    seg_q: queue.Queue[Segment | None] = queue.Queue()
    lags: list[float] = []
    wall_start = 0.0
    epoch_start = 0.0  # time.time() corresponding to audio t=0
    paced = not (args.files and args.fast)
    last_system_label: str | None = None

    def caption_hint(u: Utterance) -> str | None:
        if tracker is None or not paced or not tracker.connected:
            return None
        # Wait until captions covering the end of this segment have had time to arrive.
        lag = tracker.lag if tracker.captions_on else tracker.tile_lag  # tiles react much faster
        wait = epoch_start + u.end + lag - time.time()
        if wait > 0:
            time.sleep(wait)
        return tracker.who(epoch_start + u.start, epoch_start + u.end)

    last_system: tuple[str | None, int | None] = (None, None)
    my_label = {"name": args.name}  # set by --name or the `me=` command

    def current_me() -> str:
        return my_label["name"] or (tracker.my_name if tracker else None) or "You"

    def speaker_label(u: Utterance, seg: Segment) -> tuple[str, int | None]:
        nonlocal last_system
        default = labels.get(u.source, u.source)
        if u.source == "mic":
            return current_me(), None
        if u.source != "system":
            return default, None
        hint = caption_hint(u) if args.meet_hints else None
        name, vid = voice_id.identify(seg.samples, hint) if voice_id else (hint, None)
        if name is None:
            # Too short to judge by voice and no caption: most likely the previous speaker.
            name, vid = last_system
            if vid is not None and voice_id:
                name = next((v.name for v in voice_id.voices if v.id == vid), name)
        last_system = (name, vid)
        return name or default, vid

    def write_event(obj: dict) -> None:
        with out_lock:
            out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_file.flush()

    out_lock = threading.Lock()

    def commands() -> None:
        """Terminal commands during the meeting: `2=Tanaka`, `Speaker 2=Tanaka`, `list`."""
        while line := sys.stdin.readline():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("me="):
                before = current_me()
                my_label["name"] = line[3:].strip() or None
                if current_me() != before:
                    for o in outputs:
                        o.relabel(before, current_me())
                print(f"  your lines will be labelled {my_label['name'] or 'automatically'}", flush=True)
                write_event({"event": "me", "name": my_label["name"]})
            elif voice_id is None:
                continue
            elif line in ("list", "?"):
                print("Speakers:\n" + voice_id.describe(), flush=True)
            elif "=" in line:
                key, _, new = line.partition("=")
                msg, old_name, new_name = voice_id.rename(key, new)
                print(f"  {msg}", flush=True)
                if old_name and new_name and old_name != new_name:
                    for o in outputs:
                        o.relabel(old_name, new_name)
                write_event({"event": "rename", "from": key.strip(), "to": new.strip(), "result": msg})
            else:
                print("  commands: `2=Tanaka` renames Speaker 2, `me=Your Name` sets your label, "
                      "`list` shows speakers", flush=True)

    threading.Thread(target=commands, daemon=True, name="commands").start()

    def worker() -> None:
        while (seg := seg_q.get()) is not None:
            u: Utterance = rec(seg)
            if not u.text:
                continue
            label, vid = speaker_label(u, seg)
            en = u.text if u.lang == "en" else None
            mt_seconds = 0.0
            if translator and u.lang == "ja":
                t0 = time.perf_counter()
                try:
                    en = translator.translate(label, u.text)
                except Exception as e:  # keep transcribing even if translation fails
                    en = f"[translation failed: {e}]"
                mt_seconds = time.perf_counter() - t0
            lag = time.monotonic() - (wall_start + u.end) if paced else None
            if lag is not None:
                lags.append(lag)
            timing = f"lag {lag:.2f}s" if lag is not None else f"asr {u.asr_seconds:.2f}s mt {mt_seconds:.2f}s"
            print(f"[{hms(u.start)}] {label} ({u.lang} {u.lang_conf:.2f}, {timing}): {u.text}", flush=True)
            if en is not None and u.lang != "en":
                print(f"{'':14}-> {en}", flush=True)
            write_event({**asdict(u), "speaker": label, "voice_id": vid, "en": en, "mt_seconds": mt_seconds})
            for o in outputs:
                if u.lang == "ja" and en is not None:
                    o.add(DocLine(hms(u.start)[:8], label, en, u.text))
                else:  # English, or no translation available
                    o.add(DocLine(hms(u.start)[:8], label, u.text, None))

    wt = threading.Thread(target=worker, name="asr")
    wt.start()

    chunks: queue.Queue[Chunk] = queue.Queue()
    for s in sources.values():
        s.start(chunks)
    wall_start = min(s.wall_start for s in sources.values())
    epoch_start = time.time() - (time.monotonic() - wall_start)
    print(f"Listening ({', '.join(sources)}). Ctrl-C to stop. Transcript: {md.path}", flush=True)
    if voice_id:
        print("Rename a speaker any time by typing e.g. `2=Tanaka` + Enter; `list` shows speakers.\n", flush=True)

    try:
        while True:
            try:
                c = chunks.get(timeout=0.1)
            except queue.Empty:
                if all(s.done for s in sources.values()):
                    break
                for s in sources.values():
                    if s.error:
                        raise RuntimeError(f"{s.name}: {s.error}")
                continue
            for seg in segmenters[c.source].feed(c.samples):
                seg_q.put(seg)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for s in sources.values():
            s.stop()
        for sg in segmenters.values():
            for seg in sg.flush():
                seg_q.put(seg)
        seg_q.put(None)
        wt.join()
        out_file.close()
        if server:
            server.stop()
        if tracker:
            tracker.stop()
        if voice_id:
            voice_id.save()
        if doc:
            doc.close()
            print(f"Google Doc: {doc.url}")
        if lags:
            lags.sort()
            print(f"\nLag after end of speech: median {lags[len(lags) // 2]:.2f}s, worst {lags[-1]:.2f}s")
        print(f"Transcript saved to {md.path}")


if __name__ == "__main__":
    main()
