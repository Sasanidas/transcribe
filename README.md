# transcribe — local, real-time Meet transcription with Japanese → English translation

Transcribes Google Meet calls live, labels who is speaking, and translates Japanese to English,
entirely on this laptop. Output goes to the terminal, a local Markdown/JSONL transcript and,
optionally, a Google Doc.

```
[00:03:12.4] Speaker 1 (ja 0.98, lag 0.71s): 金曜日のリリースは難しいかもしれません
              -> A release on Friday might be difficult.
[00:03:15.0] Freyja (en 0.99, lag 0.62s): OK, let's move it to Monday then.
```

- **Real time, one sentence at a time:** a line appears about 1 s after the speaker pauses.
  It works per sentence, not per word, because Japanese puts the verb, negation and question
  marker at the end (see [Why not word by word?](#why-not-word-by-word)).
- **Fully local:** speech recognition, translation and speaker identification run on the
  GPU/CPU of this machine. Nothing leaves the laptop unless you turn on the Google Doc output.
- **Speakers:** your own speech comes from your microphone, so it is labelled with your name.
  Everyone else is told apart **by voice** as `Speaker 1`, `Speaker 2`, …, and you can rename
  them during the call.

Developed and measured on an RTX 4050 Laptop GPU (6 GB), a Ryzen 7 7435HS (8 cores), 24 GB RAM,
Ubuntu with PipeWire, and Python 3.12.

---

## Contents
- [Quick start](#quick-start)
- [Using it](#using-it)
- [How it works](#how-it-works)
  - [Pipeline overview](#pipeline-overview)
  - [1. Audio capture](#1-audio-capture)
  - [2. Speech detection and sentence segmentation](#2-speech-detection-and-sentence-segmentation)
  - [3. Language routing](#3-language-routing)
  - [4. Speech recognition](#4-speech-recognition)
  - [5. Translation](#5-translation)
  - [6. Speaker identification](#6-speaker-identification)
  - [7. Firefox add-on](#7-firefox-add-on)
  - [8. Outputs](#8-outputs)
  - [Threads and timing](#threads-and-timing)
- [Resource use and latency](#resource-use-and-latency)
- [Benchmarks](#benchmarks)
- [Project layout](#project-layout)
- [Privacy](#privacy)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Licenses of models and tools](#licenses-of-models-and-tools)

---

## Quick start

### Requirements
- Linux with **PipeWire** (or PulseAudio) and `parec` (`pulseaudio-utils`)
- NVIDIA GPU with ≥ 4 GB VRAM and a driver supporting CUDA 12.8 (tested: driver 580).
  No system CUDA toolkit is needed; the CUDA libraries come from pip.
- [`uv`](https://docs.astral.sh/uv/) for the Python environment
- About 8 GB of disk for models and tools

### Install
```bash
cd ~/Programming/transcribe
uv sync                        # Python dependencies incl. CUDA runtime libs (.venv/)
./scripts/fetch_models.sh      # models + llama.cpp (~6 GB); skips what's already there
```
The Whisper model (`large-v3-turbo`) downloads from Hugging Face on first run.

### Run
Join the Meet call **with headphones**, then:
```bash
uv run python -m transcribe.live --name "Your Name"
```
Press **Ctrl-C** to stop. The transcript is saved under `transcripts/`.

---

## Using it

### Before a call
- **Wear headphones.** With speakers, your mic picks up the other participants and their words
  appear twice, once labelled as you.
- **Close other audio** (music, videos). Everything the computer plays is captured.
- **Avoid Bluetooth headset mics** if you can. While the mic is active, most switch to a
  low-quality "headset" profile, which hurts recognition.
- **Tell participants** the call is being transcribed.

### Commands during the call
Type a command in the terminal and press Enter:

| Command | Effect |
|---|---|
| `2=Tanaka` or `Speaker 2=Tanaka` | Rename a speaker. Lines already written are updated too |
| `3=Tanaka` (when Tanaka exists) | Merge Speaker 3 into Tanaka (one person split into two voices) |
| `me=Your Name` | Set the label for your own lines |
| `list` or `?` | Show the current speakers and how many lines each has |

### Options (`uv run python -m transcribe.live --help`)

| Option | Default | Purpose |
|---|---|---|
| `--name NAME` | Meet name from the add-on, else `You` | Label for your mic |
| `--voices FILE` | off | Load/save named voiceprints so regular people are recognised by name |
| `--glossary FILE` | none | Fixed translations, one `源語 => translation` per line |
| `--doc` / `--doc-id ID` | off | Also write to a new or existing Google Doc ([setup](docs/google-docs-setup.md)) |
| `--doc-title TITLE` | date/time | Title of the transcript (Markdown and Google Doc) |
| `--no-translate` | off | Transcription only |
| `--ja-engine reazon\|kotoba` | `reazon` | Japanese recogniser: ReazonSpeech (CPU) or kotoba-whisper (GPU) |
| `--languages en,ja` | `en,ja` | Languages the router chooses between |
| `--min-silence SEC` | `0.5` | Pause that ends a sentence (longer = fewer split sentences, more delay) |
| `--voice-threshold X` | `0.55` | Cosine similarity needed to match a known voice |
| `--mic SRC` / `--system SRC` | system defaults | Audio devices (`python -m transcribe.record --list`) |
| `--files mic=a.wav system=b.wav` | live capture | Replay recordings instead of live audio (`--fast` = no real-time pacing) |
| `--meet-hints` | off | Also use Meet tile/caption names from the add-on to name voices |
| `--no-speakers` | off | Don't listen for the Firefox add-on |
| `--debug-speakers` | off | Print what the add-on sees |
| `--llm FILE` | PLaMo 2B | GGUF translation model |

### Other tools
```bash
uv run python -m transcribe.record              # record mic + meeting audio to recordings/<date>/
uv run python -m transcribe.record --list       # list audio devices
uv run python -m transcribe.gdoc --login        # one-time Google login (for --doc)
uv run python -m transcribe.gdoc --test         # create a test Google Doc
```

---

## How it works

### Pipeline overview

```
 mic (you) ──────┐                                          ┌─ terminal
                 │   parec    ┌──────────┐  ┌────────────┐  │
 speaker monitor ┴─ 16 kHz ──▶│Silero VAD│─▶│ Whisper LID│  ├─ transcripts/<date>.md
 (everyone else)    mono      │ per      │  │ {en, ja}   │  │
                              │ stream   │  └─────┬──────┘  ├─ transcripts/<date>.jsonl
                              └──────────┘        │         │
                                       ja ┌───────┴──────┐ en  └─ Google Doc (optional)
                                          ▼              ▼                 ▲
                                    ReazonSpeech    Whisper turbo          │
                                       (CPU)           (GPU)               │
                                          │              │                 │
                                          ▼              │                 │
                                 PLaMo-3-translate 2B    │                 │
                                 llama.cpp (GPU)         │                 │
                                          │              │                 │
                                          └──────┬───────┘                 │
                                                 ▼                         │
                          speaker label: mic → your name;                  │
                          meeting audio → CAM++ voiceprint clustering ─────┘
```

| Stage | Technology | Runs on | Code |
|---|---|---|---|
| Audio capture | PipeWire/PulseAudio via `parec` | CPU | `transcribe/audio.py` |
| Speech detection | Silero VAD (sherpa-onnx) | CPU | `transcribe/vad.py` |
| Language routing | Whisper large-v3-turbo language ID (faster-whisper / CTranslate2) | GPU | `transcribe/asr.py` |
| Japanese recognition | ReazonSpeech-k2 Zipformer transducer (sherpa-onnx) | CPU | `transcribe/asr.py` |
| Other languages | Whisper large-v3-turbo (faster-whisper) | GPU | `transcribe/asr.py` |
| Translation | PLaMo-3-translate 2B (GGUF Q4_K_M) on llama.cpp `llama-server` | GPU | `transcribe/translate.py` |
| Speaker ID | 3D-Speaker CAM++ embeddings (sherpa-onnx) + online clustering | CPU | `transcribe/voices.py` |
| Meet integration | Firefox WebExtension (MV2) + local HTTP endpoint | browser | `extension/firefox/`, `transcribe/speakers.py` |
| Outputs | Markdown, JSONL, Google Docs API | – | `transcribe/mdlog.py`, `transcribe/gdoc.py`, `transcribe/live.py` |

### 1. Audio capture
`transcribe/audio.py`

- **Technology:** two `parec` processes reading from PipeWire through its PulseAudio-compatible interface.
  - **`mic`:** your default input device (`pactl get-default-source`).
  - **`system`:** the *monitor* of your default output (`<sink>.monitor`), i.e. whatever the
    computer plays, which during a call is the other participants.
- **Why this way:** it needs no integration with Meet, no bot and no API, and it works the same for
  any browser or meeting app. Because the two streams are separate, "you vs. everyone else" is
  known for free.
- **Format:** PipeWire resamples to **16 kHz mono signed 16-bit** (`--format=s16le --rate=16000
  --channels=1`, 20 ms latency). Samples are read in **512-sample (32 ms) chunks**, the window
  size Silero VAD expects, and converted to float32 in [-1, 1].
- **Clock:** each chunk's timestamp is *samples received ÷ 16000*, not wall-clock time, so both
  streams share one sample-accurate timeline starting at session start. A constant startup offset
  of about 0.1 s against the wall clock was measured.
- `FileSource` replays WAV files through the same interface, optionally paced to real time. All
  tests and the `--files` option use it.

### 2. Speech detection and sentence segmentation
`transcribe/vad.py`

- **Technology:** [Silero VAD](https://github.com/snakers4/silero-vad), the sherpa-onnx ONNX build,
  one detector per stream, on the CPU.
- **Parameters:**
  - speech threshold **0.4**;
  - a segment ends after **`--min-silence` (0.5 s)** of silence;
  - minimum speech **0.25 s**;
  - segments are force-split at **20 s** for monologues.
- **Padding fix:** Silero's own segment audio clips the start and end of words; in testing it lost
  「早く」 and 「ヤンバル」 and produced an invented word. Silero is therefore only used to decide
  *where* speech is, and the audio is cut from our own 60 s ring buffer with **0.5 s padding before
  and 0.3 s after**. A segment is released once its trailing padding has arrived.
- **Why whole sentences:** a segment ends at a pause, so it is usually a whole sentence or clause.
  This matters for Japanese (see [below](#why-not-word-by-word)).

### 3. Language routing
`transcribe/asr.py` (`Recognizer.detect`)

- **Technology:** Whisper **large-v3-turbo**'s language-ID head, via faster-whisper
  (CTranslate2, `int8_float16` on the GPU). It costs about 0.23 s per segment.
- **Restricted choice:** Whisper's unrestricted language ID labelled 4 of 5 Japanese test clips as
  **Korean with ~99% confidence**, while still transcribing them correctly as Japanese. The router
  therefore compares only the candidate languages (`--languages`, default `en,ja`) and normalises
  their probabilities.
- **Fallback:** segments shorter than 1 s, or with confidence below 0.7 (「はい」, "OK"), reuse
  that stream's previous language.
- **Routing:** Japanese goes to ReazonSpeech and then translation. English (or any other candidate
  language) goes to Whisper and is written as spoken.

### 4. Speech recognition
`transcribe/asr.py`

**Japanese: ReazonSpeech-k2 v2** (`sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01`)
- **Model:** a Zipformer transducer (about 160M parameters) trained on the 35,000-hour
  ReazonSpeech Japanese corpus. It runs int8 through sherpa-onnx on 6 CPU threads with greedy
  decoding.
- **Why:** it takes **0.1–0.2 s per sentence on the CPU** and leaves the GPU to the translation
  model. On the test clips it got the same words as kotoba-whisper.
- **Quirks:** no punctuation, numbers written in kanji (十八歳, 百億円), and occasionally a line in
  plain hiragana. The translation model copes with all three.
- **Alternative:** `--ja-engine kotoba` uses **kotoba-whisper-v2.0** (Whisper distilled for
  Japanese) on the GPU: 0.3 s per sentence, with punctuation and Arabic numerals, but about 1 GB
  more VRAM.

**English and other languages: Whisper large-v3-turbo** (faster-whisper)
- **Settings:** beam size 5, `condition_on_previous_text=False`, no timestamps.
- **Hallucination filter:** a segment is dropped when Whisper itself rates it as probably
  non-speech (`no_speech_prob > 0.6` and `avg_logprob < -1`). This targets the classic made-up
  "Thank you." on silence.

**Whisper on the CPU was rejected.** The encoder always processes a 30 s window, so even a short
sentence costs about 4 s (kotoba) or 9 s (turbo) on this CPU. See [Benchmarks](#benchmarks).

**CUDA without a toolkit:** `transcribe/cuda_libs.py` preloads cuBLAS and cuDNN from the pip
packages `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` with `ctypes` (`RTLD_GLOBAL`) before
CTranslate2 loads, so no system CUDA install or `LD_LIBRARY_PATH` is needed.

### 5. Translation
`transcribe/translate.py`

**Model: [PLaMo-3-translate 2B](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF)**
(Preferred Networks), as GGUF **Q4_K_M** (1.8 GB).
- **What it is:** a model specialised in translation (53 languages, strongest for Japanese ↔
  English), with built-in support for conversation history and a glossary.
- **Why:** it beat a general instruct model (Qwen3.5-4B) and its own 8B version on our meeting
  test set, while being the fastest and smallest of the three (see [Benchmarks](#benchmarks)).

**Runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`**, official release
**b12521**, Ubuntu CUDA 12.8 build, in `tools/llama.cpp/`.
- **How it runs:** `LlamaServer` starts it as a subprocess on `127.0.0.1:8090`. All layers are on
  the GPU (`-ngl 99`), with flash attention on, an 8192-token context and one slot (`-np 1`).
  It uses `--jinja --no-prefill-assistant` so the model's built-in chat template is applied.
- **CUDA libraries:** the server gets `LD_LIBRARY_PATH` pointing at the pip CUDA libraries in
  `.venv`, so it shares the same CUDA runtime as Whisper.
- **Logs:** `llama-server.log`.

**Each request** goes to the OpenAI-style `/v1/chat/completions` endpoint, at temperature 0, with
a 256-token limit and PLaMo's stop tokens.
- **System message:** `[plamo-translate]` with `target_lang=en`, `source_lang=ja`, plus one
  `glossary=源語 => translation` line per glossary entry.
- **History:** the last **12 translated lines** as user/assistant pairs. This lets the model
  resolve omitted subjects and references; for example it turned 「…そうですが」 correctly into
  "*They* said…".
- **Speed:** `cache_prompt` keeps the KV cache between requests. The history only grows at the
  end, so each request mostly processes just the new line: about **0.2 s per line**.

Only lines routed as Japanese are translated. If translation fails, the line is still written,
with a `[translation failed: …]` note.

<a id="why-not-word-by-word"></a>
**Why not word by word?** Japanese puts the deciding words last: 「その案には賛成でき**ない**とは
言っていません」 reads, word by word, "that proposal… agree… *can't*… *not saying*". The meaning
flips twice in the last few words. Translating partial sentences would show guesses that later
reverse, so the pipeline waits for the pause that ends the sentence.

### 6. Speaker identification
`transcribe/voices.py`

- **Your lines:** everything on the mic stream is you. The label comes from `--name`, `me=`, or
  the Meet name the Firefox add-on reports, else `You`.
- **Everyone else:** told apart by voice.
  - **Embeddings:** **3D-Speaker CAM++** (`campplus_sv_zh_en_16k-common_advanced`, trained on
    Chinese and English, 28 MB) through sherpa-onnx's `SpeakerEmbeddingExtractor`. It produces a
    192-dimensional, L2-normalised voiceprint per segment, at about **8 ms per second of audio** on
    2 CPU threads.
  - **Online clustering:** each segment's voiceprint is compared by cosine similarity with every
    known voice's centroid (the normalised sum of its voiceprints).
    - Similarity ≥ **0.55** (`--voice-threshold`): it joins that voice and updates the centroid.
    - Otherwise: a new `Speaker N` is created.
  - **Short segments:** below **1.0 s** there is too little audio for a reliable voiceprint, so the
    segment is attributed to the previous meeting speaker.
- **Renaming:** `2=Tanaka` sets a manual name that always takes priority. Renaming to a name that
  another voice already has **merges** the two clusters (sums of voiceprints, counts and votes).
- **Remembering voices (`--voices FILE`):** named centroids are saved as JSON on exit and loaded
  as known voices next time, so regular participants get their name from the first sentence.
- **Why CAM++:** on 24 labelled clips from 8 speakers it identified 24/24 correctly with a clear
  margin. TitaNet-small also got 24/24 but with a smaller margin; WeSpeaker ResNet34 got 22/24.
  Voice embeddings are largely language-independent.
- **Optional, `--meet-hints`:** names from the Firefox add-on (Meet's highlighted tile or captions)
  count as **votes** for a voice, and a voice shows its majority name. This is off by default
  because Meet captions turned out to be unreliable.

### 7. Firefox add-on
`extension/firefox/` (WebExtension, Manifest V2) + `transcribe/speakers.py`

- **What it's for now:** mainly reporting **your Meet display name** for your own lines. Speaker
  hints for others are only used with `--meet-hints`.
- **`content.js`** (runs on `meet.google.com`), checked every 200 ms:
  - **Tiles:** participant tiles are found by `[data-participant-id]`. A tile counts as "speaking"
    when a `MutationObserver` sees at least 3 class/style/child changes inside it within 600 ms,
    which catches Meet's speaking animation without depending on Meet's generated class names.
    The name is read from the tile's first plausible text.
  - **Captions** (if on): the caption panel `div[role="region"][tabindex="0"]`; each block is
    name + growing text. This is the same approach as the MIT-licensed
    [TranscripTonic](https://github.com/bpetrynski/transcriptonic).
  - **Your name:** read from the pre-join screen, the Google account button ("Google Account:
    Name", email discarded) or the People panel ("Name (You)" / 「名前（あなた）」). It is stored
    in `browser.storage.local`, so it is only needed once.
  - **Only names and timestamps leave the page,** never caption text.
- **`background.js`** forwards the add-on's messages with `POST http://127.0.0.1:8765/events`.
- **`speakers.py`** runs a small `ThreadingHTTPServer` on 127.0.0.1:8765.
  - **Who it accepts:** only JSON (`Content-Type: application/json`) from extension origins
    (`moz-extension://` / `chrome-extension://`). Ordinary web pages can't send that without a
    CORS preflight, which the server never answers, so they're rejected.
  - **Hint windows:** tile hints are matched to a segment over `[start, end + 0.4 s]`, captions
    over `[start, end + 1.2 s]` (captions lag the audio). With `--meet-hints` the app waits for
    that window before labelling.
  - **Your own name** is never used as a hint for someone else.
- **Install:** `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on…** →
  `extension/firefox/manifest.json`. A temporary add-on is removed, along with its storage, when
  Firefox restarts. To keep it permanently, sign it as *unlisted* through addons.mozilla.org
  (free, private, but uploads the code to Mozilla), or use Firefox Developer Edition.
- **Caveat:** Meet's page structure is undocumented and changes, so the selectors may need updating.

### 8. Outputs

**Terminal:** one line per segment: time, speaker, detected language and confidence, lag (or
processing times with `--fast`), the original text, and `->` with the translation.

**`transcripts/<date>.md`** (`mdlog.py`): a readable transcript, written live.
- Each line is formatted as `**hh:mm:ss  Speaker:** English`, followed by `> 日本語` for
  translated lines.
- A rename rewrites the file so earlier lines get the new name.

**`transcripts/<date>.jsonl`**: the full record, one JSON object per line.
- **Utterances:** `source` (`mic`/`system`), `start`/`end` (s), `lang`, `lang_conf`, `text`
  (original), `en`, `speaker`, `voice_id`, `asr_seconds`, `mt_seconds`.
- **Events:** `{"event": "rename" | "me", …}` records.

**Google Doc** (`gdoc.py`, `--doc`): Google Docs API v1 via `google-api-python-client`.
- **Access:** OAuth desktop flow (`google-auth-oauthlib`), scope **`drive.file`** only: the app
  can access only documents it created. The token is cached in `token.json` (mode 600).
- **Writing:** lines are queued and flushed every **3 s** in a single `batchUpdate` per flush.
  The flush inserts the text at a tracked end index, resets inherited styles, then makes the
  speaker/time **bold** and the Japanese original **grey, 9 pt**.
- **Indices:** Google Docs counts positions in **UTF-16 code units**, so lengths are computed with
  `len(s.encode("utf-16-le")) // 2`. Emoji count as 2.
- **Renames:** `replaceAllText` on `"  Old: "` → `"  New: "` (exact enough that `Speaker 2` doesn't
  touch `Speaker 12`), then the end index is re-read.
- **Failures:** on network or quota errors the batch is put back and retried; the local files
  always have everything.
- **Testing:** `bench/fake_docs.py` is an in-memory fake of the API with the real index rules,
  used to test all of this offline.
- **Setup:** [docs/google-docs-setup.md](docs/google-docs-setup.md).

### Threads and timing
`transcribe/live.py`

- **Capture threads (one per stream)** read `parec` and push 32 ms chunks onto a queue.
- **Main thread** feeds each chunk to that stream's VAD and puts finished segments on the
  segment queue.
- **Worker thread** processes segments in order: language ID → recognition → (Japanese)
  translation → speaker label → outputs. A single worker keeps GPU use serial and the output
  ordered.
- **Command thread** reads `2=Tanaka`, `me=…` and `list` from stdin.
- **Google Doc thread** flushes the queue every 3 s.
- **llama-server** (separate process) and **add-on HTTP server** (thread).

Startup takes about 4–5 s (the models load while llama-server starts). On Ctrl-C the remaining
audio is flushed through the VAD, the queue is drained, voiceprints are saved and llama-server is
stopped.

---

## Resource use and latency

| | VRAM | CPU |
|---|---|---|
| Whisper large-v3-turbo (int8_float16) | ~1.1 GB (incl. CUDA context) | – |
| ReazonSpeech, Silero VAD, CAM++ | – | short bursts on 6 / 1 / 2 threads |
| PLaMo-3-translate 2B Q4_K_M | ~1.9 GB | – |
| **Total** | **~3 GB of 6 GB** | |

With `--ja-engine kotoba`, add about 1 GB of VRAM.

**Delay** from the end of speech to the printed line is about **1 s**, measured by replaying
recordings at real-time speed:

| Step | Time |
|---|---|
| Pause detection | 0.5 s |
| Trailing padding | 0.3 s |
| Language ID + recognition | ~0.3 s |
| Translation | ~0.2 s |

The measured lag from segment end was 0.71 s median and 1.0 s worst with translation. Processing
takes about 0.5 s per sentence, far less than speaking takes, so a backlog never builds up. The
Google Doc trails the terminal by up to 3 s (batching).

---

## Benchmarks
Full numbers and notes: [bench/RESULTS.md](bench/RESULTS.md). Scripts are in `bench/`.

**Speech recognition** (`bench_asr.py`, 5 Japanese news clips of 6–14 s):

| Config | Median latency per clip |
|---|---|
| ReazonSpeech, CPU | **0.14 s** |
| kotoba-whisper, GPU | 0.30 s |
| Whisper turbo, GPU | 0.54 s |
| Whisper large-v3, GPU | 0.87 s |
| kotoba-whisper, CPU | 4.1 s |
| Whisper turbo, CPU | 9.2 s |

**Translation** (`bench_translate.py`, a 16-line meeting dialogue in ReazonSpeech style, with
history):

| Model | Meaning errors (of 16) | Median / line | VRAM |
|---|---|---|---|
| **PLaMo-3-translate 2B Q4_K_M** | **0** | **0.20 s** | **1.9 GB** |
| PLaMo-3-translate 8B Q3_K_M | 0 (dropped one 「私から」) | 0.47 s | 4.4 GB |
| Qwen3.5-4B Q4_K_M | 2, plus speaker names leaked into the output | 0.43 s | 3.1 GB |

**Speaker ID** (`bench_speaker.py`, 24 clips / 8 speakers):

| Model | Correct | Separation margin |
|---|---|---|
| **CAM++ zh/en** | **24/24** | **+0.15** |
| TitaNet-small | 24/24 | +0.07 |
| WeSpeaker ResNet34 | 22/24 | −0.23 |

In a live replay with 3 speakers and 11 turns, all 14 segments were clustered consistently.

**How much to trust these:** the test sets are small and clean. The translation test set and its
references were written, and the outputs judged, by the AI assistant that built this project.
Treat them as a first signal; real meeting recordings are the real test.

---

## Project layout
```
transcribe/
  live.py         main pipeline + CLI (threads, routing, outputs, commands)
  audio.py        parec capture, WAV replay, shared clock
  vad.py          Silero VAD segmentation with padded cuts
  asr.py          language routing, ReazonSpeech / Whisper recognition
  translate.py    llama-server process + PLaMo translator with history and glossary
  voices.py       CAM++ voiceprints, online clustering, rename/merge, voices file
  speakers.py     local HTTP endpoint for the Firefox add-on
  mdlog.py        Markdown transcript
  gdoc.py         Google Docs writer + login/test CLI
  record.py       raw audio recorder with level meter
  cuda_libs.py    preloads pip-installed CUDA libraries
extension/firefox/  Meet Speaker Bridge add-on (manifest, content script, background script)
bench/            benchmarks, test data, fake Docs API, RESULTS.md
docs/             google-docs-setup.md
scripts/          fetch_models.sh
models/           downloaded models (git-ignored)
tools/llama.cpp/  llama.cpp release build (git-ignored)
transcripts/      output transcripts (git-ignored)
recordings/       raw recordings (git-ignored)
```

---

## Privacy
- **Audio, text and voiceprints never leave the machine.** All models run locally; llama-server
  and the add-on endpoint listen only on 127.0.0.1.
- **The only outbound traffic** is the one-time model downloads, and Google Docs writes when you
  use `--doc` (the transcript text then lives in your Google Drive).
- **The add-on** sends only names and timestamps, and only to 127.0.0.1.
- **Voiceprints** (`--voices` files) are biometric data. They are stored only where you choose and
  are git-ignored by default (`voices*.json`).
- **Mute doesn't apply:** the app reads your mic directly, so muting yourself in Meet does **not**
  stop it from transcribing what you say.
- **Consent:** recording and transcribing people may require their consent depending on where
  they are. Tell participants.

---

## Known limitations
- **Split sentences:** a pause of more than `--min-silence` inside a sentence produces two lines
  that are translated separately (「ヤンバルクイナとの出会いは」 / 「十八歳の時だった」 → two
  fragments). A fix that holds incomplete-looking lines briefly for the continuation is planned.
  Meanwhile, `--min-silence 0.8` gives fewer splits at the cost of 0.3 s more delay.
- **Overlapping speech:** a segment with two people talking is attributed to the dominant voice.
- **Voice clustering:** similar voices may merge into one speaker; one person on a new mic may get
  a second `Speaker N` (fix with a rename/merge). Not yet validated on Meet's compressed audio.
- **Only Japanese is translated.** Other languages are written as spoken (PLaMo supports 53
  languages; enabling them is a small change).
- **The Meet add-on depends on Meet's undocumented page structure.** Captions are unreliable in
  some meetings, which is why names for others come from voice instead.
- **Not yet tested:** a long multi-person real call end to end, and English-speech routing on real
  meeting audio.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| No lines appear | `uv run python -m transcribe.record` and watch the level meters; pick devices with `--list`, `--mic`, `--system` |
| Others' words appear as yours | Use headphones (the mic is hearing the speakers) |
| `libcublas.so.12 … not found` | Run through `uv run` so the venv's CUDA libs are used; re-run `uv sync` |
| `llama-server exited` | See `llama-server.log`; check the model exists (`./scripts/fetch_models.sh`) and GPU memory is free (`nvidia-smi`) |
| Address already in use (8765/8090) | Another transcriber instance is running; stop it, or use `--speaker-port` |
| Your lines say "You" | Start with `--name`, type `me=Your Name`, or open Meet's People panel once with the add-on loaded |
| Same person shows as two speakers | `3=Tanaka` merges into Tanaka; or lower `--voice-threshold` slightly |
| Different people merged | Raise `--voice-threshold` (e.g. 0.6) |
| Google Doc errors | See [docs/google-docs-setup.md](docs/google-docs-setup.md); re-run `python -m transcribe.gdoc --login` |

---

## Licenses of models and tools
This project's code has no license file yet. Check each component's license before any
**commercial** use:

- **PLaMo-3-translate:** *PLaMo Community License*. Free for personal and non-commercial use;
  commercial use has separate terms. See the
  [model card](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF).
- **Whisper / faster-whisper, llama.cpp:** MIT.
- **ReazonSpeech-k2, kotoba-whisper, Silero VAD, 3D-Speaker CAM++, sherpa-onnx:** see each model
  card or repository.
- **TranscripTonic** (reference for the Meet caption selectors): MIT.
