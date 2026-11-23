# ASR benchmark results

## 2026-11-18 — speed, 5 Japanese news clips (6–14 s), Ryzen 7 7435HS / RTX 4050 6GB, no Meet running

| config | median latency | worst | median RTF | lang-ID cost |
|---|---|---|---|---|
| reazon-cpu (sherpa-onnx, int8, 8 threads) | 0.14s | 0.19s | 0.01 | n/a (Japanese only) |
| kotoba-cpu (int8) | 4.12s | 4.28s | 0.41 | 3.95s |
| turbo-cpu (int8) | 9.20s | 9.47s | 0.92 | 4.55s |
| kotoba-gpu (int8_float16) | 0.30s | 0.35s | 0.03 | 0.23s |
| turbo-gpu (int8_float16) | 0.54s | 0.62s | 0.06 | 0.23s |
| large-v3-gpu (int8_float16) | 0.87s | 1.29s | 0.11 | 0.26s |

Findings:
- Whisper on CPU costs ~4 s (kotoba) / ~9 s (turbo) per utterance regardless of length
  (fixed 30 s encoder window) -> not viable for real time.
- ReazonSpeech on CPU is effectively free. Same words as kotoba on these clips, but no
  punctuation, kanji numerals (十八歳), and one clip came out in all hiragana.
- Whisper language ID (turbo and large-v3) labels 4/5 Japanese clips as Korean with ~0.99
  confidence, even though the transcribed text is correct Japanese. Fix: restrict LID to the
  candidate set {en, ja} instead of taking the global argmax. Kotoba detected ja correctly.
- faster-whisper 1.2.1 `decode_audio` is broken with the installed PyAV (`metadata_errors`
  kwarg); we load audio with soundfile instead.

Caveat: clean read news speech, not meeting audio. Accuracy (CER) must be re-measured on
real meeting recordings; no English clips yet to test en/ja routing.

## 2026-11-18 — live pipeline (VAD -> LID -> ASR), test session of the 5 clips with pauses

- Default setup (turbo on GPU for LID/English + ReazonSpeech on CPU): ~0.3 s ASR per segment,
  ~0.9 s from end of speech to printed line, worst 1.3 s (first call).
  Correction: live.py passes --min-silence (default 0.5 s) to the VAD, so these runs used a
  0.5 s pause, not the 0.8 s Segmenter default.
- VRAM: turbo + ReazonSpeech = ~1.1 GB; with kotoba instead of ReazonSpeech = ~2.1 GB.
- Using Silero's own segment samples clipped word onsets/endings (lost 「早く」, 「ヤンバル」, 「す」).
  Fixed by cutting from our own buffer with 0.5 s / 0.3 s padding: all words correct.
- Sentences still split at clause pauses (>0.5 s); to be merged by the translation step.
- 16 s of noisy mic audio (~-28 dB floor) produced no segments with threshold 0.4.

## 2026-11-18 — translation (bench/bench_translate.py, 16-line meeting dialogue, with history)

| model | meaning errors /16 | median latency | worst | VRAM (LLM only) |
|---|---|---|---|---|
| PLaMo-3-translate 2B Q4_K_M | 0 | 0.20s | 0.34s | 1.9 GB |
| PLaMo-3-translate 8B Q3_K_M | 0 (dropped 「私から」 once) | 0.47s | 0.82s | 4.4 GB |
| Qwen3.5-4B Q4_K_M | 2 (+2 speaker-name leaks) | 0.43s | 0.63s | 3.1 GB |

-> PLaMo 2B chosen. Total VRAM with speech models ~3 GB.
Caveat: small self-written test set, judged by Claude; re-check on real meetings.

Live pipeline with translation (test session): ~0.7 s median from segment end to printed
translation (~1 s after the last word). Known weakness: a sentence split across two VAD
segments is translated as two fragments (「ヤンバルクイナとの出会いは」/「十八歳の時だった」 ->
"My encounter with the Yanbaru rail" / "I was eighteen years old.").

## 2026-11-18 — speaker ID by voice (bench/bench_speaker.py; 24 clips, 8 speakers, clean audio)

| model | nearest-neighbour ID | gap (min same - max diff) | speed |
|---|---|---|---|
| 3D-Speaker CAM++ zh/en advanced | 24/24 | +0.15 (thr 0.60) | 8 ms per s audio |
| NeMo TitaNet-small | 24/24 | +0.07 | 8 ms |
| WeSpeaker ResNet34-LM | 22/24 | -0.23 | 19 ms |

-> CAM++ chosen; live threshold 0.55 (margin for compressed Meet audio).
Live test, 3 speakers / 11 turns: captions for the first 4 turns only -> 14/14 segments
named correctly, incl. all 7 after captions stopped. No captions at all -> 3 clusters,
14/14 consistent ("Speaker 1/2/3"). Not yet tested: Meet/Opus audio, overlapping speech.
