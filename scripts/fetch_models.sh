#!/usr/bin/env bash
# Downloads everything the transcriber needs that isn't a Python package (~6 GB):
#   - ReazonSpeech-k2 (Japanese ASR, sherpa-onnx)       models/reazonspeech-k2/
#   - Silero VAD (speech detection, sherpa-onnx build)   models/vad/
#   - CAM++ speaker-embedding model                      models/speaker/
#   - PLaMo-3-translate 2B GGUF (translation)            models/llm/
#   - llama.cpp b12521 CUDA 12.8 build (llama-server)    tools/llama.cpp/
# Whisper models (large-v3-turbo, optional kotoba) download from Hugging Face on first use.
# Already-downloaded files are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # fetch <url> <dest>
  if [ -s "$2" ]; then echo "have  $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "fetch $2"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# Japanese ASR + its test clips
NAME=sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01
if [ ! -d "models/reazonspeech-k2/$NAME" ]; then
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$NAME.tar.bz2" \
        "models/reazonspeech-k2/$NAME.tar.bz2"
  tar -xjf "models/reazonspeech-k2/$NAME.tar.bz2" -C models/reazonspeech-k2
  rm "models/reazonspeech-k2/$NAME.tar.bz2"
  mkdir -p samples
  cp "models/reazonspeech-k2/$NAME"/test_wavs/*.wav samples/ 2>/dev/null || true
fi

# Speech detection
fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
      models/vad/silero_vad.onnx

# Speaker identification
SPK=3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$SPK" \
      "models/speaker/$SPK"

# Translation model
fetch "https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF/resolve/main/plamo-3-translate-2b-Q4_K_M.gguf" \
      models/llm/plamo-3-translate-2b-Q4_K_M.gguf

# llama.cpp server (uses the CUDA libraries from the Python venv; see README)
if [ ! -x tools/llama.cpp/llama-b12521/llama-server ]; then
  fetch "https://github.com/ggml-org/llama.cpp/releases/download/b12521/llama-b12521-bin-ubuntu-cuda-12.8-x64.tar.gz" \
        tools/llama.cpp/llama.tar.gz
  tar -xzf tools/llama.cpp/llama.tar.gz -C tools/llama.cpp
  rm tools/llama.cpp/llama.tar.gz
fi

echo "All models and tools are in place."
