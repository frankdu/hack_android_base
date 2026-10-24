#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 MODEL_FILE [MODEL_FILE ...]" >&2
  echo "Stage files named exactly as in ImageModels.kt, then run ImageGenerationInstrumentedTest with -e imageInference true." >&2
  exit 2
fi
adb shell mkdir -p /sdcard/Android/data/dev.gemmabench/files/image-models
adb push "$@" /sdcard/Android/data/dev.gemmabench/files/image-models/
