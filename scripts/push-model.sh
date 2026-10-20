#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 {e2b|e4b|e2b-g5|e2b-g6} /path/to/model.litertlm" >&2
    exit 2
fi
case "$1" in
    e2b|e4b|e2b-g5|e2b-g6) slot="$1" ;;
    *) echo "Unknown model slot: $1" >&2; exit 2 ;;
esac
if [ ! -f "$2" ]; then
    echo "Model file does not exist: $2" >&2
    exit 2
fi

destination="/sdcard/Android/data/dev.gemmabench/files/models"
adb shell mkdir -p "$destination"
adb push "$2" "$destination/$slot.litertlm"
echo "Staged $slot. Open Models and tap 'Import files staged by ADB'."
echo "The app makes a private copy. You can remove this staging file after importing."
