#!/usr/bin/env python3
"""Stage a compiled image bundle without uninstalling the app or touching GGUF weights."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).parent / 'image_litert'))
from workflow import digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('bundle', type=Path)
p.add_argument('--serial', required=True, help='Explicit adb serial prevents accidental staging to the wrong phone')
a = p.parse_args()
m = json.loads((a.bundle / 'manifest.json').read_text())
if m['format'] != 'word-image-litert-v1' or not m['compiled']:
    sys.exit('Requires a compiled app-compatible image bundle')
if m['modelId'] not in ('SD_TURBO', 'SDXL_TURBO', 'FLUX_KLEIN') or m['targetSoc'] not in ('Tensor_G5', 'Tensor_G6'):
    sys.exit('Invalid model/SoC')
root = a.bundle.resolve()
files = []
for relative, spec in m['files'].items():
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative):
        sys.exit("Bundle filenames must be shell-safe ASCII paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
        sys.exit(f'Unsafe path or hash mismatch: {relative}')
    files.append((relative, path))
adb = ['adb', '-s', a.serial]
remote = f'/sdcard/Android/data/dev.gemmabench/files/image-litert/{m["modelId"]}/{m["targetSoc"]}'
subprocess.run(adb + ['shell', 'mkdir', '-p', remote], check=True)
# Invalidate prior marker/manifest first. Publish the new manifest last, after all files.
subprocess.run(adb + ['shell', 'rm', '-f', remote + '/.verified', remote + '/manifest.json'], check=True)
for relative, path in files:
    destination = remote + '/' + relative
    subprocess.run(adb + ['shell', 'mkdir', '-p', str(Path(destination).parent)], check=True)
    subprocess.run(adb + ['push', str(path), destination], check=True)
subprocess.run(adb + ['push', str(root / 'manifest.json'), remote + '/manifest.json'], check=True)
print('Staged. Open Research Lab → Images → Verify staged TPU bundles, then select TPU in the benchmark.')
