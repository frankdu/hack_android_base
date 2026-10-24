#!/usr/bin/env python3
"""Generate an image from a portable word-image-litert-v1 bundle on LiteRT CPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from transformers import AutoTokenizer
from java_noise import java_noise


def safe_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe artifact path: {name}')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def generate(root, prompt, seed, threads, output, verify=True):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['format'] != 'word-image-litert-v1' or manifest['compiled']:
        raise ValueError('Expected a portable, non-AOT word-image-litert-v1 bundle')
    if verify:
        for name, spec in manifest['files'].items():
            path = safe_file(root, name)
            if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
                raise ValueError(f'Artifact integrity failure: {name}')
    values, tokenizers, encoded = {}, {}, {}
    # One Java Gaussian stream across all external noise inputs, in manifest order.
    count = sum(int(np.prod(e['shape'])) for e in manifest['externals'] if e['kind'] == 'normal')
    noise = java_noise(seed, count); offset = 0
    for e in manifest['externals']:
        shape, dtype = e['shape'], np.dtype(e['dtype'])
        if e['kind'] == 'normal':
            n = int(np.prod(shape)); a = noise[offset:offset + n].reshape(shape); offset += n
        elif e['kind'] == 'constant':
            a = np.fromfile(safe_file(root, e['path']), dtype=dtype).reshape(shape)
        else:
            name = e['tokenizer']
            if name not in encoded:
                spec = json.loads(safe_file(root, name).read_text())
                tok = tokenizers.setdefault(name, AutoTokenizer.from_pretrained(str(safe_file(root, name).parent), local_files_only=True))
                def encode(text):
                    return tok(spec['prefix'] + text + spec['suffix'], max_length=spec['length'],
                               padding='max_length', truncation=True, return_tensors='np')
                for fixture in spec['fixtures']:
                    result = encode(fixture['prompt'])
                    if result['input_ids'][0].tolist() != fixture['ids'] or result['attention_mask'][0].tolist() != fixture['mask']:
                        raise ValueError(f'Tokenizer parity failed: {name}')
                encoded[name] = encode(prompt)
            a = np.asarray(encoded[name]['input_ids' if e['kind'] == 'tokens' else 'attention_mask'], dtype=dtype).reshape(shape)
        values[e['tensor']] = a
    uses = {}
    for action in manifest['actions']:
        for i in action['inputs']: uses[i['tensor']] = uses.get(i['tensor'], 0) + 1
    uses[manifest['output']] = uses.get(manifest['output'], 0) + 1
    start = time.monotonic()
    for action in manifest['actions']:
        # The 2.3.0 Python TensorBuffer wrapper cannot write int64. The public
        # CPU runner uses LiteRT Interpreter signatures; Android uses writeLong.
        model = Interpreter(model_path=str(safe_file(root, action['model'])), num_threads=threads)
        model.allocate_tensors()
        result = model.get_signature_runner(action['signature'])(
            **{i['name']: np.ascontiguousarray(values[i['tensor']]) for i in action['inputs']})
        for o in action['outputs']:
            a = result[o['name']].copy()
            if list(a.shape) != o['shape'] or not np.isfinite(a).all():
                raise ValueError(f'Invalid output from {action["model"]}')
            values[o['tensor']] = a
        del result, model
        gc.collect()
        for i in action['inputs']:
            uses[i['tensor']] -= 1
            if uses[i['tensor']] == 0: del values[i['tensor']]
        print(f'{action["phase"]} step {action["step"]}: {action["model"]}', flush=True)
    pixels = np.clip(values[manifest['output']][0].transpose(1, 2, 0) * 255 + .5, 0, 255).astype(np.uint8)
    if pixels.std() == 0: raise ValueError('Uniform output image')
    Image.fromarray(pixels).save(output)
    print(f'Saved {output} in {time.monotonic() - start:.1f}s (desktop CPU; includes graph loads)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', default=str(Path(__file__).parent))
    p.add_argument('--prompt', default='a red apple on a wooden table, studio lighting')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--threads', type=int, default=8)
    p.add_argument('--out', default='output.png'); p.add_argument('--skip-verify', action='store_true')
    a = p.parse_args(); generate(a.model_dir, a.prompt, a.seed, a.threads, a.out, not a.skip_verify)
