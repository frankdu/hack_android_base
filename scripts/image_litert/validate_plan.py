#!/usr/bin/env python3
"""Replay an exported portable plan on CPU and compare its PNG to the export reference.

For parity, use export-time PyTorch noise (seed 42) and the exact first tokenizer
fixture. Android's Java noise is a separate, reproducible random stream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from PIL import Image


def validate(bundle, out):
    root = Path(bundle); m = json.loads((root/'manifest.json').read_text())
    assert m['format'] == 'word-image-litert-v1' and not m['compiled']
    for name, spec in m['files'].items():
        path = (root/name).resolve(); assert path.is_relative_to(root.resolve())
        assert path.stat().st_size == spec['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'], name
    rng = torch.Generator().manual_seed(42)
    values = {}
    for e in m['externals']:
        shape, dtype = e['shape'], np.dtype(e['dtype'])
        if e['kind'] == 'normal': a = torch.randn(shape, generator=rng).numpy()
        elif e['kind'] == 'constant': a = np.fromfile(root/e['path'],dtype=dtype).reshape(shape)
        else:
            fixture = json.loads((root/e['tokenizer']).read_text())['fixtures'][0]
            a = np.array(fixture['ids' if e['kind']=='tokens' else 'mask'],dtype=dtype).reshape(shape)
        values[e['tensor']] = a
    uses = {}
    for action in m['actions']:
        for i in action['inputs']: uses[i['tensor']] = uses.get(i['tensor'], 0) + 1
    uses[m['output']] = uses.get(m['output'], 0) + 1
    start = time.monotonic()
    for action in m['actions']:
        model = Interpreter(model_path=str(root/action['model']), num_threads=8)
        model.allocate_tensors()
        result = model.get_signature_runner(action['signature'])(**{i['name']: values[i['tensor']] for i in action['inputs']})
        for o in action['outputs']:
            a = result[o['name']].copy()
            assert list(a.shape) == o['shape'] and np.isfinite(a).all(), (action['model'],o['name'])
            values[o['tensor']] = a
        for i in action['inputs']:
            uses[i['tensor']] -= 1
            if uses[i['tensor']] == 0: del values[i['tensor']]
        print(action['model'],round(time.monotonic()-start,1),flush=True)
        del result, model
    image = np.clip(values[m['output']][0].transpose(1,2,0)*255,0,255).round().astype(np.uint8)
    reference = np.array(Image.open(root/'reference.png')).astype(np.float64)
    mse = np.mean((image.astype(np.float64) - reference)**2)
    psnr = float(10*np.log10(255**2/max(mse,1e-12)))
    assert psnr > 40, psnr
    Image.fromarray(image).save(out)
    report = dict(modelId=m['modelId'], backend='desktop LiteRT CPU', psnrVsExportReference=psnr,
                  elapsedSeconds=time.monotonic()-start, actions=len(m['actions']), deviceValidated=False)
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('bundle'); p.add_argument('--out',required=True)
    args=p.parse_args(); validate(args.bundle,args.out)
