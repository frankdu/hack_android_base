#!/usr/bin/env python3
"""Compare whole quantized pipelines with their FP32 LiteRT source on fixed prompts."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

PROMPTS = [
    ('apple', 'a red apple on a wooden table, studio lighting', 42),
    ('teapot', 'A blue ceramic teapot on a wooden table, morning light.', 7),
    ('cat', 'A friendly orange cat sitting in a sunny garden.', 123),
]


def single(args):
    sys.path.insert(0, str(Path(args.bundle).resolve()))
    from run_portable import generate
    start = time.monotonic()
    generate(args.bundle, args.prompt, args.seed, 8, args.image)
    Path(args.metrics).write_text(json.dumps(dict(elapsedSeconds=time.monotonic()-start,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, backend='desktop LiteRT CPU',
        deviceValidated=False), indent=2)+'\n')


def compare(args):
    root = Path(args.bundle).resolve(); source = Path(args.source).resolve()
    out = root / 'validation'; out.mkdir(exist_ok=True)
    records = []; grid = Image.new('RGB', (1024, len(PROMPTS)*546), 'white')
    draw = ImageDraw.Draw(grid)
    for index, (name, prompt, seed) in enumerate(PROMPTS):
        images = []
        timings = {}
        for label, folder in [('fp32', source), ('quantized', root)]:
            image = out / f'{name}-{label}.png'; metrics = image.with_suffix('.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--single', '--bundle', str(folder),
                '--prompt', prompt, '--seed', str(seed), '--image', str(image), '--metrics', str(metrics)]
            with image.with_suffix('.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            images.append(np.asarray(Image.open(image).convert('RGB'), dtype=np.float64))
            timings[label] = json.loads(metrics.read_text())
            x = 0 if label == 'fp32' else 512
            grid.paste(Image.open(image), (x,index*546+34))
            draw.text((x+8,index*546+8), f'{name} / {label} / seed {seed}', fill='black')
        a,b=images; mse=float(np.mean((a-b)**2))
        record = dict(name=name,prompt=prompt,seed=seed,psnrDb=float(10*np.log10(255**2/max(mse,1e-12))),
            correlation=float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1]),
            quantizedPixelStd=float(b.std()),timings=timings)
        assert np.isfinite(b).all() and record['quantizedPixelStd'] > 5
        records.append(record)
        print(json.dumps(record),flush=True)
        (root / 'validation.json').write_text(json.dumps(dict(status='generated-awaiting-visual-review',
            backend='desktop LiteRT CPU',deviceValidated=False,prompts=records),indent=2)+'\n')
    grid.save(out / 'comparison.png')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True); p.add_argument('--source')
    p.add_argument('--single',action='store_true'); p.add_argument('--prompt'); p.add_argument('--seed',type=int)
    p.add_argument('--image'); p.add_argument('--metrics')
    a=p.parse_args(); single(a) if a.single else compare(a)
