#!/usr/bin/env python3
"""Reconstruct the missing constants of the pinned community FLUX 256px export.

Reads the original checkpoint (not GGUF) without loading the DiT into memory.
Writes a tiny JSON pack for the Android host loop; no prompt or random noise is baked in.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from diffusers.models.embeddings import get_timestep_embedding

SOURCE_REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
COMMUNITY_REVISION = '223f4c989d8e5275fa8e695ddebea8d7ac7bada6'


def prepare(source, output):
    source, output = Path(source), Path(output)
    with safe_open(source / 'transformer/diffusion_pytorch_model.safetensors', framework='pt') as f:
        prefix = 'time_guidance_embed.timestep_embedder.'
        # Fail on an upstream rename; the pinned checkpoint's ABI is intentional.
        weights = {key: f.get_tensor(prefix + key).float() for key in ('linear_1.weight', 'linear_2.weight')}
    with safe_open(source / 'vae/diffusion_pytorch_model.safetensors', framework='pt') as f:
        mean, var = f.get_tensor('bn.running_mean').float(), f.get_tensor('bn.running_var').float()
    vae = json.loads((source / 'vae/config.json').read_text())
    tokens, steps = 256, 4
    m200, m10 = .00016927 * tokens + .45666666, 8.73809524e-05 * tokens + 1.89833333
    a = (m200 - m10) / 190
    mu = a * steps + m200 - 200 * a
    lin = np.linspace(1, 1 / steps, steps)
    sigmas = np.append(np.exp(mu) / (np.exp(mu) + 1 / lin - 1), 0).astype(np.float32)
    t = get_timestep_embedding(torch.from_numpy(sigmas[:-1] * 1000), 256,
                              flip_sin_to_cos=True, downscale_freq_shift=0)
    t = torch.nn.functional.linear(t, weights['linear_1.weight'])
    t = torch.nn.functional.linear(torch.nn.functional.silu(t), weights['linear_2.weight'])
    document = dict(format='community-flux-host-v1', sourceRevision=SOURCE_REVISION,
                    communityRevision=COMMUNITY_REVISION, width=256, steps=steps,
                    sigmas=sigmas.tolist(), temb=t.detach().numpy().tolist(),
                    bnScale=torch.sqrt(var + vae['batch_norm_eps']).tolist(), bnShift=mean.tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, separators=(',', ':')) + '\n')
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); prepare(args.source, args.output)
