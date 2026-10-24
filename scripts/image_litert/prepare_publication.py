#!/usr/bin/env python3
"""Prepare verified, explicitly selected LiteRT artifacts for public Hub publication.

Does not create repositories or upload. Stages hardlinks on the conversion disk;
source bundles and their manifest hashes are preserved. Excludes caches, source
checkpoints, credentials, compiler work directories and unrelated project files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

PROFILES = [
    ('SD_TURBO', 'SD-Turbo-LiteRT', 'sd-export', 'stabilityai/sd-turbo', 'b261bac6fd2cf515557d5d0707481eafa0485ec2', 512, 1),
    ('SDXL_TURBO', 'SDXL-Turbo-LiteRT', 'sdxl-export', 'stabilityai/sdxl-turbo', '71153311d3dbb46851df1931d3ca6e939de83304', 512, 4),
    ('FLUX_KLEIN_LITERT', 'FLUX.2-klein-4B-LiteRT', 'flux', 'litert-community/FLUX.2-klein-4B-LiteRT', '223f4c989d8e5275fa8e695ddebea8d7ac7bada6', 256, 4),
    ('BONSAI_LITERT', 'Bonsai-Image-ternary-4B-LiteRT', 'bonsai', 'litert-community/Bonsai-Image-ternary-4B', '317d577f5f962eb2d5a3980bca2ce047a8338212', 512, 4),
    ('IRIS_LITERT', 'iris-3b-Turbo-LiteRT', 'iris', 'litert-community/iris-3b-Turbo-LiteRT', '743fd20acefd7f9ddcf3585f88bcda3257f7213a', 768, 4),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target): return
        if source.stat().st_size == target.stat().st_size and digest(source) == digest(target): return
        raise ValueError(f'Refusing to overwrite staged file: {target}')
    os.link(source, target)


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # New text files are copies so editing README/notices never modifies an original hardlink.
    if target.exists() and os.path.samefile(source, target):
        return
    shutil.copyfile(source, target)


def verified_copy(root, relative, spec, stage):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError(relative)
    if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
        raise ValueError(f'Checksum/size mismatch: {path}')
    link(path, stage / ('SOURCE_MODEL_CARD.md' if relative == 'README.md' else relative))


def prepare(root, namespace, only_model=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(root); tools = root/'publication-tools'
    catalog = json.loads((root/'publication-catalog.json').read_text())
    original = json.loads((tools/'local-validation.json').read_text())
    plans = []
    for key, name, folder, source_repo, revision, size, steps in PROFILES:
        if only_model and key not in only_model: continue
        stage = root/'publications'/name; stage.mkdir(parents=True, exist_ok=True)
        source = root/folder
        print('PREPARE', namespace+'/'+name, flush=True)
        is_sd = key in ('SD_TURBO', 'SDXL_TURBO')
        if is_sd:
            manifest = json.loads((source/'manifest.json').read_text())
            if manifest['compiled'] or manifest['source']['revision'] != revision: raise ValueError('Unexpected SD source')
            for path, spec in manifest['files'].items(): verified_copy(source, path, spec, stage)
            copy(source/'manifest.json', stage/'manifest.json')
            for p in source.glob('tokenizer*/*'):
                if p.is_file() and not (stage/p.relative_to(source)).exists(): link(p, stage/p.relative_to(source))
            for filename in ['run_portable.py','java_noise.py','validate_plan.py']:
                copy(tools/filename, stage/filename)
            for filename in ['export_pipeline.py','graph_export.py','workflow.py','requirements.txt']:
                copy(tools/filename, stage/'conversion'/filename)
            copy(tools/'APACHE_LICENSE', stage/'CODE_LICENSE')
            notice = ('This Stability AI Model is licensed under the Stability AI Community License, '
                      'Copyright © Stability AI Ltd. All Rights Reserved\n\n'
                      f'Changes by {namespace}, 2026-10-24: converted the original Diffusers checkpoint to '
                      'float32 LiteRT graphs using LiteRT Torch; split components into fixed-shape '
                      'weight-bounded shards; exported scheduler/latent operations and tokenizer fixtures '
                      'in the word-image-litert-v1 execution plan. No additional training or weight '
                      'quantization. Companion inference/conversion code is under CODE_LICENSE.\n')
            (stage/'NOTICE').write_text(notice)
            validation = dict(original['pipelines'][key])
            validation = {k:v for k,v in validation.items() if k in ['export','fullCpuPlanReplay','outputDimensions','source','parity','noiseForDesktopParity','visualReview']}
            command = 'python run_portable.py --model-dir . --prompt "a red apple on a wooden table, studio lighting" --seed 42 --out output.png'
            license_meta = 'license: other\nlicense_name: stability-ai-community\nlicense_link: LICENSE.md'
            origin = f'New LiteRT conversion of [{source_repo}](https://huggingface.co/{source_repo}). **Powered by Stability AI.**'
            detail = ('Float32 graphs for CLIP text encoding, UNet denoising, the Euler ancestral sampler and VAE decoding. '
                      'Graphs are streamed one at a time. The scheduler and default step count are fixed by the manifest. '
                      'The bundle is portable and is not Google Tensor TPU AOT-compiled. Integer tensor I/O is preserved.\n\n'
                      'Weights retain the Stability AI Community License in LICENSE.md, including its use and commercial-use conditions. '
                      'They are not relicensed under Apache 2.0. The newly added companion scripts use CODE_LICENSE.')
            artifact_note = 'manifest.json enumerates the tensor execution plan and exact graph/tokenizer/constant hashes.'
            base = source_repo
        else:
            # Fetch only missing app-selected files and small public reference/license files.
            entries = [e for e in catalog if e['model'] == key]
            missing = [e['path'] for e in entries if not (source/e['path']).is_file()]
            if missing: snapshot_download(source_repo, revision=revision, allow_patterns=missing, local_dir=source)
            snapshot_download(source_repo, revision=revision,
                              allow_patterns=['README.md','LICENSE','LICENSE.md','NOTICE','NOTICE.md','android/**'], local_dir=source)
            for e in entries: verified_copy(source, e['path'], e, stage)
            # Include the matching upstream inference source and tokenizer metadata.
            for p in source.rglob('*'):
                relative = str(p.relative_to(source))
                if not p.is_file() or relative.startswith('.cache/'): continue
                if relative.startswith(('android/','python/','tokenizer/')) and not (stage/relative).exists(): link(p,stage/relative)
            copy(source/'README.md', stage/'SOURCE_MODEL_CARD.md')
            report_key = {'FLUX_KLEIN_LITERT':'FLUX_KLEIN','BONSAI_LITERT':'BONSAI','IRIS_LITERT':'IRIS'}[key]
            validation = {k:v for k,v in original['pipelines'][report_key].items() if k in ['fullCpuGeneration','outputDimensions','detail','visualReview']}
            license_meta = 'license: apache-2.0'
            origin = (f'Public mirror of the LiteRT conversion by [LiteRT Community](https://huggingface.co/{source_repo}), '
                      'with source attribution and byte-for-byte copies of the selected model graph artifacts. '
                      'The graph conversion was performed upstream, not by this repository publisher.')
            artifact_note = 'artifact_manifest.json records the exact source repository, immutable revision, file sizes and SHA-256 digests.'
            if key == 'FLUX_KLEIN_LITERT':
                base = 'black-forest-labs/FLUX.2-klein-4B'
                copy(hf_hub_download(base,'LICENSE.md',revision='e7b7dc27f91deacad38e78976d1f2b499d76a294'),stage/'LICENSE.md')
                for filename in ['java_noise.py','flux_host.json']: copy(tools/filename,stage/filename)
                copy(tools/'generate_flux.py',stage/'generate.py')
                copy(tools/'APACHE_LICENSE',stage/'CODE_LICENSE')
                hf_tok=root/'flux-source/tokenizer'
                for p in hf_tok.iterdir():
                    if p.is_file(): link(p,stage/'tokenizer/hf'/p.name)
                (stage/'NOTICE').write_text('FLUX.2 [klein] 4B, Copyright 2026 Black Forest Labs.\nQwen3-4B text encoder and tokenizer, Alibaba Cloud, Apache 2.0.\nLiteRT graphs and tokenizer tables are mirrored from '+source_repo+' at '+revision+'.\nChanges by '+namespace+': added a reconstructed 256px CPU host loop, fixed timestep/VAE normalization constants, runnable example, checksums and desktop validation. Graph weights are unchanged.\n')
                command = 'python generate.py --model-dir . --prompt "a red apple on a wooden table, studio lighting" --seed 42 --out output.png'
                detail = '256px int8 Qwen3 encoder, diffusion transformer and VAE shards. Includes the reconstructed host constants and CPU sampler used for the desktop smoke check. The runtime streams one neural graph at a time. Image-editing graphs are not included.'
            elif key == 'BONSAI_LITERT':
                base = 'prism-ml/bonsai-image-ternary-4B-unpacked'
                copy(source/'NOTICE.md',stage/'NOTICE.md')
                copy(hf_hub_download('black-forest-labs/FLUX.2-klein-4B','LICENSE.md',revision='e7b7dc27f91deacad38e78976d1f2b499d76a294'),stage/'LICENSE.md')
                copy(source/'generate.py',stage/'generate.py')
                detail = '512px int4 ternary-weight DiT, int4 Qwen3 text encoder and fp32 VAE decoder. Includes both the CPU-shaped and GPU-shaped DiT; the CPU example uses dit_int4b32.tflite. Created using Bonsai Image by Prism ML. FLUX.2 architecture by Black Forest Labs; text encoder/tokenizer by Alibaba Cloud.'
                command = 'python generate.py --model-dir . --prompt "a red apple on a wooden table, studio lighting" --seed 42 --threads 8 --out output.png'
            else:
                base = 'speridlabs/iris-3b'
                copy(source/'LICENSE',stage/'LICENSE'); copy(source/'NOTICE',stage/'NOTICE')
                detail = '768px pixel-space Turbo diffusion with Qwen3-VL-4B conditioning. Twelve text-encoder graphs and eight diffusion graphs, streamed one at a time, with four sampling steps and no VAE. Upstream Android and Python reference engines are included.'
                command = 'python python/iris3b_litert.py --model-dir model --prompt "a red apple on a wooden table, studio lighting" --seed 42 --threads 8 --out output.png'
        requirements = 'ai-edge-litert==2.3.0\nnumpy>=2.0\npillow>=11\n'
        if key != 'IRIS_LITERT': requirements += 'transformers==5.20.2\njinja2\nftfy==6.3.1\n'
        (stage/'requirements.txt').write_text(requirements)
        validation.update(date='2026-10-24',backend='desktop LiteRT CPU',androidDeviceValidated=False,androidGpuValidated=False,tensorTpuValidated=False)
        (stage/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        image_folder={'SD_TURBO':'sd','SDXL_TURBO':'sdxl','FLUX_KLEIN_LITERT':'flux','BONSAI_LITERT':'bonsai','IRIS_LITERT':'iris'}[key]
        copy(root/(image_folder+'-apple.png'),stage/'example.png')
        card=f'''---
{license_meta}
base_model: {base}
pipeline_tag: text-to-image
library_name: litert
tags: [litert, text-to-image, android, cpu, experimental]
---
# {name}

{origin}

Source: [{source_repo}](https://huggingface.co/{source_repo}/tree/{revision}), revision `{revision}`.
Output: **{size} × {size}**, **{steps} sampling step(s)**.

![Desktop LiteRT CPU sample](example.png)

## Contents

{detail}

{artifact_note} Large source Diffusers checkpoints, download caches, credentials and compiler working files are not included.

## Run on CPU

Use Python 3.12 in a virtual environment. Download the repository with `hf download {namespace}/{name} --local-dir {name}`, then:

```sh
cd {name}
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
{command}
```

These are diffusion graph pipelines, not language-model `.litertlm` checkpoints. A standalone graph call with random inputs does not generate a meaningful image; use the full pipeline. Android integrations must run off the main thread, release tensor buffers/models between stages and use a shared LiteRT Environment. Source/model graph layout is preserved for reuse by the Research Lab Android application.

## Validation and limitations

The actual model graphs completed a full **desktop LiteRT 2.3.0 CPU** generation check on Linux x86_64. See `validation.json`. The sample was visually checked as a recognizable apple. This is a smoke check, not a comparative quality or speed benchmark. SD/SDXL source-parity checks use export-time PyTorch noise; normal inference uses the runner's documented seeded stream. Different runtimes can generate different samples with the same integer seed.

**Android CPU/GPU execution, phone memory requirements and Google Tensor TPU execution have not been validated for this repository.** Successful desktop execution does not establish phone performance. Graph loads, compilation and host tensor copies can be expensive. Use the upstream model cards for intended use, limitations and applicable restrictions.

## Attribution and license

The original model and conversion authors retain their copyrights. This distribution retains the source license and notices; conversion/mirroring does not change model license terms. See the included LICENSE/NOTICE files and `SOURCE_MODEL_CARD.md`. Added helper code is identified separately where applicable.
'''
        (stage/'README.md').write_text(card)
        files = {str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in stage.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','artifact_manifest.json']}
        artifact=dict(format='litert-publication-artifacts-v1',repository=namespace+'/'+name,sourceRepository=source_repo,sourceRevision=revision,graphConversionPerformedByPublisher=bool(is_sd),files=files)
        (stage/'artifact_manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
        files['artifact_manifest.json']=dict(bytes=(stage/'artifact_manifest.json').stat().st_size,sha256=digest(stage/'artifact_manifest.json'))
        (stage/'SHA256SUMS').write_text(''.join(f'{s["sha256"]}  {path}\n' for path,s in sorted(files.items())))
        plans.append(dict(repository=namespace+'/'+name,directory=str(stage),fileCount=len(files)+1,bytes=sum(s['bytes'] for s in files.values()),files=files))
        print('READY',namespace+'/'+name,len(files)+1,'files',sum(s['bytes'] for s in files.values()),flush=True)
    (root/'publication-plan.json').write_text(json.dumps(plans,indent=2)+'\n')
    print('Prepared selected repositories. No upload performed.',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--only-model', action='append', choices=[p[0] for p in PROFILES])
    a=p.parse_args();prepare(a.root,a.namespace,a.only_model)
