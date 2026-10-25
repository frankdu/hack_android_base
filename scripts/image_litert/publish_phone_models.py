#!/usr/bin/env python3
"""Prepare and publish validated compact SD/SDXL LiteRT packages with exact hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def prepare(root, name):
    folder=root/(name+'-Mobile-INT8-LiteRT'); repo='frankdu/'+folder.name
    report=json.loads((folder/'validation.json').read_text())
    assert len(report['prompts'])==3 and report['visualReview']['passed']
    quant=json.loads((folder/'quantization.json').read_text())
    android=report.get('androidValidation',{})
    if android:
        passed=android.get('cpu',{}).get('status')=='PASS'
        report.update(deviceValidated=passed,backend='desktop LiteRT CPU and Android LiteRT CPU',
            androidGpuValidated=False,tensorTpuValidated=False)
        quant.update(deviceValidated=passed,deviceValidationScope='CPU on Pixel 10 Pro Fold / Tensor G5 only',
            androidGpuValidated=False,tensorTpuValidated=False)
        (folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        (folder/'quantization.json').write_text(json.dumps(quant,indent=2)+'\n')
    manifest=json.loads((folder/'manifest.json').read_text())
    assert not manifest['compiled']
    manifest['validation']='desktop-cpu-three-prompts-visually-reviewed; see validation.json for device results'
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    conversion=folder/'conversion'; conversion.mkdir(exist_ok=True)
    for script in ('quantize_phone.py','litert_quantize_conv.py','validate_phone_quantization.py','audit_phone_weights.py','tensor_aliases.py','test_tensor_aliases.py','publish_phone_models.py'):
        shutil.copy2(root/script,conversion/script)
    shutil.copy2(folder/'validation/teapot-quantized.png',folder/'example.png')
    # Only samples and compact metrics belong in the public package, not verbose logs.
    for log in (folder/'validation').glob('*.log'): log.unlink()
    source='stabilityai/'+name.lower()
    device=report.get('androidValidation',{})
    device_summary=device.get('summary','See validation.json for the exact Android validation status.')
    readme=f'''---
license: other
license_name: stabilityai-community
license_link: https://huggingface.co/{source}/blob/main/LICENSE.md
base_model: {source}
base_model_relation: quantized
library_name: litert
pipeline_tag: text-to-image
tags:
  - litert
  - int8
  - android
  - on-device
  - diffusion
---
# {name} Mobile INT8 LiteRT

Powered by Stability AI. A compact, sharded 512×512 text-to-image pipeline,
converted from the pinned FP32 LiteRT export with **LiteRT CLI**. The architecture
and sampler are unchanged. Neural encoder/UNet weights use dynamic channelwise
INT8; VAE weights use FP16 with float activations. Sampler graphs remain FP32.

![Quantized CPU example](example.png)

Runtime artifacts: **{quant['runtimeBytes']/1e9:.2f} GB**, down from
{quant['originalRuntimeBytes']/1e9:.2f} GB. Largest graph: {quant['largestGraphBytes']/1e6:.1f} MB.
One neural graph is loaded at a time. Download size is not peak RAM: activations,
dequantization, driver caches and the rest of Android also consume memory.

## Intended phones and limits

The design targets the 12–16 GB RAM class of Pixel phones. Pixel 10 Pro uses
Tensor G5 and 16 GB RAM; Pixel 11 Pro uses Tensor G6 and 12 or 16 GB depending on
storage configuration. See [Pixel 10 specs](https://store.google.com/product/pixel_10_pro_specs)
and [Pixel 11 specs](https://store.google.com/product/pixel_11_pro_specs).
This is a portable **CPU pipeline**, not a Tensor AOT/TPU compilation. Current
GPU layouts fail LiteRT 2.3 compilation on the measured Tensor G5 and require
a separate compatible GPU export; compressed weights alone do not fix that.
Pixel 10 Pro and Pixel 11 Pro themselves have not been measured. Exact Android
device/backend results, including failures, are in `validation.json`.

{device_summary}

## Validation

Three complete desktop LiteRT CPU generations (apple, blue teapot, orange cat)
were compared with the FP32 source using identical Java Gaussian noise and seeds.
All outputs are finite, nonuniform and visually recognizable. Quantization can
change composition and detail; these outputs are not pixel-identical to FP32.
Per-prompt PSNR, pixel correlation, process peak RSS and timings are published.
Desktop process RSS includes Python/tokenizer overhead and is not a phone claim.

![FP32 left, quantized right, three prompts](validation/comparison.png)

## Run on CPU

```sh
hf download {repo} --local-dir {folder.name}
cd {folder.name}
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_portable.py --prompt "A blue ceramic teapot on a wooden table, morning light." --seed 7 --out teapot.png
```

Android uses the `word-image-litert-v1` manifest, LiteRT 2.3.0 CompiledModel,
TensorBuffer int64 I/O, and one graph at a time. **Reuse the input TensorBuffer for
every output carrying `aliasOfInput` in the manifest**, and close shared buffers
only once. A passthrough output can be the same physical TFLite tensor as its input;
allocating it separately overwrites valid token inputs with uninitialized memory.
The Android app includes this fix. No automatic CPU fallback is advertised as GPU execution.

## Reproduce and audit

`quantization.json` records versions, per-graph sizes, recipes and source manifest
hash. `conversion/quantize_phone.py` invokes LiteRT CLI. Its guarded
`litert_quantize_conv.py` entry point fixes an ai-edge-quantizer 0.10.0 bug in the
>32 MiB convolution chunking path: broadcast scales in the original 4D shape,
then reshape alongside weights. Installed packages are unchanged. The fix was
checked bit-for-bit against unchunked per-channel INT8 arithmetic on a 36 MiB
convolution tensor. No unsupported recipe name is silently substituted.

`manifest.json` pins runtime files; `artifact_manifest.json` and `SHA256SUMS` cover
the complete publication. Original FP32 packages remain available separately.
Source model card, LICENSE.md and NOTICE are retained. Model weights retain the
Stability AI Community License and source usage terms; this publication does not
relicense them. Companion code retains its original CODE_LICENSE.
'''
    (folder/'README.md').write_text(readme)
    files={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(folder.rglob('*')) if p.is_file() and not set(p.relative_to(folder).parts)&{'.cache','__pycache__','.venv'}
        and p.name not in ('artifact_manifest.json','SHA256SUMS')}
    (folder/'SHA256SUMS').write_text(''.join(f'{s["sha256"]}  {n}\n' for n,s in files.items()))
    files['SHA256SUMS']=dict(bytes=(folder/'SHA256SUMS').stat().st_size,sha256=sha(folder/'SHA256SUMS'))
    artifact=dict(repository=repo,files=files)
    (folder/'artifact_manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    files['artifact_manifest.json']=dict(bytes=(folder/'artifact_manifest.json').stat().st_size,sha256=sha(folder/'artifact_manifest.json'))
    return folder,repo,files


def publish(root, upload):
    api=HfApi(); assert api.whoami()['name']=='frankdu'
    results=[]
    for name in ('SD-Turbo','SDXL-Turbo'):
        folder,repo,files=prepare(root,name)
        print(repo,len(files),'files',sum(s['bytes'] for s in files.values()),flush=True)
        if not upload: continue
        api.create_repo(repo,private=False,exist_ok=True)
        assert not api.model_info(repo).private
        commit=api.upload_folder(repo_id=repo,folder_path=str(folder),allow_patterns=list(files),
            commit_message='Publish compact INT8 LiteRT image pipeline with quality and memory validation')
        info=api.model_info(repo,revision=commit.oid,files_metadata=True,token=False)
        assert not info.private and info.gated is False
        remote={f.rfilename:f for f in info.siblings}
        for n,s in files.items():
            f=remote[n]; assert f.size==s['bytes'],(repo,n,'size')
            if f.lfs: assert f.lfs.sha256==s['sha256'],(repo,n,'sha256')
            else:
                data=(folder/n).read_bytes()
                assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==f.blob_id
        graph=next(n for n in files if n.endswith('.tflite'))
        assert get_hf_file_metadata(hf_hub_url(repo,graph,revision=info.sha),token=False).size==files[graph]['bytes']
        result=dict(repository=repo,revision=info.sha,public=True,gated=False,verifiedFiles=len(files),
            bytes=sum(s['bytes'] for s in files.values()),runtimeBytes=json.loads((folder/'quantization.json').read_text())['runtimeBytes'],
            allFileSizesAndHashesMatch=True,anonymousGraphDownloadVerified=True)
        results.append(result); print(json.dumps(result),flush=True)
        (root/'phone-publication-results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--upload',action='store_true')
    a=p.parse_args();publish(Path(a.root).resolve(),a.upload)
