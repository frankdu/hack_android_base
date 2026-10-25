#!/usr/bin/env python3
"""Quantize existing SD/SDXL portable plans through LiteRT CLI, preserving the ABI.

Run in an activated LiteRT CLI virtual environment. Original exports are read-only.
Validation and publication are separate steps; a smaller file is not a phone pass.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantize(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    assert source != output and not output.is_relative_to(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['modelId'] in ('SD_TURBO', 'SDXL_TURBO') and not manifest['compiled']
    output.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.name + '-logs')
    logs.mkdir(exist_ok=True)
    from ai_edge_quantizer import recipe_manager, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex='.*', operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT), compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=recipe_manager.AlgorithmName.FLOAT_CASTING)
    recipe_path = output / 'fp16-recipe.json'
    recipe_path.write_text(json.dumps(rm.get_quantization_recipe(), indent=2) + '\n')
    report = dict(tool='litert quantize', versions={name: importlib.metadata.version(name) for name in
        ('litert-cli-nightly', 'ai-edge-quantizer', 'ai-edge-litert')},
        sourceManifestSha256=digest(source / 'manifest.json'), graphs=[], deviceValidated=False)
    start = time.monotonic()
    for name, spec in manifest['files'].items():
        src, dst = source / name, output / name
        assert src.resolve().is_relative_to(source) and dst.resolve().is_relative_to(output)
        assert src.stat().st_size == spec['bytes'] and digest(src) == spec['sha256'], name
        dst.parent.mkdir(parents=True, exist_ok=True)
        neural = name.startswith(('graphs/clip', 'graphs/unet', 'graphs/vae')) and name.endswith('.tflite')
        if neural:
            recipe = 'fp16' if name.startswith('graphs/vae') else 'dynamic_wi8_afp32'
            command = [sys.executable, str(Path(__file__).with_name('litert_quantize_conv.py')),
                'quantize', str(src), '--output', str(dst)]
            command += ['--custom-recipe', str(recipe_path)] if recipe == 'fp16' else ['--recipe', recipe]
            with (logs / (src.name + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert dst.stat().st_size < src.stat().st_size, f'No compression: {name}'
            report['graphs'].append(dict(path=name, recipe=recipe, originalBytes=src.stat().st_size,
                bytes=dst.stat().st_size, sha256=digest(dst), command=command))
            print(name, recipe, src.stat().st_size, '->', dst.stat().st_size, flush=True)
        else:
            shutil.copy2(src, dst)
    # Public tokenizers need their standard files as well as the Android bpe.json.
    for folder in source.glob('tokenizer*'):
        if folder.is_dir():
            for src in folder.iterdir():
                if src.is_file():
                    dst = output / folder.name / src.name
                    if not dst.exists(): shutil.copy2(src, dst)
    for name in ('NOTICE', 'LICENSE.md', 'CODE_LICENSE', 'SOURCE_MODEL_CARD.md', 'run_portable.py', 'java_noise.py', 'requirements.txt'):
        if (source / name).is_file(): shutil.copy2(source / name, output / name)
    manifest['files'] = {name: dict(bytes=(output / name).stat().st_size, sha256=digest(output / name))
        for name in manifest['files']}
    manifest['validation'] = 'quantized-awaiting-pipeline-validation'
    manifest['quantization'] = dict(neural='dynamic_wi8_afp32', vae='float16 weights / float32 activations',
        sourceManifestSha256=report['sourceManifestSha256'], tool='LiteRT CLI')
    from tensor_aliases import annotate_manifest
    report['signatureInputOutputAliases']=annotate_manifest(output,manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report.update(elapsedSeconds=time.monotonic() - start,
        originalRuntimeBytes=sum(s['bytes'] for s in json.loads((source / 'manifest.json').read_text())['files'].values()),
        runtimeBytes=sum(s['bytes'] for s in manifest['files'].values()),
        largestGraphBytes=max(s['bytes'] for n, s in manifest['files'].items() if n.endswith('.tflite')))
    (output / 'quantization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'graphs'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); quantize(a.source, a.output)
