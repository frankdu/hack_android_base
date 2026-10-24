#!/usr/bin/env python3
"""Download, AOT compile and package image models. No credentials are printed."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

SOURCES = {
    "SD_TURBO": "stabilityai/sd-turbo",
    "SDXL_TURBO": "stabilityai/sdxl-turbo",
    "FLUX_KLEIN": "black-forest-labs/FLUX.2-klein-4B",
}
SOURCE_REVISIONS = {
    "SD_TURBO": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
    "SDXL_TURBO": "71153311d3dbb46851df1931d3ca6e939de83304",
    "FLUX_KLEIN": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
}
FLUX_LITERT = "litert-community/FLUX.2-klein-4B-LiteRT"
FLUX_REVISION = "223f4c989d8e5275fa8e695ddebea8d7ac7bada6"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def doctor(_):
    print(f"Host: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    for package in ["torch", "diffusers", "transformers", "litert-torch", "ai-edge-litert", "ai-edge-litert-sdk-google-tensor-nightly", "huggingface-hub"]:
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
    sdk = os.environ.get("GOOGLE_TENSOR_SDK_BETA", "")
    print("Tensor SDK artifact: " + ("present" if sdk and Path(sdk).is_file() else "MISSING; set GOOGLE_TENSOR_SDK_BETA"))
    print("Supported compiler host: " + str(platform.system() == "Linux" and platform.machine() == "x86_64"))
    try:
        from huggingface_hub import HfApi
        # Only report login state. Do not disclose the token or the account's private repositories.
        HfApi().whoami()
        print("Hugging Face: logged in")
    except Exception:
        print("Hugging Face: no verified login (public downloads can still work)")


def discover(args):
    """Query actual Hub file lists, not search-engine snippets. Save the audit."""
    from huggingface_hub import HfApi
    api = HfApi()
    audit = []
    for query in ["sd-turbo", "sdxl-turbo", "FLUX.2-klein-4B"]:
        for entry in api.list_models(search=query, limit=100):
            info = api.model_info(entry.id)
            files = [f.rfilename for f in info.siblings if f.rfilename.endswith((".tflite", ".litertlm"))]
            if files:
                audit.append(dict(repository=entry.id, revision=info.sha, files=files))
                print(entry.id, info.sha, ", ".join(files))
    write_json(args.output, audit)


def fetch(args):
    from huggingface_hub import HfApi, snapshot_download
    repo = FLUX_LITERT if args.ready_flux else SOURCES[args.model]
    if args.ready_flux and args.model != "FLUX_KLEIN":
        raise ValueError("--ready-flux requires FLUX_KLEIN")
    revision = args.revision or (FLUX_REVISION if args.ready_flux else SOURCE_REVISIONS[args.model])
    # Resolve mutable references once; downstream commands use this immutable snapshot.
    info = HfApi().model_info(repo, revision=revision)
    # Diffusers only needs component checkpoints. Avoid duplicate root checkpoints,
    # fp16 variants and ONNX external-data blobs (which do not end in .onnx).
    patterns = ["model_index.json", "README.md", "*LICENSE*", "*NOTICE*", "*policy*",
                "*/config.json", "*/preprocessor_config.json", "scheduler/*", "*/*.safetensors", "*/tokenizer*",
                "*/vocab*", "*/merges*", "*/special_tokens*", "*/added_tokens*", "*/chat_template*"]
    if args.ready_flux:
        patterns = ["ke_*.tflite", "kc_*.tflite", "kv_vae.tflite", "tokenizer/*", "README.md", "*LICENSE*"]
    path = snapshot_download(repo, revision=info.sha, allow_patterns=patterns,
                             ignore_patterns=None if args.ready_flux else ["*.fp16.*", "*.onnx*", "*.bin", "*.msgpack", "*.h5"],
                             local_dir=args.output)
    write_json(Path(args.output) / "source.json", dict(repository=repo, revision=info.sha, model=args.model,
        format="community-litert-256" if args.ready_flux else "diffusers"))
    print(path)


def compile_graph(source, destination, soc, truncation):
    # The stable 2.3.0 package requires this before importing its AOT modules.
    os.environ["GOOGLE_TENSOR_BACKEND_ENABLED"] = "1"
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.vendors.google_tensor import target
    result = aot_lib.aot_compile(str(source), output_dir=str(destination.parent / "compiler-work" / source.stem),
        target=[target.Target(soc_model=target.SocModel(soc))], keep_going=False,
        google_tensor_truncation_type=truncation,
        google_tensor_enable_large_model_support=True)
    if result.failed_backends or len(result.models) != 1:
        raise RuntimeError(f"Compilation failed for {source.name}: {result.compilation_report()}")
    # Model.save is the documented Model object API; avoid guessing export-generated filenames.
    result.models[0].save(destination, export_only=True)
    report = result.compilation_report()
    destination.with_suffix(".compiler.txt").write_text(report)
    print(report)
    stats = result.models[0].partition_stats
    if stats is None:
        raise RuntimeError("Compiler did not report partition coverage; refusing an unverifiable TPU label")
    return dict(offloadedOps=sum(s.num_ops_offloaded for s in stats.subgraph_stats),
                totalOps=sum(s.num_total_ops for s in stats.subgraph_stats),
                partitions=sum(s.num_partitions_offloaded for s in stats.subgraph_stats))


def compile_bundle(args):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Google Tensor AOT requires Linux x86_64. Run this command on the conversion workstation.")
    source, out = Path(args.source), Path(args.output)
    if source.resolve() == out.resolve():
        raise SystemExit("Use separate source and output directories.")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest and manifest.get("format") != "word-image-litert-v1":
        raise SystemExit("Unknown image bundle format")
    out.mkdir(parents=True, exist_ok=True)
    if manifest:
        for name, spec in manifest["files"].items():
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()) or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError(f"Unsafe or modified source artifact: {name}")
        graphs = [source / name for name in manifest["files"] if name.endswith(".tflite")]
    else:
        graphs = sorted(source.rglob("*.tflite"))
    # A failed recompile must not leave an earlier manifest claiming the directory is ready.
    (out / "manifest.json").unlink(missing_ok=True)
    if not graphs:
        raise SystemExit("No .tflite graphs found. Run export first, or fetch --ready-flux.")
    coverage = {}
    for graph in graphs:
        relative = graph.relative_to(source)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        coverage[str(relative)] = compile_graph(graph, target, args.soc, args.truncation)
    # Copy tokenizer/constants/licenses/reference images. Never copy raw HF checkpoints.
    if manifest:
        for relative in manifest["files"]:
            if relative.endswith(".tflite"):
                continue
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        manifest.update(targetSoc=args.soc, compiled=True, offload=coverage, validation="not-device-validated",
                        compiler=dict(litert=importlib.metadata.version("ai-edge-litert"), truncation=args.truncation),
                        files={name: dict(bytes=(out / name).stat().st_size, sha256=digest(out / name)) for name in manifest["files"]})
        if not any(coverage[a["model"]]["offloadedOps"] > 0 for a in manifest["actions"]
                   if a["phase"] == "denoise" and ("unet-" in a["model"] or "dit-" in a["model"])):
            raise RuntimeError("No denoiser operations offloaded; cannot create a TPU bundle")
        write_json(out / "manifest.json", manifest)
    else:
        # The community FLUX graphs need their matching host pipeline, not our different 512px plan.
        write_json(out / "compilation-only.json", dict(source=json.loads((source / "source.json").read_text()),
            targetSoc=args.soc, validation="not-device-validated", androidImportable=False,
            reason="Community FLUX graphs are 256px with a separate host loop. Use export for an app-compatible 512px bundle."))
    print(f"Compiled graphs: {out}. AOT success is not proof of numerical correctness or TPU execution.")


def publish(args):
    from huggingface_hub import HfApi
    folder = Path(args.source)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("compiled"):
        raise SystemExit("Only a compiled app bundle can be published with this command.")
    for relative, spec in manifest["files"].items():
        if not (folder / relative).resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"Unsafe bundle path: {relative}")
        if digest(folder / relative) != spec["sha256"]:
            raise SystemExit(f"Hash mismatch: {relative}")
    model = manifest["modelId"]
    license_name = "apache-2.0" if model == "FLUX_KLEIN" else "other"
    card = f'''---
license: {license_name}
base_model: {SOURCES[model]}
pipeline_tag: text-to-image
tags: [litert, google-tensor, android, experimental]
---
# {model} — experimental LiteRT {manifest['targetSoc']} bundle

Source revision: `{manifest['source']['revision']}`. Resolution: 512×512.
Steps: {manifest['steps']}. Format: `word-image-litert-v1` for this Research Lab app.

Validation: **{manifest['validation']}**. No TPU speed/quality claim has been established.
Google Tensor AOT may partition unsupported operations onto CPU. See the per-graph compiler reports.
CPU tokenization, random noise and inter-graph copies are part of end-to-end latency.

Use the source model's license and acceptable-use requirements (included in this repository).
Conversion does not change those terms. This repository is a derivative of the base model above.
Do not compare different resolutions or claim a TPU speedup without on-device benchmarks.
'''
    (folder / "README.md").write_text(card)
    if not args.upload:
        print(f"Prepared {folder / 'README.md'}. Review licenses, compiler reports and device results. No upload performed.")
        return
    if not args.repo or args.repo.count("/") != 1:
        raise SystemExit("--repo must be username/repository or organization/repository")
    api = HfApi()
    api.whoami()  # fail before creating anything if login is absent
    api.create_repo(args.repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(folder),
                      allow_patterns=list(manifest["files"]) + ["manifest.json", "README.md", "*.compiler.txt"],
                      ignore_patterns=["**/compiler-work/**", "*.tmp", ".cache/**"],
                      commit_message="Upload experimental Google Tensor image-generation bundle")
    print(f"https://huggingface.co/{args.repo}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").set_defaults(func=doctor)
    d = sub.add_parser("discover"); d.add_argument("--output", default="artifacts/image-litert/discovery.json"); d.set_defaults(func=discover)
    f = sub.add_parser("fetch"); f.add_argument("model", choices=SOURCES); f.add_argument("--revision"); f.add_argument("--ready-flux", action="store_true"); f.add_argument("--output", required=True); f.set_defaults(func=fetch)
    e = sub.add_parser("export"); e.add_argument("--source", required=True); e.add_argument("--output", required=True); e.add_argument("--shard-mib", type=int, default=256); e.set_defaults(func=export)
    c = sub.add_parser("compile"); c.add_argument("--source", required=True); c.add_argument("--output", required=True); c.add_argument("--soc", choices=["Tensor_G5", "Tensor_G6"], required=True); c.add_argument("--truncation", choices=["auto", "bfloat16", "half", "no_truncation"], default="no_truncation"); c.set_defaults(func=compile_bundle)
    u = sub.add_parser("publish"); u.add_argument("--source", required=True); u.add_argument("--repo"); u.add_argument("--upload", action="store_true"); u.set_defaults(func=publish)
    return p


def export(args):
    from export_pipeline import export_pipeline
    export_pipeline(args)


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
