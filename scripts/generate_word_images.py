#!/usr/bin/env python3
"""Generate resumable vocabulary illustrations with a local Diffusers model.

Install scripts/requirements-image-generation.txt; see docs/LOCAL_WORD_IMAGES.md.
Planning (--dry-run) needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import string
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_PROMPT = (
    "{scene}. Children's vocabulary illustration, category: {category}. "
    "Clear subject, accurate proportions, bright natural colors, soft light, "
    "uncluttered composition. Show the meaning visually. "
    "No text, letters, labels, watermark or collage."
)


def read_dictionary(path: Path) -> list[tuple[str, str]]:
    """Read the same Category: sections as the Android dictionary parser."""
    entries = []
    category = None
    count = 0

    def finish():
        if category is not None and count == 0:
            raise ValueError(f"Category {category!r} has no words")

    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("Category:"):
            finish()
            category = line.removeprefix("Category:").strip() if line else None
            count = 0
            if category == "":
                raise ValueError(f"Missing category at line {line_number}")
        else:
            if category is None:
                raise ValueError(f"Expected Category: header at line {line_number}")
            entries.append((line, category))
            count += 1
    finish()
    if not entries:
        raise ValueError(f"No vocabulary found in {path}")
    return entries


def safe_component(value: str) -> str:
    # Preserve Unicode letters/numbers; replace spaces, punctuation and separators.
    value = unicodedata.normalize("NFC", value)
    result = re.sub(r"[^\w-]+", "-", value, flags=re.UNICODE).strip("-")
    if not result:
        raise ValueError(f"Cannot make a filename from {value!r}")
    return result


def output_name(word: str, category: str, extension: str) -> str:
    name = f"{safe_component(word)}_{safe_component(category)}.{extension}"
    if len(name.encode("utf-8")) > 240:
        raise ValueError(f"Filename is too long: {name!r}")
    return name


def make_prompt(word: str, category: str, template: str = DEFAULT_PROMPT) -> str:
    subject = " ".join(word.split())
    context = " ".join(category.split())
    if re.search(r"action|verb|movement", context, re.IGNORECASE):
        scene = f"one friendly character clearly performing {subject}, with the action easy to recognize"
    elif re.search(r"emotion|feeling", context, re.IGNORECASE):
        scene = f"a friendly character showing {subject} through facial expression and body language in a simple everyday scene"
    elif re.search(r"color|colour", context, re.IGNORECASE):
        scene = f"one familiar object in a clear {subject} color, filling most of the frame"
    elif re.search(r"shape", context, re.IGNORECASE):
        scene = f"one large {subject} shape with a clean outline on a plain background"
    else:
        scene = f"{subject}, recognizable features, a simple setting appropriate to {context}"
    return template.format(word=subject, category=context, scene=scene)


def entry_seed(word: str, category: str, seed: int) -> int:
    """Stable across restarts, filtering, and changes to dictionary order."""
    digest = hashlib.sha256(f"{seed}\0{word}\0{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def existing_image(path: Path, size: tuple[int, int], extension: str, *, dry_run=False) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if dry_run:
        return True  # A dependency-free plan cannot check image decoding.
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            if image.size != size or image.format != ("PNG" if extension == "png" else "JPEG"):
                return False
            image.load()  # Truncated files must not count as completed work.
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def save_image(image, path: Path, size: tuple[int, int], extension: str) -> None:
    """Publish a complete image atomically, including after Ctrl-C or disk errors."""
    from PIL import Image

    image = image.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            options = {"quality": 95} if extension == "jpg" else {}
            image.save(stream, format="PNG" if extension == "png" else "JPEG", **options)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_pipeline(args):
    # Delay heavyweight imports and model downloads until there is pending work.
    import torch
    from diffusers import DiffusionPipeline

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cpu":
            print("No CUDA or Apple MPS GPU is available to PyTorch; using CPU.", flush=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable in this PyTorch installation")
    if args.cpu_offload and device != "cuda":
        raise ValueError("--cpu-offload requires --device cuda (or auto selecting CUDA)")
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    print(f"Loading {args.model} on {device} ({dtype}); missing weights download to the HF cache.", flush=True)
    kwargs = {"torch_dtype": getattr(torch, dtype), "local_files_only": args.offline}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir.expanduser())
    if args.revision:
        kwargs["revision"] = args.revision
    pipeline = DiffusionPipeline.from_pretrained(args.model, **kwargs)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, torch


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Diffusers Hub repository ID or local pipeline directory")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "seed/kids_english_dictionary.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("./output/images"))
    parser.add_argument("--width", type=positive_int, default=520, help="exact saved width")
    parser.add_argument("--height", type=positive_int, default=520, help="exact saved height")
    parser.add_argument("--format", choices=("png", "jpg"), default="png", dest="extension")
    parser.add_argument("--steps", type=positive_int, default=4, help="Klein distilled default; adjust for other models")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="base for a stable per-word seed")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="auto prefers CUDA GPU, then Apple MPS GPU, then CPU")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--cpu-offload", action="store_true", help="offload between CPU and CUDA to reduce VRAM")
    parser.add_argument("--cache-dir", type=Path, help="override the standard Hugging Face cache")
    parser.add_argument("--revision", help="optional model revision or commit hash")
    parser.add_argument("--offline", action="store_true", help="load cached/local weights without network access")
    parser.add_argument("--category", action="append", help="exact category filter, case insensitive; repeatable")
    parser.add_argument("--word", action="append", help="exact word filter, case insensitive; repeatable")
    parser.add_argument("--limit", type=positive_int, help="generate at most this many pending images this run")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT, help="template using {word}, {category}, and/or {scene}")
    parser.add_argument("--continue-on-error", action="store_true", help="report failures and try remaining entries; exit nonzero if any fail")
    parser.add_argument("--dry-run", action="store_true", help="show plan without downloads or writes; existing nonempty files are presumed complete")
    args = parser.parse_args(argv)
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("--guidance-scale must be finite and nonnegative")
    try:
        fields = {field for _, field, _, _ in string.Formatter().parse(args.prompt_template) if field is not None}
        if not fields or fields - {"word", "category", "scene"}:
            raise ValueError("use at least one of {word}, {category}, {scene}; other fields are unsupported")
        if not make_prompt("dog", "animal", args.prompt_template).strip():
            raise ValueError("prompt must not be blank")
    except (ValueError, KeyError, IndexError) as error:
        parser.error(f"Invalid --prompt-template: {error}")
    return args


def run(args) -> int:
    entries = read_dictionary(args.dictionary.expanduser())
    # Check the whole source before filtering so collisions cannot silently skip words.
    filenames = {}
    for word, category in entries:
        name = output_name(word, category, args.extension)
        key = name.casefold()
        if key in filenames:
            raise ValueError(f"Filename collision: {filenames[key]!r} and {(word, category)!r} both map to {name!r}")
        filenames[key] = (word, category)
    for field in ("category", "word"):
        requested = getattr(args, field)
        if requested:
            wanted = {value.casefold() for value in requested}
            index = 1 if field == "category" else 0
            available = {entry[index].casefold() for entry in entries}
            if wanted - available:
                raise ValueError(f"Unknown {field} filter: {', '.join(sorted(wanted - available))}")
            entries = [entry for entry in entries if entry[index].casefold() in wanted]

    size = (args.width, args.height)
    output = args.output_dir.expanduser()
    if not args.dry_run:
        try:
            import PIL  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install dependencies: python -m pip install -r scripts/requirements-image-generation.txt") from error
    pending = []
    for word, category in entries:
        path = output / output_name(word, category, args.extension)
        if not existing_image(path, size, args.extension, dry_run=args.dry_run):
            pending.append((word, category, path))
    skipped = len(entries) - len(pending)
    selected = pending[:args.limit] if args.limit else pending
    print(f"Selected {len(entries)} entries; {skipped} complete; {len(pending)} pending; {len(selected)} to generate.", flush=True)
    print(f"Output: {output.resolve()} | {args.width}x{args.height} {args.extension}", flush=True)
    if args.dry_run:
        for word, category, path in selected[:5]:
            print(f"  {path.name}: {make_prompt(word, category, args.prompt_template)}")
        return 0
    if not selected:
        return 0

    try:
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError("Install dependencies: python -m pip install -r scripts/requirements-image-generation.txt") from error

    output.mkdir(parents=True, exist_ok=True)
    pipeline, torch = load_pipeline(args)
    # Flux packs 2x2 latent patches; other common pipelines need VAE alignment.
    alignment = int(getattr(pipeline, "vae_scale_factor", 8))
    if "Flux" in type(pipeline).__name__:
        alignment *= 2
    width = math.ceil(args.width / alignment) * alignment
    height = math.ceil(args.height / alignment) * alignment
    if (width, height) != size:
        print(f"Generating at aligned {width}x{height}, then resizing to {args.width}x{args.height}.", flush=True)
    generated = failed = 0
    # Only uncached entries selected for this run contribute to the progress total.
    with tqdm(total=len(selected), desc="Generating images", unit="image", dynamic_ncols=True) as progress:
        for word, category, path in selected:
            progress.set_postfix(file=path.name, generated=generated, failed=failed)
            try:
                # CPU random generators also work with CUDA/MPS and avoid MPS RNG limits.
                generator = torch.Generator(device="cpu").manual_seed(entry_seed(word, category, args.seed))
                with torch.inference_mode():
                    result = pipeline(
                        prompt=make_prompt(word, category, args.prompt_template),
                        width=width, height=height, num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale, generator=generator,
                    )
                save_image(result.images[0], path, size, args.extension)
                generated += 1
            except Exception as error:
                failed += 1
                tqdm.write(f"FAILED {path.name}: {error}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
            progress.set_postfix(file=path.name, generated=generated, failed=failed, refresh=False)
            progress.update(1)
    print(f"Finished: {generated} generated, {skipped} skipped, {failed} failed. Rerun the same command to resume.", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed images are safe; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}\nCompleted images are safe; rerun the same command to resume.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
