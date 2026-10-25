# Generate vocabulary images locally

## Bundled demo illustrations

The demo APK includes 10,000 precomputed 520×520 JPEGs in `app/src/main/assets/word-images/`,
moved from `output/images/`. These originals occupy about 548 MB and are packaged with every build.
`WordImageAsset.path` follows the generator's filename rules, including capitalization, punctuation,
and category names. Keep this folder's files when sharing or rebuilding the demo.

The word list decodes small thumbnails off the main thread. The word assistant shows a larger
illustration. By default, introductions and follow-up replies send text only. After reply generation,
a separate coroutine job sends the JPEG to Gemma once to produce a caption and visual summary.
Completed descriptions are silently included as text context in subsequent replies; they are never
shown as chat messages or spoken. Descriptions are cached in app-private storage by image content
and caption prompt, so changed artwork or instructions invalidate the cache. Foreground questions
cancel unfinished caption inference and wait for native cleanup; captioning retries after the reply.
Leaving the assistant or backgrounding the app cancels unfinished work. A missing image, unavailable
vision model or caption failure leaves the text conversation usable. The Settings switch
**Assistant: include image in Gemma input** opts back into sending the JPEG with every reply.
Microphone transcription receives audio only; the audio/vision engine stays loaded between turns. Assistant replies remain
plain speech, while the photo lab retains its structured object-detection output.
If a word has no bundled image, its row and assistant continue without an illustration.

`scripts/generate_word_images.py` reads `seed/kids_english_dictionary.txt` (100 categories × 100 words). Defaults: `black-forest-labs/FLUX.2-klein-4B`, `./output/images`, exactly 520×520 pixels, PNG, four inference steps, guidance 1.0. The dictionary path is relative to the script's project; the output path is relative to your working directory.

## Setup and run

Use Python 3.11 or 3.12 in a virtual environment:

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r scripts/requirements-image-generation.txt

# Inspect all 10,000 entries without downloading a model or writing files.
python scripts/generate_word_images.py --dry-run

# Generate a small sample, then run the complete job.
python scripts/generate_word_images.py --limit 3
python scripts/generate_word_images.py
```

For NVIDIA, install the appropriate CUDA PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The default `--device auto` places the pipeline on the first available device in this order: CUDA GPU → Apple MPS GPU → CPU. Availability means the GPU is usable by the installed PyTorch build. Startup prints the selected device and reports when no supported GPU is available. GPU initialization errors stop the run rather than silently switching to CPU. Use `--device cuda` or `--device mps` to require that GPU backend, or `--device cpu` to explicitly choose CPU. Automatic precision uses BF16 on supported CUDA GPUs and FP32 otherwise for compatibility. On Apple Silicon, `--device mps --dtype bfloat16` can reduce unified memory use when supported by your PyTorch/macOS combination. CPU inference is supported but a 10,000-image run can take a long time.

The [official Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) supplies the four-step, guidance-1.0 sampling settings and an example using CUDA CPU offload. This script uses full Diffusers weights, including its text encoder and VAE, rather than the Android app's quantized GGUF model. Allow substantial RAM and disk space for the complete pipeline. Use `--cpu-offload` on CUDA to reduce VRAM use.

## Options

```sh
# Custom output, dimensions and JPEG encoding.
python scripts/generate_word_images.py --output-dir ./output/jpg --width 768 --height 512 --format jpg

# Select categories/words; repeated flags select a union within each filter.
python scripts/generate_word_images.py --category animal --category bird --limit 10
python scripts/generate_word_images.py --category animal --word dog --word cat

# Keep weights in a chosen cache; later require cached files only.
python scripts/generate_word_images.py --cache-dir ./output/model-cache --cpu-offload
python scripts/generate_word_images.py --cache-dir ./output/model-cache --offline

# Other text-to-image Diffusers repositories or local pipeline directories.
# Choose sampling settings appropriate to the selected model.
python scripts/generate_word_images.py --model stabilityai/sd-turbo --steps 1 --guidance-scale 0
python scripts/generate_word_images.py --model /path/to/local/diffusers-model --offline

# Override the vocabulary illustration prompt.
python scripts/generate_word_images.py --prompt-template 'A clear illustration of {word}, in the context of {category}. No text.'
```

`--model` accepts a complete Diffusers text-to-image pipeline that returns PIL images and accepts `prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, and `generator`. Single GGUF/checkpoint files and pipelines requiring additional inputs are outside this script's loader. `--revision` selects a model branch/tag/commit; `--dictionary` selects another file with the same `Category:` format. `--help` lists all flags.

## Downloads, filenames and resume

`DiffusionPipeline.from_pretrained` automatically downloads missing components into the Hugging Face cache and reuses cached weights. `--cache-dir` overrides the normal location (`HF_HOME` is also respected). Interrupted downloads retain the Hub's partial-download state; rerun to retry. `--offline` requires every needed component to be cached. If a replacement model is gated/private, obtain access and authenticate with `hf auth login` or `HF_TOKEN` before running. Inference runs locally.

Files are named `<word>_<category>.png` or `.jpg`; for example, `guinea-pig_animal.png` and `sea-lion_sea-creature.png`. Unicode letters, numbers, underscores and dashes are preserved; runs of other characters become a dash. The script rejects filename collisions, including case-insensitive collisions, before loading a model.

The tqdm generation bar starts at zero and counts only pending images selected for the current run, after filters and `--limit`; cached images are excluded from both the counter and total. It shows elapsed time, estimated remaining time, generation rate, current filename, and generated/failed counts. If every selected image is cached, no generation bar or model load is needed. With `--continue-on-error`, the bar counts processed attempts, including failures; the failed count and exit status still report unsuccessful images.

Resume state is the image files themselves. Existing images are skipped when they decode successfully and match the requested dimensions and format. Empty, corrupt or wrong-size files are regenerated. Each new image is written to a temporary `.part` file and atomically renamed after a successful save. A killed process may leave a `.part` file; it never counts as complete and can be removed. Ctrl-C exits with status 130. Rerun the same command to resume, with no separate checkpoint to recover. `--limit` counts pending images, so each rerun can generate another chunk. By default an error stops the run; `--continue-on-error` attempts the remaining entries and exits nonzero if any failed.

Seeds are derived from the base `--seed`, word and category, so filtering or resuming does not change a word's seed. Hardware/library differences can still change generated pixels. A valid existing image is skipped even if the model, prompt or seed changes; use a separate output folder for variants. PNG and JPG are tracked separately. Run one generator per output folder.

FLUX needs dimensions aligned with its latent packing. The script rounds generation dimensions up to the pipeline's alignment (528×528 for the default 520×520 request), then resizes with Lanczos to the exact requested output dimensions. Prompts follow the Android app's vocabulary illustration style, with action, feeling, color and shape scene variants. Illustrations for abstract or ambiguous words should be reviewed for meaning.

`--dry-run` requires no third-party packages. It lists counts and previews the first five pending prompts; because Pillow is not required in this mode, nonempty existing files are presumed complete until a real run validates them.

## Script checks

The regression suite uses real Pillow PNG/JPEG encoding and a fake inference pipeline. It covers resume after failure, stable seeds, corrupt-file recovery, atomic saves, exact sizes, filters, collisions, and model-loading options without downloading weights:

```sh
python -m unittest discover -s scripts -p test_generate_word_images.py -v
```
