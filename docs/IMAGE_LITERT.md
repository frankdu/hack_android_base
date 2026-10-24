# LiteRT text-to-image

The Images screen includes five LiteRT entries alongside the three existing GGUF
baselines. They run through LiteRT **2.3.0 CompiledModel**, independently of the
Gemma/LiteRT-LM engine. No Tensor SDK is needed for CPU or GPU. Tensor TPU remains
a separate AOT bundle workflow; see [IMAGE_TPU.md](IMAGE_TPU.md).

| LiteRT entry | Output | Steps | Installation |
|---|---|---|---|
| FLUX.2 Klein 4B | 256 × 256 | 4 | In-app download, 7.45 GB |
| Bonsai Image 4B | 512 × 512 | 4 | In-app download, 6.54 GB including CPU/GPU DiTs |
| Iris 3B Turbo | 768 × 768 | 4 | In-app download, 13.71 GB |
| SD-Turbo | 512 × 512 | 1 | Stage converted portable bundle, 5.03 GB |
| SDXL-Turbo | 512 × 512 | 4 | Stage converted portable bundle, 13.75 GB |

The community repositories, immutable revisions, sizes and SHA-256 hashes are in
`CommunityImageModels.kt`. Android DownloadManager transfers survive process
exit. All component files must finish verification before generation is enabled.
GGUF and `.litertlm` files are not accepted as image graph inputs.

## Android use

1. Open **Research Lab → Images** and choose an entry ending in **LiteRT**.
2. Download it, or stage an SD/SDXL portable bundle and tap **Verify staged LiteRT bundles**.
3. Select **CPU** or **GPU**, enter the prompt and seed, and generate. The model's
   default step count is fixed. Use the 30-word benchmark to save comparisons.

FLUX and Iris GPU mode requests FP32 GPU for every neural graph. Bonsai GPU mode
uses its GPU-shaped DiT and CPU for the text encoder and VAE. Portable SD/SDXL
GPU mode requests FP32 GPU for every graph, but the generic converted graphs are
not GPU-tuned and GPU compatibility has not been established. A backend error is
reported; generation does not retry on another accelerator. A failed LiteRT
model/method skips only that combination's remaining benchmark jobs.

Each image runs in the isolated `:imagebenchmark` worker process. Stop and timeout
can terminate a stalled driver call. One Environment is shared across an image's
graphs, and every model and TensorBuffer is closed. FLUX and Iris stream one graph
at a time. Bonsai keeps its DiT loaded through four steps, with text/VAE loaded
separately. Portable SD plans stream shards. LiteRT sessions are recreated per
image, including in the retained benchmark worker; compilation/loading are included
in latency. Large downloads and sequential graphs do not guarantee that a model
fits a particular phone's runtime memory budget.

Results record actual dimensions, runtime, accelerator description, hashes and
encode/denoise/decode times. FLUX, Iris and SD plans use `java.util.Random` Gaussian
noise; Bonsai uses the upstream Android SplitMix64/Box–Muller stream. A matching
integer seed does not imply identical noise across runtimes. Benchmark results
at 256, 512 and 768 px should be compared with resolution and quality in view.

## Portable SD/SDXL conversion

The Linux work directory used for this integration is
`192.168.4.103:/mnt/sda1/frankdu/litert-image-support/`. The secondary disk has space;
the machine's root disk is nearly full. Checkpoints, caches and temporary files
were placed on the secondary disk. Existing jobs and environments were preserved.

```sh
python3.12 -m venv .venv-image-litert
. .venv-image-litert/bin/activate
pip install -r scripts/image_litert/requirements.txt

python scripts/image_litert/workflow.py fetch SD_TURBO --output models/sd-source
python scripts/image_litert/workflow.py export --source models/sd-source --output models/sd-export
python scripts/image_litert/validate_plan.py models/sd-export --out models/sd-check.png

# Repeat fetch/export/validate_plan for SDXL_TURBO and separate directories.
# Copy a finished export back to the development machine, then:
python scripts/push-image-litert.py models/sd-export --serial DEVICE_SERIAL
```

The exported `manifest.json` has `compiled:false`. Stage it under
`image-litert/SD_TURBO/portable/` or `image-litert/SDXL_TURBO/portable/`.
The staging script validates hashes and publishes the manifest last. The app
checks hashes, dependencies and tokenizer fixtures before enabling a bundle.
Export uses original Diffusers components, model-specific schedulers and static
weight-bounded shards. Integer I/O is explicitly preserved with `enable_x64=True`.
All shards must pass numerical parity before the manifest is published. The fetch
filters exclude duplicate root checkpoints, fp16 variants and ONNX data blobs.
Source licenses are included; conversion does not change them.

Use `litert run GRAPH.tflite --desktop --cpu` to inspect one graph. This exercises
one graph with dummy inputs; a full pipeline check needs `validate_plan.py`.
LiteRT CLI setup is isolated in a Python virtual environment, as required by its
skill. No new model repository was published.

## FLUX host reconstruction

The published FLUX repository contains the graphs and tokenizer but omits the
host constants referenced by its model card. `prepare_flux_host.py` reconstructs
the fixed time embeddings and VAE normalization values from the pinned original
checkpoint. The resulting 256 KB `flux_host.json` is packaged in app assets; it
contains no prompt or random noise. The Kotlin host computes Qwen rotary tables,
a causal/padding mask expanded to all 32 heads, the three hidden-state taps,
DiT rotary tables, Euler updates and VAE unpatchify. Graph inputs are mapped by
numbered argument names, never by shape. Every output is checked for finiteness.

```sh
python scripts/image_litert/prepare_flux_host.py \
  --source models/flux-source \
  --output app/src/main/assets/image-litert/flux_host.json
```

The source checkpoint revision and community graph revision are recorded in the
asset. `validate_flux.py` replays this graph ABI on desktop CPU with a supplied
Android-compatible starting-noise file. The graph signatures were inspected
from the downloaded flatbuffers, including output order and rotary tensor shapes.

Bonsai math/tokenization and the Iris reference engine are adapted under Apache
2.0. See [source attribution and changes](../third_party/litert-image-pipelines/README.md).
License and notice files are also included in the APK.

## Validation

See [IMAGE_LITERT_VALIDATION.json](IMAGE_LITERT_VALIDATION.json) for exact outcomes.
The APK, lint, all 92 JVM tests and six Python workflow tests pass. All five
models generated recognizable apple images on desktop LiteRT CPU. SD/SDXL
full-plan replays matched their export references at 79.8 / 74.0 dB PSNR. These
are smoke checks, not comparable performance or quality benchmarks. Published tokenizer fixtures and
host tensor-layout/sampler tests run without model downloads. Desktop generation
checks exercise actual weights and graph execution. They do not establish Android
GPU compatibility, phone memory use or phone latency. No Android device was
attached when the initial integration was built.
