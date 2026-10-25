# Four-model LiteRT testing workflow

The active image catalog is **FLUX.2 Klein 4B, Bonsai Image 4B, SD-Turbo and
SDXL-Turbo**. Iris was removed from the selectable/download list at the user's
request. Its catalog/runtime identifiers remain for reading existing results and
recovering older transfers. Historical runs and published repositories are preserved.

SD-Turbo and SDXL-Turbo now use the compact INT8/FP16 phone bundles (1.32 GB and
3.53 GB). See the [Pixel quantization audit](PIXEL_IMAGE_MODELS.md) for recipes,
quality comparisons, memory evidence and validation limits.

## Source preference and download recovery

- FLUX and Bonsai prefer their pinned `litert-community` artifacts. If an upstream
  artifact fails with a missing/rejected/transient HTTP response or fails integrity
  verification, DownloadManager retries the corresponding pinned `frankdu` mirror.
- No compatible SD-Turbo/SDXL-Turbo conversion was found in the `litert-community`
  Hub catalog audit. These entries use `frankdu/SD-Turbo-Mobile-INT8-LiteRT`
  and `frankdu/SDXL-Turbo-Mobile-INT8-LiteRT` snapshots directly.
- Every primary/mirror file shares the same expected size and SHA-256. A mirror
  cannot substitute a different model or recipe. Upstream README files map to the
  mirror's preserved SOURCE_MODEL_CARD.md. Revisions are immutable; the app does
  not consume arbitrary future `main` revisions or incompatible graph layouts.
- Wi-Fi waits are left to Android's retry policy. Storage failures do not trigger
  mirror downloads. Source indices and Wi-Fi policy survive process restart.
- Each result records its artifact hashes and known download sources. Older or
  manually staged files with no recorded origin are labeled `verified-local`.

SD/SDXL download into the portable-bundle directories already used by the executor.
After every file passes integrity checking, the app validates the manifest and
Python tokenizer fixtures, creates the bundle verification marker, and enables
CPU/GPU generation. Manual ADB staging remains supported but is no longer needed
for the standard portable downloads.

The live source audit passed for **136 unique files**, **164 primary/mirror file
checks** and **six immutable repository snapshots**. Re-run with:

```sh
python scripts/image_litert/audit_android_catalog.py
```

It checks Hub file sizes and LFS SHA-256 values, and downloads small Git-backed
metadata files to verify their SHA-256. It does not download model weights.

## Images tab

**Generate image** shows four model choices, CPU/GPU/TPU choices, a reason for each
backend's readiness/unavailability, selected-model download controls, quick prompt
presets, an editable prompt and seed, and recent output images. Word/category
helpers and detailed artifact links are expandable. Model changes retain the
selected backend so testing choices remain explicit; Generate is disabled until
that exact model/backend is ready.

**Compare models** opens the benchmark directly. It defaults to the four models
and three backend columns and shows a model/backend installation matrix. Use
Select all 4 or Select installed to adjust the comparison.

The Models tab uses the same active catalog and download cards. Legacy GGUF
baselines are available only under advanced model options and remain readable in
historical benchmark results.

## Runtime availability

LiteRT 2.3.0 requires both `litert` and `litert-gpu` dependencies for GPU execution.
The APK now includes `libLiteRtClGlAccelerator.so`. See Google's
[GPU dependency instructions](https://developers.google.com/edge/litert/next/gpu).
Runtime accelerators are probed once on the Lab worker. A Vulkan-capable device
alone does not enable LiteRT GPU.

- CPU: four-thread LiteRT. Portable SD manifests preserve int64 tensors.
- GPU: explicit FP32 LiteRT GPU. Bonsai uses its GPU-shaped DiT with text encoder
  and VAE on CPU; FLUX requests GPU for every neural graph. The current compact
  SD/SDXL exports are GPU-unavailable after actual Tensor G5 shape-compilation
  failures, including a denoiser-only GPU attempt. Their GPU benchmark jobs are
  skipped with a reason. Compilation/invocation errors never retry another backend.
- TPU: available only if the runtime exposes NPU, the device is supported Tensor
  G5/G6, and a verified AOT bundle matches the model, dimensions and step count.
  The portable/public files are not labeled TPU artifacts. The generic AOT
  executor now accepts compatible plans for active LiteRT IDs and uses the
  model's real output dimensions. No new TPU bundle was manufactured by this change.

## Benchmark protocol

The fixed 30-word suite is unchanged. Four models × three backends × 30 words
produce up to **360 sample records**. Runnable combinations generate images;
unavailable/uninstalled combinations are recorded as skipped with a reason.
A benchmark can run any installed subset without pretending other models ran.
Backend failure skips the remaining jobs for that model/backend; other combinations
continue. Speed and quality summaries exclude failed/skipped samples.

Portable and TPU manifest hashes are pinned separately for each model/backend
combination. Resume and invocation reject a changed manifest. Results and image
cards use actual dimensions: FLUX 256×256, the other active models 512×512.
Resolution, weights and runtime must be considered when comparing timings and quality.
The worker is retained per combination; LiteRT models/environments still reload
per image and those costs remain included.

## Validation

- Debug APK and instrumentation APK build successfully; Android lint passes.
- All 98 JVM tests pass, including source fallback rules, four-model selection,
  hardware/bundle availability and benchmark scheduling.
- Two emulator UI tests pass: benchmark results/history rendering, and the actual
  Research Lab Generate/Compare workflow with all four model choices visible and
  Iris absent. Screenshots were inspected on Pixel 8a API 36.
- The live Hub audit verifies all 136 catalog files and 164 primary/mirror entries.

The emulator tests do not download or execute large models. All four CPU pipelines
subsequently completed physical Pixel 10 Pro Fold / Tensor G5 generation; see the
[Pixel audit](PIXEL_IMAGE_MODELS.md) for memory, timings, alias-buffer correction
and GPU failures. Pixel 10 Pro, Pixel 11 Pro and Tensor TPU still need their own
device/backend validation.
