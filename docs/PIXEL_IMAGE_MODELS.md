# Pixel text-to-image quantization audit — 2026-10-24

The old SD-Turbo and SDXL-Turbo LiteRT bundles were **FP32**, not phone-sized
quantized exports. They have now been quantized with LiteRT CLI on
`192.168.4.103` and published separately. Original public repositories are preserved.

| Active model | Previous app download | Current app download | Stored weights / decision |
|---|---:|---:|---|
| FLUX.2 Klein 4B | 7.45 GB | 7.45 GB | Already INT8, streamed shards; retain community conversion |
| Bonsai Image 4B | 6.54 GB | 6.54 GB | Already INT4; CPU is the practical phone candidate; GPU requires measurement |
| SD-Turbo | 5.03 GB | **1.32 GB** | New dynamic INT8 encoders/UNet, FP16 VAE weights; recommended starting model |
| SDXL-Turbo | 13.75 GB | **3.53 GB** | Same mixed recipe; retain as the larger comparison model |

Sizes are decimal GB for the app's runtime file lists, not the entire Hub repo.
The SD reductions are 73.75% and 74.35%. Both new bundles have a largest graph
of 99.35 MB. All 21 SD and 55 SDXL neural graphs passed a flatbuffer audit of
their stored INT8/FP16 constants, including external-buffer storage.

## Why file size is not a phone memory requirement

[Pixel 10 Pro](https://store.google.com/product/pixel_10_pro_specs) has Tensor G5
and 16 GB RAM. [Pixel 11 Pro](https://store.google.com/product/pixel_11_pro_specs)
has Tensor G6 and either 12 or 16 GB RAM depending on storage configuration.
Android, other apps, activations, runtime buffers and driver caches all need RAM.
The conversions target this memory class, but RAM capacity alone does not certify
an accelerator or exact phone. Tensor G6 must be measured independently.

FLUX already streams its roughly 912 MB maximum graph, so a 7.45 GB download
does not mean 7.45 GB resident at once. Its current community INT8 pipeline
completed an actual app CPU generation on **Pixel 10 Pro Fold / Tensor G5**:
256×256, four steps, blue-teapot prompt, seed 7, **85.76 seconds**, process peak
**2.09 GiB RSS**. The image was inspected as a recognizable blue teapot. This is
one smoke sample on the Fold, not a Pixel 10 Pro/11 Pro benchmark.

All four current pipelines completed CPU generation on that Tensor G5 phone:

| Model | Output / steps | Time including graph loads and PNG | Worker peak RSS |
|---|---|---:|---:|
| FLUX Klein INT8 | 256×256 / 4 | 85.76 s | 2.09 GiB |
| Bonsai INT4 | 512×512 / 4 | 135.96 s | 4.18 GiB |
| SD-Turbo mobile INT8/FP16 | 512×512 / 1 | 26.92 s | 1.75 GiB |
| SDXL-Turbo mobile INT8/FP16 | 512×512 / 4 | 86.90 s | 1.75 GiB |

The fixed test prompt was a blue ceramic teapot, seed 7. All final outputs were
inspected as recognizable blue teapots. Each measurement used a fresh isolated
worker; these are smoke results, not controlled multi-run benchmarks. SD/SDXL and
Bonsai used a separate debug application ID and private model files so another
session's main-app deployments could continue. The LiteRT execution code, weights
and four-thread CPU configuration match the app. Private model storage can affect
loading time compared with the main app's external-files directory.

**Start with SD-Turbo CPU.** FLUX CPU also has a modest measured footprint despite
its larger download. Bonsai CPU fits the measured 16 GB phone but needs much more
headroom. SDXL CPU now fits as well, with a larger download and longer generation.
No result here guarantees the 12 GB Pixel 11 Pro under other memory loads.

Bonsai's 6.54 GB app download includes two approximately 2.27 GB DiT variants
(CPU-shaped and GPU-shaped), plus the approximately 1.80 GB INT4 text encoder
and 0.20 GB VAE. A CPU-only required file set would be about 4.27 GB; the current
shared CPU/GPU catalog still downloads both variants. Re-quantizing an already
INT4 model is not a reliable memory optimization. GPU weight expansion can make
the runtime much larger than the compressed file. The upstream
[Bonsai card](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B)
reports phone CPU operation and GPU memory/compatibility limitations; those are
upstream measurements, not measurements of either requested Pro phone.

## Published compact models

- [frankdu/SD-Turbo-Mobile-INT8-LiteRT](https://huggingface.co/frankdu/SD-Turbo-Mobile-INT8-LiteRT)
- [frankdu/SDXL-Turbo-Mobile-INT8-LiteRT](https://huggingface.co/frankdu/SDXL-Turbo-Mobile-INT8-LiteRT)

Both repositories are public and ungated. All 184 published files passed remote
size/hash checks; see [immutable publication revisions](PIXEL_IMAGE_PUBLICATION.json)
and [CPU device measurements](PIXEL_IMAGE_VALIDATION.json). Runtime manifests retain the existing
`word-image-litert-v1` ABI, 512×512 output, original sampler, original fixed step
count, tokenizer fixtures and int64 token I/O. They are portable CPU/GPU
candidates, not Tensor TPU AOT binaries. Source licenses, notices, attribution,
reproduction scripts, examples, checksums and complete quality reports are included.

Android now pins the new compact snapshots for SD/SDXL. FLUX and Bonsai continue
to prefer the pinned community artifacts, with equivalent frankdu mirrors.
The updated live source audit passes for all 136 runtime files / 164 source checks.

## Conversion and quality validation

Tool versions: `litert-cli-nightly 0.3.0.dev20261024`,
`ai-edge-quantizer 0.10.0`, `ai-edge-litert 2.3.0`.

`quantize_phone.py` calls the LiteRT CLI quantization command for every neural
graph. Encoders and UNet use `dynamic_wi8_afp32`; VAE uses a custom FLOAT_CASTING
FP16 weight recipe. Activations and sampler remain float. Original exports are
read-only; output paths, sizes and SHA-256 values are recorded in quantization.json.

The installed quantizer has a >32 MiB convolution broadcasting bug: its chunking
code reshapes weights to 2D before trying to broadcast 4D scales. The guarded
`litert_quantize_conv.py` CLI entry point broadcasts in the original shape first,
then reshapes weights and scales together. It changes the function in memory,
leaves installed packages unchanged and fails if the audited version/source changes.
Its result matched unchunked per-channel INT8 arithmetic **bit for bit** on a
36 MiB convolution tensor.

Both complete quantized pipelines generated apple, blue-teapot and orange-cat
images with fixed prompts/seeds and the same Java Gaussian stream as their FP32
LiteRT baselines. All six quantized images were inspected: subjects remain clear,
but composition/detail can change. SD pixel PSNR spans 14.24–18.74 dB;
SDXL spans 22.37–24.64 dB. These are three-case smoke comparisons, not broad
semantic-quality evaluation or pixel-equivalence claims. The Hub cards include
side-by-side grids and full per-prompt metrics.

Desktop quantized generation took about 10.3–10.5 seconds for SD and 31.8–35.7
seconds for SDXL, including graph loads and validation overhead. Python process
peak RSS was about 2.3 GiB, dominated by more than stored weights. These desktop
figures must not be presented as phone timings or phone memory guarantees.

## Reproduction

The Android investigation also found a separate, significant input/output buffer
alias bug. Some CLIP shards expose their input token tensor as a passthrough output.
The old runner allocated a fresh buffer for this output, replacing the input's
allocation with uninitialized data. This caused incorrect conditioning in SD and
an out-of-bounds embedding lookup/JNI abort in SDXL. The new manifests identify
exact aliases by flatbuffer tensor index using `tensor_aliases.py`. Android reuses
the input buffer for those outputs and closes shared buffers once. Both corrected
pipelines completed phone generation; a real int64 passthrough conversion test
also passes. Future FP32 exports receive the same alias annotations.

Both SD GPU attempts failed in LiteRT 2.3's model builder on batch/layout shapes,
including when text encoder, sampler and VAE stayed on CPU and only UNet used GPU.
These bundles are therefore marked GPU-unavailable in the app, with the reason
visible and benchmark jobs skipped. A compatible GPU export is still required;
quantized file size alone does not establish GPU support. FLUX/Bonsai GPU status
is unchanged and has not been validated on this phone in this audit. TPU remains
dependent on a separate verified Tensor AOT bundle.

Activate the workstation venv, then run:

```sh
python quantize_phone.py --source publications/SD-Turbo-LiteRT --output phone-quantization/SD-Turbo-Mobile-INT8-LiteRT
python validate_phone_quantization.py --source publications/SD-Turbo-LiteRT --bundle phone-quantization/SD-Turbo-Mobile-INT8-LiteRT
python audit_phone_weights.py phone-quantization/SD-Turbo-Mobile-INT8-LiteRT
```

Repeat for SDXL. Publication requires completed visual-review records and uses an
exact file whitelist, verifies all remote hashes/sizes and checks anonymous graph
downloads. Authentication uses the existing token file without exposing its value.

Physical inference uses `PhoneImageModelInstrumentedTest` with explicit
`phoneImageInference`, `imageModel` and `imageMethod` arguments. Each invocation
runs one model/backend in an isolated worker and records a PNG, hashes, timings,
process peak RSS or a failure reason. A separate validation application ID is used
when another session is deploying the main app. Main-app FP32 portable bundles
were preserved as `portable-fp32-backup-20261024` before staging compact versions.
