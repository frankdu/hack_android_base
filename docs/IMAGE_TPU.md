# Image generation on Google Tensor TPU — experimental integration

For the newer portable CPU/GPU integration of all five requested LiteRT image models, see [IMAGE_LITERT.md](IMAGE_LITERT.md). The historical results below describe the original GGUF/TPU work.

Research and implementation date: 2026-10-23. Models: SD-Turbo, SDXL-Turbo,
FLUX.2 Klein 4B. **No converted bundle or end-to-end TPU image has been validated
on a Pixel yet.** The Android integration and conversion workflow are prepared;
conversion/compiler/model compatibility and resulting image quality remain to be
established on the required workstation and phones.

## Research findings

| Model | Reusable Android artifact found | Google Tensor TPU status | Path prepared here |
|---|---|---|---|
| SD-Turbo | Existing GGUF is for stable-diffusion.cpp. No matching LiteRT artifact found in the Hub search. | Requires conversion + Tensor AOT compilation + device validation. | Export original Diffusers CLIP, UNet, sampler and VAE as static, sharded LiteRT graphs. |
| SDXL-Turbo | Existing GGUF is for stable-diffusion.cpp. No matching LiteRT artifact found in the Hub search. | Same; substantially larger UNet/two text encoders. | Export both CLIP encoders, UNet, sampler and VAE. |
| FLUX.2 Klein 4B | [`litert-community/FLUX.2-klein-4B-LiteRT`](https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT), revision `223f4c989d8e5275fa8e695ddebea8d7ac7bada6`. | Published results concern GPU and Qualcomm Hexagon, not Google Tensor TPU. | Reuse its graphs for a Tensor compiler feasibility test; export the original checkpoint for the app's consistent 512px execution plan. |

`workflow.py discover` queried up to 100 repository matches for each of the three
model names and inspected their Hub file lists. Only the FLUX repository above
had `.tflite` artifacts in that search. This is a bounded discovery result, not a
claim that no other conversion exists. Audit: `artifacts/image-litert/discovery.json`.

The community FLUX graphs are **256×256**, with a matching custom tokenizer,
embedding lookup, rotary/mask preparation, sampler and latent transformations.
They cannot be dropped into this app's 512×512 pipeline. Its card also requires
FP32 GPU arithmetic to avoid overflow; successful Tensor AOT compilation alone
would not establish accuracy. `fetch --ready-flux` + `compile` supports testing
those exact graphs without pretending they form an app-ready bundle.

[Google's Tensor SDK documentation](https://developers.google.com/edge/litert/next/tensor-sdk)
currently lists **Tensor_G5 and Tensor_G6**. The
[NPU guide](https://developers.google.com/edge/litert/next/npu) requires AOT
compilation for Google Tensor; on-device JIT is not supported. Qualcomm QNN
artifacts and the existing GGUF downloads cannot run on Tensor's TPU.

## Required setup

The current workstation is macOS ARM64. Its Docker engine is Linux ARM64 with
about 12 GB RAM. It is not Google's documented conversion host.

Provide a native **Linux x86_64** workstation (Google documents Ubuntu 22.04),
Python 3.11/3.12, **at least 64 GB RAM**, and substantial free disk. Float32
checkpoint loading/tracing of FLUX can need more than 64 GB; 128 GB is preferable.
The exporter currently prioritizes correctness over download size and does not
quantize weights; exported bundles can be much larger than the GGUF downloads.
All three large-model exports remain unvalidated. Start with SD-Turbo.

Obtain `litert_plugin_compiler.tar.gz` through the
[Tensor SDK beta signup](https://developers.google.com/edge/litert/next/tensor-sdk).
The [official AOT notebook](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/colab/LiteRT_AOT_Compilation_Tutorial.ipynb)
sets `GOOGLE_TENSOR_SDK_BETA` to that artifact before installing
`ai-edge-litert-sdk-google-tensor-nightly`.

```sh
python3.11 -m venv .venv-image-litert
. .venv-image-litert/bin/activate
pip install -r scripts/image_litert/requirements.txt
export GOOGLE_TENSOR_SDK_BETA=/absolute/path/litert_plugin_compiler.tar.gz
export GOOGLE_TENSOR_BACKEND_ENABLED=1
# Use the plugin version supplied/tested with your SDK artifact. This dated
# wheel was published when researched; do not mix arbitrary nightlies.
pip install ai-edge-litert-sdk-google-tensor-nightly==2.4.0.dev20261023
python scripts/image_litert/workflow.py doctor
```

The app uses LiteRT **2.3.0** and the matching Tensor dispatch AAR. Google's
[compilation FAQ](https://developers.google.com/edge/tensor-sdk/tensor-faqs)
allows stable `ai-edge-litert==2.3.0` with
`GOOGLE_TENSOR_BACKEND_ENABLED=1`. Compiler/dispatch/firmware compatibility must
still be checked with the supplied beta SDK. `doctor` reports package/login state
without exposing credentials. Resolve any dependency or SDK-version conflict on
the conversion workstation before exporting.

Use `hf auth login` locally if a source is gated or for later publishing. Accept
any source-model access/license terms on Hugging Face. Never put a token in chat,
source files, commands committed to Git, or a model card. Public source tokenizer
and metadata downloads succeeded here without login.

## Conversion commands

Each command has `--help`. Downloads resolve pinned source revisions, write
`source.json`, and do not fetch duplicate `.bin`/ONNX weights. GGUF is not an
input to this conversion.

```sh
python scripts/image_litert/workflow.py fetch SD_TURBO --output artifacts/image-litert/sd-source
python scripts/image_litert/workflow.py export --source artifacts/image-litert/sd-source --output artifacts/image-litert/sd-export
python scripts/image_litert/workflow.py compile --source artifacts/image-litert/sd-export --output artifacts/image-litert/sd-g5 --soc Tensor_G5
# Compile independently for Pixel 11 / G6:
python scripts/image_litert/workflow.py compile --source artifacts/image-litert/sd-export --output artifacts/image-litert/sd-g6 --soc Tensor_G6
```

Repeat `fetch` with `SDXL_TURBO` or `FLUX_KLEIN` and separate directories. Source
revisions are pinned in `workflow.py`; `--revision` explicitly overrides one.
Export uses **512×512, one step for SD-Turbo, four for SDXL/FLUX**, no classifier-free
guidance. It exports model-specific scheduler/latent transforms as graphs too.
The SD sampler is Euler ancestral from the source Diffusers configuration; FLUX
uses its source flow-matching schedule. These are separately converted pipelines,
so comparisons against GGUF include runtime/weights/sampler differences as well
as hardware. The same numeric seed is reproducible within a runtime, and does
not guarantee identical noise across the Java and native generators.

`export_pipeline.py` wraps the actual Diffusers components. `graph_export.py`
uses static `torch.export`/FX partitions bounded by weight bytes (default 256 MiB),
binds parameters into graphs, and reuses component graphs across timesteps.
Every shard is checked against its PyTorch output through desktop LiteRT CPU;
unknown ABIs, numerical mismatch, non-finite values, graphs ≥2 GB and unsupported
export operations fail explicitly. The weight bound is not a bound on total
runtime activation memory, and a single large tensor can exceed it. These are
experimental conversion recipes, not prevalidated conversion claims.

`compile` invokes the actual **LiteRT Python AOT command API**
`ai_edge_litert.aot.aot_compile.aot_compile`, not the LLM-only assumptions of an
unverified shell command. It selects Google's Tensor target, exports each compiled
model, saves compiler reports and partition counts, and updates artifact hashes.
It refuses a TPU bundle if no denoiser operations offload. Default
`--truncation no_truncation` avoids silently reducing precision; if the SDK cannot
compile that, test another
[documented compiler precision](https://developers.google.com/edge/tensor-sdk/compilation-flags)
(e.g. `--truncation bfloat16`) and recheck image quality before claiming success.

To test reuse of the existing community FLUX graphs first:

```sh
python scripts/image_litert/workflow.py fetch FLUX_KLEIN --ready-flux --output artifacts/image-litert/flux-ready
python scripts/image_litert/workflow.py compile --source artifacts/image-litert/flux-ready --output artifacts/image-litert/flux-ready-g5 --soc Tensor_G5
```

That produces `compilation-only.json`, not an Android-importable manifest. It
intentionally does not mix 256px timings into the 512px benchmark.

## Android provisioning and behavior

```sh
python scripts/push-image-litert.py artifacts/image-litert/sd-g5 --serial DEVICE_SERIAL
```

This stages only manifest-listed files under
`/sdcard/Android/data/dev.gemmabench/files/image-litert/SD_TURBO/Tensor_G5/` and
publishes the manifest last. It preserves the installed app, GGUF files, and run
history. Open **Research Lab → Images → Verify staged TPU bundles**. The app checks
SoC, manifest dependencies, file sizes/hashes, compiler offload counts and Python
BPE tokenizer fixtures. Changes to a verified file invalidate its ready marker.

Once verified, select TPU in the word/category playground or the 30-word
benchmark. TPU compilation fixes the model's step count. The manual TPU call has
a 20-minute timeout; benchmarks use the selected per-image timeout. Both execute
in the isolated `:imagebenchmark` process, where Stop/timeout can terminate a
stalled driver call.

Execution uses `CompiledModel` with NPU options for offloaded graphs; a failed NPU
graph is never retried as CPU. Graphs with zero offloaded operations explicitly
run on CPU, and unsupported partitions can remain on CPU inside compiled graphs.
Every result therefore says **TPU + CPU host/partitions**, reports unique graph
op coverage, stores actual artifact SHA-256 and LiteRT version, and includes
loading/encoding/denoising/decoding in end-to-end latency. Operator counts are
compiler reports, not measured percentages of runtime or FLOPs. Any claimed TPU
speedup requires the actual benchmark and image review.

The benchmark pins the TPU bundle manifest hash when planning a run and refuses resume/invocation after it changes. It schedules only installed model/method combinations. On TPU
failure, the current sample fails and the affected model's remaining TPU samples
are skipped. CPU and other models can continue. Recent images, statistics,
ratings, exports and historical runs use the existing persistence pipeline.

## Publishing later

Review the source license, model card, compiler reports and device outputs first.
SD/SDXL licensing is not changed by conversion. The script includes original
notices and creates a derivative-model card marked with the real validation state.

```sh
# Prepare README only; no upload or repository creation.
python scripts/image_litert/workflow.py publish --source artifacts/image-litert/sd-g5
# After choosing your account/repository and logging in locally:
hf auth login
python scripts/image_litert/workflow.py publish --source artifacts/image-litert/sd-g5 --repo YOUR_ACCOUNT/sd-turbo-litert-tensor-g5 --upload
```

Explicit `--upload` creates a private repository by default and uploads only
reviewed bundle artifacts/reports. Do not relabel `not-device-validated` until
actual Pixel execution and quality checks establish a stronger claim. No upload
was performed during this implementation.

## Validation status

- Android build, lint and 80 JVM unit tests passed, including all three committed tokenizer fixture sets.
- Six Python FX planner/sampler/publishing helper tests passed locally without model
  weights or the Tensor SDK; they do not prove LiteRT conversion or NPU execution.
- Python-generated tokenizer fixtures were compared with Android's Java/Kotlin BPE
  implementation, using actual tokenizer files from all three source checkpoints.
- On the connected Pixel 10 Pro Fold / Tensor G5, LiteRT 2.3.0 loaded and reported `[NPU, CPU]`; the read-only dispatch/missing-bundle instrumented smoke passed. This is not a diffusion invocation.
- Full source-weight export, Tensor compilation, Pixel invocation, speed/quality
  measurements and publication are pending the conversion workstation and SDK.

Machine-readable validation status: [IMAGE_TPU_VALIDATION.json](IMAGE_TPU_VALIDATION.json).
