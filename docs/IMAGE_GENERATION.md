# On-device word illustrations — research and prototype

Research date: **2026-10-23**. Open **Settings → Research Lab → Images**.

## Finding

Yes: open text-to-image models have a portable Android execution path. This prototype integrates **stable-diffusion.cpp through JNI**, with ARM CPU inference, rather than assuming that a Snapdragon/Qualcomm NPU package can run on Google's Tensor TPU. Pixel throughput and quality need device measurements; architecture support is not a performance guarantee.

Google's specifications list **Tensor G5 and 16 GB RAM for Pixel 10 Pro**, and **Tensor G6 and 12 GB RAM for base Pixel 11**. Total system RAM is not all available to an app. Pixel 11 therefore does not automatically provide more model memory than Pixel 10 Pro. Check the actual variant before comparing results.

- [Pixel 10 Pro official specifications](https://store.google.com/product/pixel_10_pro_specs)
- [Pixel 11 official specifications](https://store.google.com/product/pixel_11_specs)
- [Runtime supported architectures, Android, and backends](https://github.com/leejet/stable-diffusion.cpp)

## Three selected models

| Model | Prototype artifact download | Configuration | Quality / practicality assessment |
| --- | ---: | --- | --- |
| SD-Turbo | 2.02 GB | Combined Q8/F16 GGUF, 512×512, Euler, SGM uniform, 1 step default; 2/4 optional | Smallest baseline. Good starting point for a single concrete noun. Older SD 2.1 lineage, so do not expect FLUX-level composition or understanding. |
| SDXL-Turbo | 4.10 GB | Combined Q8 GGUF, 512×512, Euler, SGM uniform, 4 steps default; 1/2 optional | Higher-detail baseline. Its publisher recommends it over SD-Turbo for quality/prompt understanding. More memory and compute. |
| FLUX.2 Klein 4B | 5.29 GB total | Q4_0 diffusion transformer + Qwen3-4B Q4_K_M encoder + full FLUX.2 VAE, 512×512, Euler, FLUX2 scheduler, 4 steps | Actual FLUX-family model and best candidate for the requested quality direction. Quantization, reduced resolution, and phone execution still need evaluation. |

Sizes are download sizes, **not peak RAM estimates**. The FLUX transformer alone is 2.46 GB; its text encoder is 2.50 GB and VAE 0.34 GB. Omitting the encoder from a phone memory/storage estimate would be misleading.

Primary model descriptions and inference settings:

- [SD-Turbo model card](https://huggingface.co/stabilityai/sd-turbo): distilled SD 2.1; 1–4 evaluations, preferred 512 px, no negative prompt or CFG. The app's native `txt_cfg=1` disables CFG, corresponding to Diffusers `guidance_scale=0`.
- [SDXL-Turbo model card](https://huggingface.co/stabilityai/sdxl-turbo): SDXL distillation; 1–4 steps, 512 px preferred. Its examples and study support better quality with four steps. Text and faces remain limitations.
- [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): 4B rectified-flow model, Apache 2.0. Its ~13 GB VRAM / subsecond claims describe consumer GPU execution; they are **not Pixel benchmarks**.
- [Runtime FLUX.2 integration guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md): separate diffusion weights, Qwen3 encoder, and FLUX.2 VAE; distilled Klein uses four steps and CFG 1.

Artifact conversion sources are [Green-Sky SD-Turbo](https://huggingface.co/Green-Sky/SD-Turbo-GGUF), [OlegSkutte SDXL-Turbo](https://huggingface.co/OlegSkutte/sdxl-turbo-GGUF), [runtime author's Klein conversion](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF), [Unsloth Qwen3 GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF), and [Comfy-Org Klein VAE](https://huggingface.co/Comfy-Org/flux2-klein-4B). The catalog pins revision, byte length, and publisher LFS SHA-256 for each file. Quantization can affect quality; these are not the original full-precision artifacts.

SD/SDXL Turbo retain their source licensing conditions; their cards direct commercial users to Stability's license terms. FLUX.2 Klein 4B and Qwen3 use Apache 2.0; the decoder follows its source model license. Downloads are explicit and source/license links are available in the screen.

## Other paths investigated

[Google MobileDiffusion](https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/) reports 520M parameters and ~0.5-second 512×512 generation on premium Android/iOS hardware. This research did not establish a public reusable weight checkpoint that we can integrate. A paper/demo alone is not an installable model.

[Qualcomm's Stable Diffusion demonstration](https://www.qualcomm.com/news/onq/2023/02/worlds-first-on-device-demonstration-of-stable-diffusion-on-android) optimizes SD 1.5 for Snapdragon 8 Gen 2. Qualcomm QNN/Hexagon artifacts cannot be treated as Tensor TPU artifacts. Similarly, a model being supported by LiteRT-LM for text does not make that runtime a diffusion pipeline.

FLUX.1 schnell has a larger 12B denoiser plus CLIP/T5 encoders. Klein 4B is the more reasonable FLUX experiment here. No cloud inference fallback or API key is used.

## Prototype behavior

1. Download a model. Android DownloadManager persists transfers across process exit, reports progress/wait/error, supports cancellation/retry, and defaults to Wi-Fi only. FLUX downloads all three components. Inference waits for active provisioning to finish.
2. Enter **word** and **category**, or choose an example. Both are required, limited to 80 characters. The concept leads the prompt so the short CLIP context sees it first. Actions depict a character doing the action; emotions use expression/body language; colors and shapes get concrete visual treatments. The preview is editable and can be rebuilt. A generic template is not semantic disambiguation: “bat / Sports” still needs review.
3. Select steps and a reproducible nonnegative seed. FLUX uses four steps. Generate a real 512×512 image with JNI. The Gemma engine is unloaded first, and all model operations use the existing serial lab worker. Four CPU threads maximum, optimized C/C++ kernels, memory-mapped weights, flash attention, and tiled VAE decoding are enabled. The CPU kernels target ARMv8.2 with FP16 and dot-product instructions; the JNI entry point rejects older CPUs before running them. No GPU/TPU speed is implied.
4. Keep the Research Lab foreground. Screen stays awake during inference; Stop/background requests native cancellation. Cancellation during model loading is observed when loading returns. Native OOM/assertion/driver/process failures may still terminate the process; Kotlin cannot recover from those.
5. Review the saved PNG and prompt, model, seed, steps, elapsed times, process high-water RSS and device identity. Results survive process restarts. Save individual PNGs or export JSON measurements, including component hashes and runtime commit.

All inputs, outputs and inference remain local after provisioning. This lab prototype has no built-in image safety classifier; review images before displaying them in the children's word browser. Generated images are not automatically added to that browser.

## Build and validation

```sh
git submodule update --init third_party/stable-diffusion.cpp
git -C third_party/stable-diffusion.cpp submodule update --init ggml
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest
```

The runtime is pinned at `95ff76230889898e1bf40058ba272a3699b50dc6`, with its pinned ggml submodule. Android NDK 27.0.11902837 and CMake 3.22.1 build the ARM64 JNI library. Video/server/WebP/WebM examples are disabled. The manual word playground uses CPU; the [30-word benchmark](IMAGE_BENCHMARK.md) additionally builds and exercises a Vulkan GPU method in an isolated worker process. No model weights or generated binaries are committed.

The real-device opt-in test verifies staged model SHA-256, creates an actual image, checks dimensions and nonblank pixels, and persists timing/metadata. Pixel quality evaluation is a manual visual check; nonblank pixels do not establish concept accuracy.

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
# Stage the catalog's files (unmodified) under the app's external files/image-models directory.
adb shell am instrument -w \
  -e class dev.gemmabench.ImageGenerationInstrumentedTest \
  -e imageInference true -e imageModel SD_TURBO \
  dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

Use `SDXL_TURBO` or `FLUX_KLEIN` for the other model slots. Use `scripts/push-image-model.sh FILE...` to stage the original catalog files. The test verifies staged files before marking them ready. It does not uninstall the app or remove other downloaded models.

Times include loading on each invocation. `nativeMs` includes load, prompt encoding, denoising and image decode; `totalMs` also includes PNG creation. New results also include loading/encoding, denoising, and decoding/saving phase times, measured at native denoising progress boundaries. `peakRssKb` is `/proc/self/status` VmHWM for the **process lifetime**, not an isolated per-model peak; don't rank models by that field across multiple runs in one process. Seed equality permits reproducibility within the same model/settings/runtime but does not imply identical latents across model architectures.

## Pixel validation

All three models generated real, nonblank 512×512 images on the connected **Pixel 10 Pro Fold / Tensor G5** with the same apple/Fruit prompt and seed 42. Each model used a fresh process; these are single CPU smoke runs, not repeated thermal-controlled benchmarks. USB model staging overlapped part of the SD/SDXL tests. Pixel 10 Pro slab and Pixel 11 were not tested.

| Model | Steps | Total | Load + encode | Denoise | Decode + save | Process peak RSS | Visual review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SD-Turbo | 1 | 69.754 s | — | ~7.35 s* | — | 2.42 GiB | Recognizable botanical apple illustration; extra fruit/leaves. |
| SDXL-Turbo | 4 | 143.776 s | 3.435 s | 51.727 s | 88.618 s | 4.27 GiB | Recognizable apples, but multiple subjects and stray text-like glyphs. |
| FLUX.2 Klein 4B | 4 | 546.823 s | 21.378 s | 437.938 s | 87.510 s | 4.81 GiB | Clearest teaching picture in this trial: one red apple, simple background, no stray text. |

*SD-Turbo's denoiser time comes from native progress log timestamps; phase fields were added after that run. Total/native timing and RSS were captured in its saved record. Full VAE decoding is a major cost even for the one-step baseline. FLUX's denoiser dominates its much longer total.

**Practical conclusion:** quantized FLUX Klein fits and generates a good concept illustration on this Tensor G5 phone, but ~9 minutes on CPU is unsuitable for an immediate interactive picture response. SD-Turbo is the faster starting point for this prototype, still ~70 seconds. SDXL's broader publisher-reported quality advantage did not produce the best flashcard in this one example. GPU/TPU acceleration needs its own implementation and hardware validation before promising responsive generation; none is implied here. The 12 GB Pixel 11 requires separate memory/performance validation.

Exact prompts, component hashes and measurements are in [IMAGE_GENERATION_VALIDATION.json](IMAGE_GENERATION_VALIDATION.json). PNGs and UI screenshots from this workspace's validation are under ignored `artifacts/image-validation/`; the generated PNGs also remain in the app's local results.

Validation completed:

- All **68 host unit tests**, Android lint, debug APK and test APK builds pass.
- Real Pixel image/dimension/nonblank tests pass for **all three models**; their PNGs were visually reviewed.
- Native cancellation-before-load test passes.
- Emulator download lifecycle test passes: real bytes transferred, persistent ticket recovered by a new store instance, and partial file removed on cancellation.
- ARM emulator navigation and Images layout were visually reviewed. Choosing **jump / Actions** produced an action prompt; typing **swim** regenerated the prompt while preserving the category.

The final prototype is installed on the connected phone with all three model bundles verified and the three sample results retained. The phone was locked during final UI review, which used the emulator; model timings above are from the physical Pixel.

## Experimental LiteRT Tensor TPU workflow

See [IMAGE_TPU.md](IMAGE_TPU.md) for the separate compiled-bundle backend, Hugging Face download/export/AOT/publishing scripts, required Linux/SDK setup, and current validation limits. No diffusion TPU performance claim has been established.
