# Images: 30-word speed and quality benchmark

Open **Settings → Research Lab → Images**, then scroll to **30-word generation benchmark** at the end of the tab.

Select one or more installed models and the available CPU/GPU methods, then press **Run 30-word benchmark** once. Each selected model/method generates all **30 fixed vocabulary pictures**. For example, one model with CPU and GPU schedules 60 images; three models with both methods schedule 180. No model downloads start automatically. This can take hours with the current models.

## Inputs and comparison protocol

`ImageBenchmarkSuite` defines `word-concepts-30-v1`: 30 hand-crafted prompts covering fruit, animals, everyday objects, actions, feelings, colors and shapes. Each word has a deterministic seed (42–71). The complete prompt, category, seed, model, step count, hardware method and protocol are copied into each run manifest. History/resume never regenerates prompts from a newer suite.

All images are 512×512. Each model uses its default step count (SD-Turbo 1; SDXL-Turbo and FLUX Klein 4). Hardware comparisons within a model use identical prompts/seeds/settings. Model/method order reverses on alternating words to reduce a fixed CPU-first warming bias. These are end-to-end workflow measurements, with a fresh process and engine per image, one second between samples, and no excluded warm-up. They are not steady-state retained-engine throughput measurements, nor thermally controlled experiments.

The primary latency includes worker startup, model loading, text encoding, denoising, VAE decoding and PNG saving. Each result additionally retains the image generator's total/native/phase times, component hashes, dimensions, seed, runtime identity and process high-water RSS. A GPU's additional allocations are not established by the RSS value alone.

Statistics are grouped by **model + method**, using successful samples only:

- Average (arithmetic mean), median, minimum and maximum end-to-end latency.
- Successful/attempted sample counts, with failures and skips reported separately.
- Mean quality rating and number of rated images. Unrated/failed/skipped samples do not contribute a zero.

Average and mean are the same statistic. Median is provided to show the center without as much sensitivity to outliers. A partial run's statistics are provisional and may cover different subsets of words between methods; compare complete runs for the fairest result.

## Hardware methods

- **CPU:** existing ARM CPU pipeline, up to four threads, with explicit CPU placement.
- **GPU:** compiled `stable-diffusion.cpp` Vulkan backend. Denoising requests Vulkan; the text encoder and VAE decoder stay on CPU. This is explicitly a hybrid GPU pipeline: full GPU decoding failed to complete within a bounded Pixel trial, so the decoder is kept on the established CPU path. Device availability requires advertised Vulkan 1.1 support and a Vulkan-enabled build. Availability does not establish model/driver compatibility: actual initialization and generation happen in the worker. Backend operation support can still result in mixed execution inside the runtime; this is a GPU-configured pipeline, not a claim that every operation runs on GPU. The run retains the configured placement, and no failed GPU job is relabeled as CPU.
- **TPU:** shown as unavailable, with a reason. The installed GGUF models are not Tensor-compiled LiteRT diffusion artifacts, and there is no diffusion TPU backend in this application. Existing Gemma TPU support does not apply to these image models. Unavailable method reasons are saved with each run.

A GPU failure marks the current job failed and skips remaining GPU jobs in that run, avoiding 30 repeated driver failures. CPU jobs can continue. A native CPU worker crash skips remaining jobs for that model/CPU combination. Other ordinary CPU errors are recorded per image.

Native generation runs in a non-exported `:imagebenchmark` service process. Driver crashes, aborts, timeouts and Stop terminate that worker without destroying the main UI or run manifests. The per-image timeout is selectable (5/10/20 minutes); 20 is the default to accommodate slow FLUX CPU generation and cold GPU compilation. No cloud fallback is used.

## Quality review and history

The most recent run displays its statistics and a word-by-word comparison gallery. Every image shows the model, CPU/GPU method, status, prompt, seed, steps, end-to-end and phase timings, and actual configured backend. Save PNGs or export a run's JSON.

Quality is **human reviewed**, not an unvalidated automatic accuracy score:

- 1: wrong or unrecognizable concept.
- 3: recognizable concept with visible problems.
- 5: accurate, clear teaching picture.

Consider concept accuracy, subject clarity, anatomy, clutter and unwanted writing. Rate each image 1–5 after the run stops; ratings can be changed or cleared. These persist and update the per-model/method quality averages. This feature does not calculate FID, CLIP scores, or a statistically validated educational-accuracy metric.

Each sample is checkpointed with Android `AtomicFile` into `files/image-benchmarks/<run-id>/run.json`; PNGs and their generation records stay in that run's `images/` folder. Historical runs are available after app/process restart, with pagination and a detailed gallery for every word. Historical results are independent of the manual image playground's 24-result list.

Keep Research Lab foreground while a benchmark runs. Stop or leaving the activity cancels the active native worker, preserves completed samples and leaves the interrupted word pending. **Resume pending** continues that same run without regenerating successful images. A process exit recovers the run as **Interrupted** on next startup. Resume checks that model bundles/methods remain available and that the device build fingerprint has not changed. Start a new run after a firmware change to keep hardware comparisons consistent.

## Build

Vulkan is enabled by default. The first CMake configuration fetches pinned header dependencies (Vulkan-Headers `19725e4d48082fe78e26622b15d3080ccd54112b`; SPIRV-Headers `01e0577914a75a2569c846778c2f93aa8e6feddd`) into the ignored native build tree. The SDK's NDK glslc compiles shaders and a host tool assembles them; no host Vulkan driver/runtime is required for Android cross-compilation. The existing Android NDK/CMake requirements remain in README.

```sh
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest
```

Host tests execute the complete 30-word CPU/GPU schedule through a fake generator to validate ordering and statistics, and check failed methods and cancellation/resume. This is scheduler validation, not synthetic phone throughput.

The instrumented tests check atomic persistence/interruption recovery and ratings. A separate, explicitly opted-in test runs one bounded real native sample in the isolated worker:

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w \
  -e class dev.gemmabench.ImageBenchmarkInstrumentedTest \
  -e imageBenchmarkMethod GPU \
  dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

Use `CPU` to verify the CPU worker route. A contained GPU failure is reported in `ImageBenchmarkSmoke` logcat and establishes crash containment, not GPU throughput. The full hours-long suite is started by the user's benchmark button; it is not automatically run while installing the app.

## Validation on 2026-10-23

All 74 host tests, Android lint and debug/test APK builds pass. The host runner tests exercise the complete 60-job CPU/GPU schedule with a fake generator (30 words × 2 methods), plus statistics, failures and cancel/resume. Device checkpoint/rating persistence and cancellation-before/during-bind tests pass. Emulator screen tests pass for controls, statistics, comparison gallery and opening historical runs; their screenshots are explicitly labeled synthetic UI fixtures.

On Pixel 10 Pro Fold / Tensor G5, a real SD-Turbo hybrid GPU sample generated a recognizable 512×512 apple in **118.225 s end-to-end / 117.475 s generation**. Its phases were **14.757 s loading/encoding, 42.488 s denoising, 60.233 s decoding/saving**. The GPU denoiser completed; full-GPU VAE trials did not finish within 3- and 5-minute limits, so the deployed GPU method keeps decoding on CPU. This sample demonstrates execution, not a general GPU speedup. SDXL/FLUX GPU execution has not been established by this SD-Turbo smoke test.

The real application controller also started the 30-word CPU plan and stopped after its first image: **71.109 s end-to-end / 70.550 s generation**, with one successful image and 29 jobs pending. That labeled **VALIDATION SAMPLE** run, its image and its resumable pending jobs remain on the phone. The full hours-long suite was not run during development. Exact evidence is recorded in [IMAGE_BENCHMARK_VALIDATION.json](IMAGE_BENCHMARK_VALIDATION.json).

## Experimental LiteRT Tensor TPU workflow

See [IMAGE_TPU.md](IMAGE_TPU.md) for the separate compiled-bundle backend, Hugging Face download/export/AOT/publishing scripts, required Linux/SDK setup, and current validation limits. No diffusion TPU performance claim has been established.
