# Gemma Bench

A native Android starting point for **Gemma 4 E2B and E4B through LiteRT-LM**. It includes one-tap model downloads, a fixed-token benchmark, a streaming prompt playground, and local JSON/CSV results. Network access provisions models; inference runs offline.

The lab is the initial app surface. Its inference adapter, benchmark runner and UI are separate so this feature can move behind internal developer settings later.

## Build and install

Requirements: JDK 17+, Android SDK platform 37/build tools 36, and an **ARM64 physical Android phone running Android 12/API 31 or later**. A recent Pixel with current firmware is recommended. Hardware inference performance must be measured on a real phone.

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n dev.gemmabench/.MainActivity
```

Set `ANDROID_HOME` to your SDK, or create an ignored `local.properties` with `sdk.dir=/absolute/path/to/android-sdk` if needed. Versions are pinned: Gradle 9.5.0, AGP 9.3.1, Compose compiler 2.4.10 and LiteRT-LM 0.18.0.

## Download the models

Open **Models** and tap **Download E2B** or **Download E4B**. Downloads come directly from the official LiteRT Community repositories on Hugging Face. No browser, file picker, Hugging Face login or access token is needed for these public models.

- E2B: **2,588,147,712 bytes** (2.59 GB).
- E4B: **3,659,530,240 bytes** (3.66 GB).
- **Wi-Fi only** is on by default. Switching it changes the policy for newly started downloads.
- Each card shows bytes, percentage, network waiting state, and cancellation/retry actions.
- Android's download service keeps transfers running outside the app and retries interrupted connections. Tracked jobs and progress are recovered after app/process restart. User cancellation removes the partial file; a retry starts a fresh transfer.
- Files are checked against pinned size/SHA-256 values and inspected by LiteRT-LM before becoming ready. Verification completes while the app is running, or on the next launch after a process exit.
- Downloaded files are promoted in place inside app-owned storage, without a second multi-GB import copy. Keep 128 MiB of spare space beyond each artifact's size.

Both models can be queued. Inference/import operations wait until active downloads and verification finish, keeping benchmark measurements separate from provisioning work.

The standard `.litertlm` files match Gallery's E2B/E4B catalog entries. See [download implementation notes and inspected Gallery sources](docs/MODEL_DOWNLOADS.md).

## Advanced: import an existing model

Under **Models → Advanced**, choose **Import file** for an existing standard `.litertlm` artifact (not `.task`, GGUF or SafeTensors):

- [Official E2B files](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)
- [Official E4B files](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/tree/main)

Models are several GB. Imports make a local copy and calculate SHA-256. The app preserves the previous model until the copy finishes. Model family/variant labels are assigned by the selected slot; verify the source artifact and exported hash. The app does not infer model identity from filenames.

The importer reads native model metadata before accepting the file and shows supported backends/context. Standard artifacts are `gemma-4-E2B-it.litertlm` and `gemma-4-E4B-it.litertlm`; the repositories also include `-gpu.litertlm` variants. If a file declares a different backend than the one selected, import the compatible artifact instead. Exports record its hash and metadata so these variants are distinguishable.

Alternatively, after launching the app once:

```sh
./scripts/push-model.sh e2b /absolute/path/to/gemma-4-E2B-it.litertlm
./scripts/push-model.sh e4b /absolute/path/to/gemma-4-E4B-it.litertlm
```

Then tap **Import files staged by ADB**. Staging files are retained until you remove them; staging + the private copy require additional free space. Do not replace an artifact while a run is active.

## Run

1. In **Benchmark**, select CPU or GPU and one or both downloaded models.
2. Use 256/128 for a quick check or 1024/256 for the reference workload. Default: one excluded warm-up and three measured runs, with two seconds between runs.
3. Keep the app foreground. It keeps the screen awake while running. Going to the background requests a stop.
4. Review medians, p95, native token counts and per-run details. **Results** exports all saved results as JSON or CSV, including prompts/outputs and device/build metadata.
5. In **Playground**, try explanation, JSON extraction, summarization, or an editable prompt. Run one model repeatedly to observe retained-engine behavior. **Unload engine** releases it.

Native backend failures are reported, never silently rerouted. A failed model does not prevent the next selected model from being attempted. Fixed-token cancellation takes effect after the current native invocation returns; prompt-mode cancellation requests native stream cancellation. A native driver crash cannot be converted into a recoverable Kotlin error.

## Documentation and checks

- [Initial research](docs/RESEARCH_0.md)
- [Benchmark methodology, metrics and TPU setup](docs/BENCHMARKING.md)

```sh
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

These checks validate build, benchmark orchestration/statistics/export behavior and Android lint. They do not establish phone throughput or verify GPU/TPU drivers. See [validation status](docs/VALIDATION.md).
