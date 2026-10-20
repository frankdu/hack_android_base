# Benchmark methodology

## Two workloads

### Fixed-token benchmark

Uses the pinned LiteRT-LM **0.18.0 native `benchmark()` helper**. This is the reproducible throughput path. The prompt's last text input is tokenized, padded with zero tokens if shorter than the requested prefill, or truncated if longer. Output is a synthetic fixed-length workload, not a text quality evaluation.

Every invocation creates and destroys an engine/conversation. Native `initTimeInSecond` measures initialization. Wall time wraps the entire invocation including initialization and teardown. Native TTFT/prefill/decode measurements are supplied by the runtime and should not be derived by dividing the full wall duration.

Warm-up invocations prime runtime/OS/compiler caches but do **not** retain the engine. They remain in raw exports with `warmup=true` and are excluded from summaries. Runtime caches remain on disk between runs and app restarts. Consequently, the first run is **process-first/cache-dependent**, not a guaranteed cold-filesystem test. We do not clear OS page caches or characterize storage cold-start latency.

The native helper controls context sizing and CPU thread count. The exported `contextTokens` setting is only applied by prompt mode; it is not forwarded to the helper. No CPU thread count is claimed. Baseline MTP is explicitly false. The harness uses its native sampling/thinking defaults, unlike the separately configured prompt mode.

Defaults: 256 input, 256 output tokens, 1 warm-up, 3 measured invocations, 2-second cooldown. Presets: quick 256/128, reference 1024/256. CPU, GPU and TPU are explicit, with no fallback. Three samples are exploratory; p95 with this sample size is the largest observation, not a reliable tail-latency estimate.

### Prompt playground

Uses `Engine` and streaming `Conversation.sendMessageAsync()`, with `ExperimentalFlags.enableBenchmark=true` and `Conversation.getBenchmarkInfo()` after completion. Exact native input/output counts include the runtime's tokenization/template behavior; they are not inferred from characters or callback count.

- Engine is retained when model SHA-256, backend, context and MTP configuration match.
- Every request uses a **fresh conversation**, so there is no accumulated chat history or prefix reuse.
- `engineLoadMs` is measured around `Engine.initialize()`. It is null for reused engines; `engineReused=true` distinguishes reuse from missing data.
- `conversationSetupMs` is measured separately around conversation creation.
- `elapsedMs` wraps send/stream completion and excludes engine load and conversation setup.
- `firstVisibleTextMs` measures first non-empty visible response text at the application boundary. It is not interchangeable with native TTFT; channel markers/thinking can affect them differently.
- Thinking disabled with a zero thinking budget; top-k 1, top-p 1, temperature 0, seed 0.
- Output token limit is a maximum, not a forced count. Early EOS may return fewer tokens.
- Context is configurable, default 2048 tokens. A large actual prompt can exceed context and fail; fixed-mode input length does not constrain an editable prompt.
- Only text is configured. Camera/audio integration is future product work; encoders are not loaded for this baseline.

Select both models to run the same prompt sequentially. To inspect retained-engine behavior, select one and run it twice. Changing models necessarily releases the preceding engine.

## Interpretation and exports

`NativeMetrics` records native initialization, TTFT, actual prefill/decode token counts and rates. Non-positive/non-finite native time/rate values become null, shown as an em dash; missing metrics are never fabricated. Summaries omit missing metrics and warm-ups, and use the median and nearest-rank p95.

Each raw sample also records wall time and before/after environment snapshots:

- Android thermal status (0 none, 1 light, 2 moderate, 3 severe, 4 critical, 5 emergency, 6 shutdown).
- Battery percentage, plugged state and **battery** temperature, not GPU/SoC temperature.
- Process PSS and available system RAM. PSS is sampled at boundaries; **it is not peak RAM**, and may undercount accelerator allocations.
- Device manufacturer/model, SoC, RAM, OS/API/build fingerprint, ABIs, app/runtime versions.
- Assigned model slot, size and SHA-256 computed while importing. The hash identifies the imported bytes but does not verify authenticity against a publisher checksum.
- Native artifact metadata: supported text backends, context limit/type, minimum runtime and drafter availability. Import rejects non-generative artifacts; declared backend/context/MTP incompatibilities are surfaced before inference.

JSON schema version 1 preserves full nested results. CSV flattens one row per sample and includes a row for failures with no completed samples. Prompts/output are quoted; spreadsheet formula prefixes are escaped. Failed/stopped runs preserve completed samples and status. All results are saved locally after each model finishes, with a history limit of 100 result groups. A process/native crash during a model's batch can lose that in-progress batch. Export periodically during driver experiments.

Compare only equivalent artifacts, backends, workload lengths, MTP settings and similar environment conditions. Record charging state, let a hot phone cool, and avoid comparing cold initialization against warm TTFT. Network requests provision the model files; no external model calls occur during inference. See [model downloads](MODEL_DOWNLOADS.md).

## Tensor TPU support

The default APK includes LiteRT-LM's ARM64 CPU/GPU runtime plus Google's Maven `com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.0` package. That package supplies the Tensor dispatcher; the vendor TPU implementation comes from the phone's OS. The manifest declares the optional `libedgetpu_util.so` and `libedgetpu_litert.so` libraries so CPU/GPU devices can still install the app.

At startup, the serialized worker checks the exact normalized SoC name (Tensor G5/G6), ARM64 support, dispatcher presence, and whether the vendor libraries and dispatcher can load. Only the matching E2B TPU download is enabled. An unsupported device/runtime shows its reason and disables TPU controls. A successful probe establishes prerequisites; engine initialization remains the final model/firmware compatibility check. Native failures are surfaced without retrying on CPU/GPU.

1. In **Models**, download the matching **E2B TPU · G5/G6** artifact. The usual Wi-Fi policy, background transfer recovery, size/SHA-256 checks and native metadata inspection apply.
2. In **Benchmark** or **Playground**, select **TPU**, then the installed compatible model. The other Tensor generation is excluded. E4B TPU has no published official artifact.
3. In **Photos**, select E2B in the first row and TPU in the second. The option requires an installed compatible artifact with a vision encoder. Text and vision backends are recorded separately; vision honors artifact constraints and can use NPU through `Backend.NPU`.

For custom runtime experiments, obtain compatible ARM64 libraries from [official LiteRT releases](https://github.com/google-ai-edge/LiteRT/releases) or build them using the [Tensor FAQ](https://developers.google.com/edge/tensor-sdk/tensor-faqs). Match SDK/runtime/OS using the [compatibility matrix](https://developers.google.com/edge/tensor-sdk/release-notes). Place them under a directory containing `arm64-v8a/` and build with `-PtensorNpuLibDir=/absolute/path/to/jniLibs`. This replaces the Maven Tensor dependency to avoid duplicate libraries. Do not replace LiteRT-LM's JNI library with an unrelated version.

The [official model list](https://developers.google.com/edge/tensor-sdk/slm-compilation) and [Pixel TPU reference sample](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/google/sample_app_tpu/README.md) describe the supported bundles. Do not infer TPU performance from CPU/GPU results.

## Architecture

- `runtime/LiteRtRuntime.kt`: the reusable LiteRT-LM adapter and retained-engine lifecycle.
- `benchmark/`: data, validation, statistics and sequential warm-up/run orchestration.
- `data/`: atomic model/result persistence, model copying/hash, environment snapshots and export.
- `LabController`: application-owned serialized worker and observable state; survives activity rotation. Operations do not close a native handle while inference is in flight.
- `ui/LabApp.kt`: standalone Models/Benchmark/Playground/Results UI. Later replace the launcher surface and route this composable from internal settings.

Only one native engine operation runs at a time, because LiteRT-LM experimental flags are process-global. Fixed-token stop is cooperative between invocations, and prompt stop requests native conversation cancellation. Keep the app foreground; backgrounding requests stop rather than launching a benchmark service.
