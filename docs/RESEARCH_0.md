# Gemma on Pixel: initial Android hackathon research

Research date: **October 19, 2026**. This is an online research snapshot, not a report of measurements taken in this repository. Official support, firsthand reports, and recommendations are distinguished below.

## Recommendation and event context

**Gemma 4 E2B + LiteRT-LM is the strongest starting point for our Android prototype.** Both Pixel 10 and Pixel 11 have official TPU-compatible E2B downloads. E4B is a quality upgrade to evaluate, while FunctionGemma and EmbeddingGemma support fast commands and private local search.

The [hackathon page](https://www.agihouse.org/events/the-google-gemma-hardware-hackathon) requires a Gemma-based model and entirely on-device execution, with no cloud in the loop. It has a Mobile AI track, an Android/Pixel experience award, and LiteRT awards. A native app using the camera, microphone, sensors, or local actions is a good fit. The event is October 24, 2026, in Hillsborough. Prize descriptions were changing at research time; consult the live page for final details.

The event page is a client-rendered app. Its public event description was retrieved from the public endpoint used by that app: `https://jtwthn6xog.execute-api.us-east-1.amazonaws.com/events/slug/the-google-gemma-hardware-hackathon`. No account was used. Private attendee information and incidental event access details are not included here.

## 1. Which Gemma models can run?

| Model | Pixel 10 / Pixel 11 deployment status | Capabilities | Proposed use |
|---|---|---|---|
| **Gemma 4 E2B IT** | Official Tensor G5 and G6 TPU bundles; CPU/GPU variants available | Text, images, audio, visual/video understanding; text output; reasoning and function calling | Default prototype model |
| **Gemma 4 E4B IT** | Android LiteRT-LM CPU/GPU support; firsthand reports on both Pixel generations. No E4B TPU bundle in the current Tensor SDK list | Similar modalities, stronger reasoning | Quality comparison with E2B |
| **Gemma 3n E2B / E4B** | Mobile-oriented LiteRT-LM models; CPU/GPU deployment path. Exact device/backend combination still needs testing | Text, image, audio and video input → text | Existing projects or task-specific comparisons |
| **Gemma 3 1B IT** | Official G5/G6 TPU bundles | Text input/output | Lightweight extraction, classification and simple assistance |
| **FunctionGemma 270M** | Official G5/G6 Mobile Actions bundles; custom G6 compilation supported | Text → structured function calls | Small, fast command vocabulary |
| **EmbeddingGemma v1 / v2** | Official G5/G6 bundles | Retrieval embeddings; v2 includes text, image, video and audio variants | Local search, memory and RAG |

Sources: [Tensor SDK model list](https://developers.google.com/edge/tensor-sdk/slm-compilation), [LiteRT-LM model catalog](https://developers.google.com/edge/litert-lm/overview).

Smaller Gemma 3 270M and older Gemma 2 2B models are additional options. Gemma 3 4B supports images and is a possible quantized-runtime candidate, but it is not the initial recommendation. Gemma 3 270M/1B are text-only; vision starts at 4B. [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3)

### Model size and memory

- E2B/E4B means **effective** parameters. Gemma 4 E2B has 2.3B effective parameters and 5.1B including embeddings; E4B has 4.5B effective and 8B including embeddings. [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- Standard LiteRT-LM files are approximately **2.58 GB for E2B** and **3.66 GB for E4B**. File size differs from active memory: embeddings can be memory-mapped and vision/audio encoders loaded on demand. [E2B package](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), [E4B package](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)
- Google's approximate mobile model-loading memory figures are **1.1 GB E2B** and **2.5 GB E4B**, before additional context/runtime requirements. These are planning figures, not total Android process memory. [Memory requirements](https://ai.google.dev/gemma/docs/core)
- Gemma 4 12B might be an experiment on higher-memory hardware, but is excluded from the initial phone shortlist. The 26B/31B models have much larger memory requirements. The 26B A4B model still stores all experts; its active parameter count does not imply a 4B memory footprint. [Memory requirements](https://ai.google.dev/gemma/docs/core)

### Phone configurations

Base Pixel 10 and Pixel 11 have 12 GB RAM. Pixel 11 Pro/XL 256 GB configurations have 12 GB, while 512 GB/1 TB configurations have 16 GB. Record the exact device configuration, OS and firmware for every benchmark. [Pixel 10 specs](https://store.google.com/product/pixel_10_specs?hl=en-US), [Pixel 11 specs](https://store.google.com/product/pixel_11_specs?hl=en-US), [Pixel 11 Pro specs](https://store.google.com/product/pixel_11_pro_specs?hl=en-US)

## 2. Deployment options

| Route | What is deployed | Advantages | Constraints |
|---|---|---|---|
| **LiteRT-LM in a native Android app** | Kotlin API, native runtime, `.litertlm` model | Direct model choice; streaming, multimodal inputs, tools; CPU/GPU/NPU configuration | App manages files, initialization, memory and backend compatibility |
| **AICore + ML Kit GenAI Prompt API** | App calls Android's system service | Shared system-managed model; less model distribution work; local inference | Model/device availability, quotas, foreground restrictions and preview enrollment |
| **llama.cpp with NDK/JNI** | Native runtime and GGUF weights | Flexible quantization and custom-model experiments | More native integration; acceleration and multimodal support need separate validation |

**Recommended initial route: LiteRT-LM.** Google's [Pixel TPU sample app](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/google/sample_app_tpu/README.md) demonstrates model loading, multimodal chat, streaming metrics, and TPU → GPU → CPU fallback handling. For benchmarks, explicitly select a backend and report failures; silent fallback would invalidate comparisons.

Suggested workflow:

1. Test the artifact in [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery).
2. Build using the [LiteRT-LM Android Kotlin API](https://developers.google.com/edge/litert-lm/android).
3. Sideload the APK/model during development; use app-private model storage for the app.
4. Load the correct artifact for the backend, initialize off the UI thread, and keep the engine warm for real interactions.
5. Verify the complete demo offline after initial model provisioning.

### Tensor TPU specifics

Use the G5 bundle on Pixel 10 and G6 bundle on Pixel 11. A CPU/GPU artifact is not automatically TPU-compatible. Google recommends precompiled models for G5. Custom compilation requires Tensor SDK Beta access. The documented G6 fine-tuned export path covers Gemma 3 1B, FunctionGemma and EmbeddingGemma v1. [Compilation guide](https://developers.google.com/edge/tensor-sdk/slm-compilation), [SDK release notes](https://developers.google.com/edge/tensor-sdk/release-notes)

The app/runtime also needs matching NPU dispatch libraries. Matching the `.litertlm` filename alone is insufficient. Use the SDK compatibility matrix for runtime, chipset and OS versions. Avoid assuming that old community runtime failures describe the current SDK.

### AICore specifics

Production models are exposed as **Gemini Nano**. Gemma 4 models form the basis for Nano 4; they are related deployment paths rather than interchangeable arbitrary weight-loading APIs. Current documentation lists Pixel 10 under Nano v3 and Pixel 11 under Nano v4. Preview availability must be checked at runtime. AICore enforces per-app quotas and permits inference only while the app is the top foreground application. A foreground service does not bypass this restriction. [ML Kit support/restrictions](https://developers.google.com/ml-kit/genai), [AICore preview](https://developers.google.com/ml-kit/genai/aicore-dev-preview)

These restrictions matter for background or always-listening ideas. The Pixel preview includes a development quota bypass, but it is not a production guarantee.

### Older tutorials and alternative runtimes

The MediaPipe LLM Inference API is now **maintenance-only**; Google recommends LiteRT-LM for Android projects. [Migration notice](https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference/android)

llama.cpp provides an Android Studio example and Android NDK build instructions for a GGUF alternative. [Official Android guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)

## 3. Capabilities and boundaries

| Capability | Android app opportunities | Boundary |
|---|---|---|
| Text understanding/generation | Summaries, extraction, classification, rewriting, translation and explanations | Validate accuracy on the actual task |
| Image understanding | Camera questions, scene descriptions, visual instructions, document/chart interpretation | Image encoding adds latency; test fine details and small text |
| Audio understanding | Transcription, spoken translation, questions/summaries about recordings | Documented maximum clip length: 30 seconds; mono, 16 kHz preprocessing |
| Video understanding | Reason about visual events and changes | Model support does not establish real-time phone camera throughput |
| Reasoning/thinking | Conditional decisions, planning and multi-step tasks | Extra thinking increases time before an actionable answer |
| Function calling | Select app actions and arguments | App executes actions and implements permissions/integrations |
| Embeddings | Search local notes, images, recordings and manuals | Requires a local index/retrieval pipeline |

Gemma 4 supports over 140 languages and native system prompts. Output is **text**, so spoken replies need a separate local text-to-speech component. [Model capabilities](https://ai.google.dev/gemma/docs/core/model_card_4)

Google documents transcription, translation, a 30-second audio clip limit and 16 kHz mono input preparation. Gemma 4 audio costs 25 tokens per second versus 6.25 for Gemma 3n. [Audio guide](https://ai.google.dev/gemma/docs/capabilities/audio)

Video is supported at the model level, but the [published tutorial](https://ai.google.dev/gemma/docs/capabilities/vision/video) is a Transformers example, not a phone camera benchmark. Sampling selected frames is an implementation option to validate, not an established frames-per-second claim.

Function calls could implement `start_timer(...)`, `save_observation(...)`, or app-specific commands. Gemma does not execute code itself; Kotlin code handles the action. [Function calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)

FunctionGemma is intended for task-specific fine-tuning rather than general chat. Google reports Mobile Actions accuracy improving from **58% to 85%** after fine-tuning, supporting a narrow command vocabulary. [FunctionGemma model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)

### Context length

Gemma 4 E2B/E4B have **128K architectural context**, but current mobile LiteRT-LM package cards advertise **up to 32K**. Specific TPU bundles can have smaller compiled caches. Inspect the deployed artifact rather than designing around the family maximum. [Architecture](https://ai.google.dev/gemma/docs/core/model_card_4), [Mobile package](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)

## 4. Published performance measurements

**Prefill** is input processing; **decode** is output generation; **TTFT** is time to first token. TTFT often excludes model loading. These results describe different workloads and are not a controlled CPU/TPU/GPU comparison.

### Pixel generation results

| Device | Model / backend | Prefill tokens/s | Decode tokens/s | TTFT | Evidence |
|---|---|---:|---:|---:|---|
| Pixel 10 Pro | Gemma 4 E2B / CPU | 165.97 | **10.42** | **1.65 s** | Publisher's own Gallery test |
| Pixel 11 Pro XL | Gemma 4 E2B / CPU | 271.48 | **24.67** | **0.98 s** | Same publisher's comparison |
| Pixel 10 / 10 Pro | Gemma 4 E2B / TPU, official quantized bundle | 91 | **14.7** | **0.30 s** | Developer runtime investigation |
| Pixel 11 Pro XL | Gemma 4 E4B / GPU | 277.99 | **9.60** | **1.03 s** | Firsthand Gallery benchmark |
| Pixel 10 Pro XL | Gemma 4 E4B / GPU | Not reported | **~11 estimated** | Not reported | Developer app report; estimated token counts |

Sources and qualifications:

1. [Beebom E2B CPU comparison](https://gadgets.beebom.com/guides/google-tensor-g6-vs-tensor-g5-benchmark-specs), updated September 30, 2026. App version, quantization and prompt details are incomplete. Its statement that third-party TPU access is unavailable conflicts with current official SDK documentation; use this source only for its reported CPU measurements.
2. [E2B TPU runtime investigation](https://github.com/bejoyfuuul/litert-lm-tensor-g5-npu). EdgeTPU runtime 0.14, with a performance optimization unavailable in that firmware. These measurements do not establish a current ceiling; prompt/token-count methodology is not the same as the official 1024/256 tests.
3. [Pixel 11 Pro XL E4B GPU firsthand benchmark](https://www.reddit.com/r/GooglePixel/comments/1w4ts4z/pixel_9_pro_xl_v_11_pro_xl_ai_edge_benchmark/). Gallery 1.0.18, 3 runs, 256 input and 256 output tokens. Separately reports **43.7 s first initialization** and **10.8 s steady initialization**. Community report, not an independently replicated result.
4. [Pixel 10 Pro XL E4B GPU developer report](https://www.reddit.com/r/LocalLLaMA/comments/1ww2hux/i_built_an_openaicompatible_server_that_runs/). Approximate throughput: token counts were estimated from characters. Reports ~12 s engine warm build and follow-up turns around 1.7 s with prefix reuse. Do not treat that follow-up figure as standardized TTFT.

No comprehensive official Gemma generation benchmark table covering both Pixel generations was found.

### Official comparison results on other Android phones

| Device | Model / backend | Prefill tokens/s | Decode tokens/s | TTFT |
|---|---|---:|---:|---:|
| Samsung S26 Ultra | Gemma 4 E2B / CPU | 557 | 46.9 | 1.8 s |
| Samsung S26 Ultra | Gemma 4 E2B / GPU | 3,808 | 52.1 | 0.3 s |
| Samsung S26 Ultra | Gemma 4 E4B / GPU | 1,293 | 22 | — |
| Samsung S24 Ultra | Gemma 3n E2B / GPU | 816 | 16 | — |
| Samsung S24 Ultra | Gemma 3n E4B / GPU | 548 | 9 | — |
| Samsung S25 Ultra | FunctionGemma Mobile Actions / CPU | 1,718 | 125.9 | 0.3 s |

Sources: [E2B benchmark/methodology](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), [LiteRT-LM benchmark catalog](https://developers.google.com/edge/litert-lm/overview), [FunctionGemma benchmark](https://ai.google.dev/gemma/docs/functiongemma/model_card).

The E2B tests use 1,024 input tokens, 256 output tokens, context 2,048, initialized caches, and exclude load time. CPU uses four XNNPACK threads. FunctionGemma Mobile Actions uses 512 input / 32 output tokens and context 1,024. **Samsung results are not Pixel estimates.**

Google reports S26 Ultra GPU E2B speculative decoding at **66.5–91.7 tokens/s**, depending on task, against a 51.5 baseline. CPU results include regressions on some tasks; measure the effect rather than assuming a universal speedup. [Speculative-decoding results](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)

### Official Pixel embedding latency

Google's Pixel 11 Pro TPU EmbeddingGemma 2 table reports:

| Input | Latency |
|---|---:|
| Text | 8.3 ms |
| Text + vision | 49 ms |
| Text + audio | 193.5 ms |
| Text + vision + audio | 149 ms |

These are embedding latencies, **not generated-answer latencies**. The non-monotonic values are transcribed from the source rather than inferred. [Official embedding benchmarks](https://developers.google.com/edge/litert-lm/overview)

### Resolved compatibility issue

A base Pixel 10 running Android 16 failed to initialize the E2B TPU model. The reporter confirmed that **updating to Android 17 resolved it**; the issue was closed. It is not evidence that the 12 GB phone is intrinsically unsupported. Verify OS/runtime versions before blaming RAM. [Closed issue and resolution](https://github.com/google-ai-edge/gallery/issues/1008)

## Implications for the prototype and brainstorm

At the reported rates, generating 50 tokens takes approximately **4.8 seconds at 10.42 tokens/s**, or **2 seconds at 24.67 tokens/s**, before other overhead. These are arithmetic illustrations, not end-to-end predictions. A short action or concise answer can fit the interaction better than a long paragraph.

Initial technical baseline:

- Gemma 4 E2B through LiteRT-LM, comparing explicit backends; E4B as the quality comparison.
- Short answers/structured actions; thinking and speculative decoding disabled for the initial baseline.
- Camera snapshots or short microphone recordings as future inputs.
- EmbeddingGemma only when retrieval improves the chosen use case.
- Measure cold/process-first loading, cached loading, warm TTFT, prefill/decode rates, completed-action latency, accuracy and sustained thermals separately.

Potential directions remain an offline visual helper, spoken task assistant, field observation tool, or contextual controller. The research establishes feasibility; user-problem selection remains open.

## Implementation follow-up

When inspecting the pinned LiteRT-LM 0.18.0 artifact for this repository, native `BenchmarkInfo` and `Conversation.getBenchmarkInfo()` were verified. Our benchmark should use those actual native token counts/rates, not the character-based estimation used by one community report. The fixed-token `benchmark()` helper pads/truncates input to the requested count and reinitializes the engine each invocation. Prompt-mode measurements need a separately retained engine and fresh conversations to measure warm interactive behavior. See [BENCHMARKING.md](BENCHMARKING.md) for the implemented methodology.
