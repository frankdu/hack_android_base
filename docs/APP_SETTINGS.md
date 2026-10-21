# Application defaults

Settings contains two persistent selectors. Stored values use app-private SharedPreferences and survive activity recreation and app restart. Missing or unrecognized values use automatic defaults. **Reset defaults** clears both saved choices.

## Text-to-speech Engine

The three options are **System**, **Kitten TTS**, and **Kokoro TTS**. On a fresh install, the app initializes the system TTS service and looks for an installed offline English voice. Automatic selection prefers System, then Kitten, then Kokoro. Local voices use Kiki and Adam respectively.

Tap a dictionary word to use the default engine. A saved engine is tried first; if unavailable or if synthesis fails, remaining engines are tried in System → Kitten → Kokoro order, without duplicate attempts. Cancellation stops the request rather than triggering fallback. Audio-focus denial is reported without trying another model. Failed engines are excluded from automatic selection for the current controller session; choosing that engine in Settings or using its comparison button permits a retry. Reopening Settings rechecks the system voice, and Reset defaults clears the failed-engine exclusions. A system voice that never starts speaking times out after eight seconds so fallback can proceed.

Local models may download on first use, including during fallback. Opening Settings does not download them. The three per-word comparison buttons explicitly test their named engine, so an error never silently plays another comparison voice.

## Default Local LLM

The two options are **Gemma 4 E2B** and **Gemma 4 E4B**. Automatic selection uses this priority among installed compatible artifacts:

1. E2B TPU
2. E4B TPU, if supported by a future catalog/runtime
3. E2B GPU
4. E4B GPU
5. E2B CPU
6. E4B CPU

The final repeated “E2B CPU” in the original request is interpreted as E4B CPU. The current catalog has E2B TPU artifacts for Tensor G5/G6 and no E4B TPU artifact. The unavailable E4B TPU branch is therefore skipped; the app does not invent a compatible artifact or download.

Selecting a family explicitly prefers its compatible TPU → GPU → CPU configurations before falling back to the other family. Availability requires an installed artifact, compatible model metadata, and (for TPU) the correct Tensor generation plus loadable runtime libraries. Missing models do not trigger downloads; Settings links to Research Lab for provisioning. With no compatible artifact, Settings shows that no local model is ready and preserves the preferred family.

The resolved default initializes Research Lab's Benchmark and Playground model/backend controls, and the Photos controls when the model supports vision. Manual choices in those screens are preserved. Saved preference changes apply on the next eligible screen entry/update; active or unfinished photo batches retain their original model/backend. Defaults do not alter the benchmark's existing rule that an explicitly started run reports native initialization/inference failures rather than silently changing the measured backend. Backend metadata and runtime library checks are prerequisites, not proof of a successful model initialization on every driver/firmware combination.

## Validation

`DefaultSelectionTest` covers speech ordering, saved preferences, failed-engine exclusion, TPU priority, missing/wrong-generation models, unsupported GPU backends, CPU fallback, E4B preference, no installed artifacts, and vision compatibility. Android tests cover settings navigation, persistence across activity recreation, reloading stored preferences, propagation to Research Lab and word playback, and cancellation without another-engine playback.

Validation passed: debug APK/test APK builds, all 52 JVM unit tests, lint (warnings only), and seven ARM64 Android emulator tests. The Android checks include actual default-voice playback, automatic fallback from the emulator's failed system engine to Kitten, explicit local-engine playback, preference persistence, and the dictionary navigation/toolbar regression. Native Gemma backend priority is covered with model metadata/runtime eligibility fixtures; this change does not claim new physical Pixel inference benchmarks.
