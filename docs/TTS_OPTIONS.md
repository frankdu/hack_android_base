# Dictionary TTS engines

The word browser now has one speaker icon using the Settings default and one assistant icon. The engine comparison below records the original research; see [word assistant](WORD_ASSISTANT.md) for generated sentences and highlighting.

Research checked 2026-10-21. The recommendations below are for short English dictionary words on ARM64 Android phones. They balance startup latency, voice quality, storage, licensing, and integration maturity; they are not a universal TTS leaderboard.

| Button | Selection | Why include it | Commercial use | Download |
| --- | --- | --- | --- | --- |
| Android | Android `TextToSpeech`, installed offline English voice, US preferred | Lowest integration cost; strong default for isolated words; device vendor maintains the engine | Platform API can be used in commercial apps. The installed engine/voice has its own terms; we do not redistribute it or its weights | No app model download; English voice data may need installing in system settings |
| Kokoro | Kokoro-82M v1.0, INT8 ONNX, Adam (`am_adam`) | Quality-focused open-weight candidate, consistent voice across phones, modest model size | Apache-2.0 weights and voice artifacts; preserve licenses/notices | 92,361,116 bytes, plus a bundled 522,240-byte voice style |
| Kitten | KittenTTS Nano v0.8, INT8 ONNX, Kiki (`expr-voice-5-f`) | Compact CPU-focused candidate, 15M parameters; useful contrast with the larger Kokoro model | Apache-2.0 weights and voice artifacts; preserve licenses/notices | 24,369,971 bytes, plus a bundled 409,600-byte extracted voice style |

The [Android API](https://developer.android.com/reference/android/speech/tts/TextToSpeech) provides engine initialization, voice selection, queued playback, completion/error callbacks, and shutdown. [Voice properties](https://developer.android.com/reference/android/speech/tts/Voice) expose locale, quality, latency, network requirements, and installation features. Our implementation excludes network voices and voices marked not installed. Availability and actual offline behavior ultimately depend on the installed engine.

The [Android SDK agreement](https://developer.android.com/studio/terms) grants SDK use to build applications for compatible Android devices and separately addresses third-party engine/content terms. Calling the installed TTS API does not grant permission to extract or redistribute its voices.

The [original Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M) identifies the 82M parameter architecture and Apache-2.0 weights. The [ONNX conversion card](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) documents the token/style/speed inputs, 24 kHz output, quantization variants, and voice styles. Kokoro is the quality-oriented recommendation here; listening tests with the target vocabulary should determine the final choice.

The [Kitten Nano INT8 model card](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) specifies 15M parameters, roughly 25 MB, Apache-2.0, and the Kiki voice. We follow the [official 0.8 reference implementation](https://github.com/KittenML/KittenTTS/releases/tag/0.8), including punctuation, token mapping, style selection, Kiki's 0.8 speed prior, and trailing-padding removal. This uses the current 0.8 Nano rather than the older 0.1 model in some Android demos.

Both local options run through [ONNX Runtime 1.31.0's Android Java/Kotlin API](https://onnxruntime.ai/docs/get-started/with-java.html), with [MIT licensing](https://github.com/microsoft/onnxruntime/blob/v1.31.0/LICENSE). Runtime and model notices are packaged under `assets/tts/`. No cloud API key, subscription, per-character fee, or server inference is required.

## Other options considered

- [sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/tts/apk.html) offers working Android integrations and many offline model families. Its Apache-2.0 runtime license does not replace each model's own license. We chose direct ONNX execution for these two models and prepared the fixed vocabulary ahead of time, avoiding an additional native text frontend in the app.
- Piper/VITS remain practical small CPU models, especially for broader language coverage. Licensing must be checked for each specific voice, and quality varies. Kokoro plus Kitten gives this experiment a clearer larger-model/smaller-model comparison.
- [Supertonic](https://github.com/supertone-oss-archive/supertonic) is an interesting fast on-device alternative, but the original repository was archived on September 9, 2026. It is not one of the three integrations selected here.

## Implementation and behavior

Every word has **Android**, **Kokoro**, and **Kitten** buttons. Tap a local model's button to download it and immediately speak the selected word. Progress appears under that button. Downloads can use mobile data, as stated on the word page. While speech is active, the title bar shows a Stop icon immediately before the Settings gear; it cancels playback/download work and disappears when the request ends. Playback controls do not insert an extra row into the word list. Transfers are foreground/session-owned and restart from the beginning after cancellation. No model is fetched just by opening a category.

Settings also provides a persistent **Text-to-speech Engine** selector. Tap the word itself to use the saved/default engine, with System → Kitten → Kokoro fallback when unavailable or failed. The three comparison buttons continue to use only their named engine. See [application defaults](APP_SETTINGS.md) for availability and preference rules.

The two model URLs use immutable Hugging Face revisions. Downloads are streamed to a `.part` file, checked against exact byte counts and SHA-256, then renamed into app-private storage. Failed transfers show a retryable error. Completed models are reused. Concurrent transfers are serialized, including across activity recreation. New playback cancels the previous request, stale synthesis results cannot play, and leaving the activity stops playback.

Neural inference uses two CPU threads, 24 kHz mono output, and one resident model at a time. Switching neural models reloads the session; repeated taps on the same model can be warm. GPU/NNAPI/Tensor TPU acceleration is not enabled or claimed. Session ownership is synchronized so cancellation/destruction cannot free a native session while inference is running.

ARM Android emulators (`ranchu`/`goldfish`) disable the optional KleidiAI kernels with the runtime's [documented session configuration](https://github.com/microsoft/onnxruntime/blob/v1.31.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h). This avoids an unsupported streaming-SME instruction path observed during testing; physical phones keep the runtime's normal hardware dispatch.

The UI reports neural **synthesis milliseconds**, with model load time separately when material. Android reports **time to the engine's `onStart` callback**. These are different measurements: neither is guaranteed to equal the time at which sound reaches the speaker. Download time is excluded from synthesis timing, and audio playback duration is not synthesis latency. Compare voice quality by listening, and compare timings only with these definitions in mind.

## Fixed vocabulary pronunciation assets

`scripts/prepare_dictionary_tts.py` prepares Kokoro and Kitten token IDs for all 7,537 unique dictionary entries. This keeps runtime pronunciation work tiny and avoids bundling eSpeak/phonemizer code or databases. It supports the packaged dictionary, not arbitrary runtime text. Pronunciation quality still needs listening review, especially ambiguous words such as “read.” Kokoro uses [Misaki](https://github.com/hexgrad/misaki); Kitten follows its own reference phonemizer.

After changing the seed dictionary, regenerate the assets with Python 3.12:

```sh
python3.12 -m venv /tmp/dictionary-tts
/tmp/dictionary-tts/bin/pip install 'misaki[en]==0.9.4' phonemizer-fork espeakng-loader numpy
/tmp/dictionary-tts/bin/python scripts/prepare_dictionary_tts.py --work-dir /tmp/tts-models
```

The generator selects `KOKORO_VOICE = 'am_adam'` and `KITTEN_VOICE_ID = 'expr-voice-5-f'` (Kiki). It downloads voice styles/configuration when absent and verifies the original voice artifact hashes. Kokoro's download cache includes the voice ID in its filename so changing the selection cannot reuse an older voice. The bundled filenames remain `kokoro-voice.bin` and `kitten-voice.bin`, as expected by the Android runtime. Misaki may fetch its English spaCy language package on first use. An instrumentation check verifies dictionary SHA-256, complete word coverage, and token lengths so stale generated assets are detected.

## Validation

Build, all 52 JVM unit tests, and Android lint passed (lint reports dependency/style warnings; no errors). The vocabulary-to-token SHA-256, word coverage, token ranges, and style bounds are checked by a JVM unit test. Both actual INT8 models also passed eight host synthesis checks with the current Adam and Kiki voice assets, using the Android implementation's token/style inputs and ONNX Runtime 1.24.1 CPU; every output was finite and non-silent. Raw host measurements are in [TTS_HOST_VALIDATION.json](TTS_HOST_VALIDATION.json), explicitly labeled as macOS host timings and including each voice ID.

Android tests in `DictionarySpeechInstrumentedTest` cover pronunciation coverage, real model downloads and non-silent audio, cold/warm synthesis timing, all three playback routes, and cancellation. `KidsEnglishInstrumentedTest` covers the speaker and assistant icon controls, category navigation, recreation, and Research Lab navigation.

Six tests passed on the existing ARM64 Android 16 emulator before the voice change: both local downloads/inference, both local playback routes and cancellation, vocabulary coverage, and all three dictionary UI/navigation tests. [TTS_EMULATOR_VALIDATION.json](TTS_EMULATOR_VALIDATION.json) preserves those historical Heart/Bella measurements and identifies their voice IDs; these are emulator results, not Pixel benchmarks or measurements of Adam/Kiki. The separate system-TTS playback test could not pass because the emulator's preinstalled `com.google.android.tts` crashes with SIGILL in `libgoogle_speech_sbg_tts_jni.so`; the app reports a timeout and permits a fresh engine initialization on retry. System playback still needs a physical device check.

The connected Pixel 10 Pro Fold disconnected before app installation, so phone latency and listening quality have not yet been measured. Pixel 11 has not been tested. CPU compatibility is expected from the ARM64 Android runtime; that is an engineering expectation, not measured Pixel 11 performance.

Run on an unlocked phone:

```sh
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -e class dev.gemmabench.DictionarySpeechInstrumentedTest,dev.gemmabench.KidsEnglishInstrumentedTest dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
adb shell run-as dev.gemmabench cat files/tts/benchmark.json
```

The synthesis test downloads about 117 MB if the models are missing and writes a JSON benchmark and two sample WAV files under app-private `files/tts/`. CPU timings and an RMS check are not perceptual quality evaluations.
