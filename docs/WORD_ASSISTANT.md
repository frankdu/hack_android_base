# Word assistant

Each word has a speaker icon and a sparkle assistant icon. The speaker uses the TTS preference in Settings. The assistant opens a conversation about that word, gives a one-sentence concept introduction and asks what the child wants to know. Allow microphone access on first use. Install a multimodal Gemma E2B/E4B in **Settings → Research Lab → Models** before starting; **Default Local LLM** controls the preferred family.

The assistant speaks, then automatically listens. The main button is bright green while listening, transcribing, thinking or speaking; it uses the normal theme color when ready. While listening, **Done** ends and submits the current recording, and a separate **Cancel** text button discards it. While transcribing or thinking, the main button is **Cancel**; while speaking, it is **Stop speaking**. Both actions cancel the current turn and return to ready without reopening the microphone. **Talk** starts a new turn with the completed conversation history; interrupted speech cannot be resumed. No Pause control is shown. Settings remains available in the title bar. Eight seconds without speech returns to ready; a spoken turn ends after 1.4 seconds of silence or twenty seconds total. A typed question is also supported. Back ends the session and returns to the word list. Backgrounding stops audio and inference; rotation ends the Activity-owned session. Conversation text is held in memory, and microphone audio is never saved to disk. TTS WAV files are temporary cache files deleted after playback/cancellation.

While the assistant is listening, transcribing, thinking or speaking, its root layout uses Compose's [`Modifier.keepScreenOn()`](https://developer.android.com/reference/kotlin/androidx/compose/ui/keepScreenOn.modifier). Ready and error states remove that request, as does leaving the assistant layout. Android also releases the visible window's screen hold when the app is backgrounded. The normal dim/sleep countdown starts again when the hold ends; the app does not change screen timeout or brightness settings, acquire its own wake lock, or prevent the power button from turning the screen off. This follows [Android's keep-screen-on guidance](https://developer.android.com/develop/background-work/background-tasks/awake/screen-on); Android's window manager uses `ON_AFTER_RELEASE` for its screen hold so releasing it resets the idle timer ([platform implementation](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/wm/DisplayContent.java)).

## Reviewing the layout

In a debug build, open **Settings → Preview word assistant layout**. This displays the actual `WordAssistantPage` with labeled sample data, without loading Gemma, recording audio or playing speech. The State menu switches between speaking, listening, transcribing, thinking and ready layouts. Android Studio previews for those states are in `app/src/debug/java/dev/gemmabench/ui/WordAssistantPreviewActivity.kt`; edit the real layout in `app/src/main/java/dev/gemmabench/ui/WordAssistantPage.kt`.

Dictionary playback is shown with outward-moving sound waves on the speaker icon, only during actual audio output. Word cards show a text line only for playback errors; preparation, download and playback status never add a line or change the card height.

## Model and speech pipeline

`VoiceAgentController` manages the turn state machine. `VoiceRecorder` captures mono 16 kHz PCM and packages a WAV in memory. Gemma receives `Content.AudioBytes` with a transcription-only instruction. The resulting transcript is shown as the child's message and used with the selected word, category and recent dialogue in a second Gemma request. Inaudible/empty results ask for a retry. An energy gate rejects silence before transcription; it is a simple voice activity detector, so noisy rooms and quiet children's voices need physical-device tuning.

The same `LabController` worker and `LiteRtRuntime` own native engine resources for both the lab and assistant. Audio modality/backend metadata is checked before initialization, and the engine is reused between transcription and response requests. A lab operation already in progress produces a retry message. The model gets a fresh bounded context per request; recent text dialogue supplies memory without accumulating raw audio or exhausting a native chat session. Replies are capped at 192 output tokens and prompted to use at most fifty spoken words.

Replies omit image bytes by default. A session-owned coroutine job starts after foreground reply generation, using the same Gemma worker to produce a bounded `Caption:` and `Summary:` for the displayed image. Its tokens never enter chat or TTS. Completed descriptions are added silently to later text prompts and cached by image bytes and caption instructions. A foreground transcription or reply cancels unfinished captioning and joins its native cleanup before requesting inference; captioning can retry after the next reply. Switching words discards the previous in-memory context, and late cancelled results cannot attach to a new word. Backgrounding or leaving the assistant cancels unfinished work; completed cache entries remain reusable. The image-input switch in Settings enables direct-image replies for comparison. See [bundled word images](LOCAL_WORD_IMAGES.md).

The assistant uses an 8,192-token context when model metadata is missing or reports a nonpositive context size (as the standard E4B artifact does). Positive dynamic limits cap that default; positive fixed limits are preserved. An unknown size must never be passed to LiteRT-LM as a zero-token engine context.

The speech adapter is an application-controlled `speakText(text, onRange)` operation, rather than model-generated tool syntax. Complete sentences enter an ordered queue as model text streams, allowing playback to begin before the entire reply is generated. The microphone stays closed until all queued speech completes, with a short delay for the speaker's audio tail. Cancellation invalidates callbacks and cancels active JNI processing. A recognized stop request pauses after its reply. Child-appropriate prompting guides the model but does not constitute a separately trained content filter.

## Voices and synchronized text

All speech uses the configured TTS preference and the existing availability fallback order. System uses an installed offline English voice. Local Kitten/Kokoro models download once if missing; this can use mobile data. Both local engines now synthesize generated sentences using a bundled 174,945-entry US pronunciation lexicon derived from [Misaki 0.9.4](https://github.com/hexgrad/misaki), with its Apache-2.0 license and modification notice. Common function words use reduced forms. Unknown words are spelled; numbers should be written out. Context-sensitive heteronyms use the default lexicon reading, and dictionary-based sentence pronunciation is less sophisticated than Misaki's full Python contextual pipeline.

The current spoken sentence is displayed in large text with the active word highlighted. System uses [`UtteranceProgressListener.onRangeStart`](https://developer.android.com/reference/android/speech/tts/UtteranceProgressListener); engines without range callbacks use an approximate timed fallback. Kitten exports 40 Hz phoneme duration frames mapped back to displayed word ranges and synchronized to `MediaPlayer.currentPosition`. The current Kokoro ONNX artifact exports only audio, so its highlights use estimated word weights over the actual audio duration. **Kokoro and System engines without range callbacks do not provide exact word alignment.**

Regenerate assets in the dictionary preparation Python environment:

```sh
python scripts/prepare_voice_lexicon.py --work-dir /tmp/tts-models
```

This uses the Kokoro config prepared by `prepare_dictionary_tts.py` and `misaki==0.9.4`. The gzip lexicon deliberately uses a `.bin` extension because AAPT transparently unpacks `.gz` assets and strips that suffix.

## Validation

[Validation measurements](WORD_ASSISTANT_VALIDATION.json) record the local speech checks and remaining device work.

`VoiceAgentTest` also covers text-only introductions, hidden image context on follow-ups, direct-image opt-in, foreground preemption with delayed native cleanup, late results when switching words, and caption failures. `WordImageCaptionTest` covers response validation and disk cache invalidation. Opt in to a real on-device caption plus text-only follow-up check with:

```sh
adb shell am instrument -w -e wordImageCaptionInference true -e class dev.gemmabench.VoiceInferenceInstrumentedTest#backgroundCaptionGroundsTextOnlyFollowup dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

`VoiceAgentTest` checks sentence chunking, displayed character ranges, WAV encoding, speaking-before-listening, Gemma transcription routing, conversation history, silence handling and cancellation. `KidsEnglishInstrumentedTest` checks the two icon controls, navigation, settings and missing-model recovery. Two generated-speech tests in `DictionarySpeechInstrumentedTest` exercise real local ONNX sentence synthesis, word timing and saved-voice playback highlighting.

`WordAssistantScreenInstrumentedTest` checks screen-on requests across all active phases, ready, error and layout removal. Its opt-in emulator test verifies that an untouched active conversation outlasts a 15-second system timeout and that returning to ready starts a fresh countdown before the screen sleeps. It temporarily changes the emulator's timeout and charging stay-awake settings and restores both in `finally`:

```sh
adb -s emulator-5554 shell am instrument -w -e screenTimeout true -e class dev.gemmabench.WordAssistantScreenInstrumentedTest dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

```sh
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -e class dev.gemmabench.KidsEnglishInstrumentedTest,dev.gemmabench.DictionarySpeechInstrumentedTest#generatedSentencesHaveNonSilentAudioAndValidWordTiming,dev.gemmabench.DictionarySpeechInstrumentedTest#generatedSentenceUsesSavedVoiceAndPublishesHighlights dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

The opt-in test below requires an installed multimodal standard model and enough device memory. It synthesizes a question, passes actual WAV audio to Gemma, checks transcription and requests a contextual answer. It is not a live child/microphone test.

```sh
adb shell am instrument -w -e voiceInference true -e class dev.gemmabench.VoiceInferenceInstrumentedTest dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

To target E4B for the audio test as well, add `-e voiceModel E4B`. The same test class also checks an E4B introduction through `LabController.voiceText` with E4B selected in the saved preferences, restoring the original selection afterward.

Live Gemma transcription, microphone turn detection, perceptual fluency and latency still need a physical phone check with the target child's voice. The development emulator has no installed Gemma model; mock dialogue tests do not establish real-model response quality.

The multimodal integration follows the [version-matched LiteRT-LM Kotlin API](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.18.0/docs/api/kotlin/getting_started.md).
