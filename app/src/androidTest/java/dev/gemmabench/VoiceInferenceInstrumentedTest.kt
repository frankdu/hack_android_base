package dev.gemmabench

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.*
import dev.gemmabench.data.LabStorage
import dev.gemmabench.data.BundledWordImages
import dev.gemmabench.runtime.LiteRtRuntime
import dev.gemmabench.tts.*
import dev.gemmabench.voice.*
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/** Opt-in real Gemma audio test: -e voiceInference true, with an installed multimodal E2B/E4B. */
@RunWith(AndroidJUnit4::class)
class VoiceInferenceInstrumentedTest {
    @Test fun bundledWordIllustrationProducesASpokenReply() = runBlocking {
        assumeTrue("Enable with -e wordImageInference true",
            InstrumentationRegistry.getArguments().getString("wordImageInference") == "true")
        val app = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as GemmaBenchApplication
        withTimeout(30_000) { while (app.lab.state.value.busy) delay(50) }
        assumeTrue("Install a multimodal Gemma model first", app.lab.state.value.models.any {
            it.metadata?.visionSupported == true && it.metadata.audioSupported == true
        })
        val jpeg = requireNotNull(BundledWordImages(app.assets).jpeg("dog", "animal"))
        val prompt = VoicePrompts.reply("dog", "animal",
            listOf(VoiceMessage(true, "What does the dog in this picture look like?")), hasImage = true)
        var streamed = ""
        val reply = withTimeout(600_000) {
            var result: String? = null
            while (result == null) {
                try { result = app.lab.voiceText(prompt, jpeg = jpeg) { streamed = it } }
                catch (error: IllegalStateException) {
                    // Startup immediately polls downloads; that short reservation is not foreground inference.
                    if (!error.message.orEmpty().startsWith("The local model is busy.")) throw error
                    delay(100)
                }
            }
            result
        }
        Log.i("VoiceInference", "Bundled dog illustration: $reply")
        assertTrue("No picture explanation: $reply", reply.length > 20)
        assertTrue("Reply should discuss dog: $reply", reply.contains("dog", ignoreCase = true))
        assertFalse("Assistant should speak naturally, not return photo JSON: $reply", reply.trim().startsWith("{"))
        assertEquals(reply, streamed)
    }

    @Test fun gemmaTranscribesAudioAndExplainsTheSelectedWord() = runBlocking {
        assumeTrue("Enable with -e voiceInference true",
            InstrumentationRegistry.getArguments().getString("voiceInference") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val storage = LabStorage(context)
        val requestedSlot = InstrumentationRegistry.getArguments().getString("voiceModel")
        val model = storage.refreshModelMetadata().firstOrNull {
            !it.slot.tpu && it.metadata?.audioSupported == true && "CPU" in it.metadata.textBackends &&
                (requestedSlot == null || it.slot.name == requestedSlot)
        }
        assumeTrue("Install a multimodal standard Gemma model first", model != null)
        val installed = requireNotNull(model)
        val file = SpeechModelStore(context).prepare(SpeechOption.KITTEN) {}
        val samples = withContext(Dispatchers.IO) {
            LocalSpeech(context).use { it.generate(SpeechOption.KITTEN, "Why do dogs bark?", file).samples }
        }
        // 24 kHz → 16 kHz, interpolated, for the same WAV path used by the microphone.
        val pcm = ByteBuffer.allocate(samples.size * 2 / 3 * 2).order(ByteOrder.LITTLE_ENDIAN)
        repeat(pcm.capacity() / 2) { i ->
            val position = i * 1.5
            val left = position.toInt()
            val fraction = (position - left).toFloat()
            val sample = samples[left] * (1f - fraction) + samples[minOf(left + 1, samples.lastIndex)] * fraction
            pcm.putShort((sample.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        }
        Executors.newSingleThreadExecutor().asCoroutineDispatcher().use { worker ->
            withContext(worker) {
                val runtime = LiteRtRuntime(context, storage::file)
                try {
                    val settings = RunSettings(accelerator = Accelerator.CPU, workload = Workload.PROMPT,
                        contextTokens = installed.metadata.voiceContextTokens(), outputTokens = 128, prompt = VoicePrompts.TRANSCRIBE)
                    val transcript = withTimeout(600_000) { runtime.voiceText(installed, settings, pcmWav(pcm.array())) {} }
                    Log.i("VoiceInference", "Transcript: $transcript")
                    assertTrue("Unexpected transcription: $transcript", transcript.contains("dog", ignoreCase = true))
                    assertTrue("Unexpected transcription: $transcript", transcript.contains("bark", ignoreCase = true))
                    val reply = runtime.voiceText(installed, settings.copy(prompt = VoicePrompts.reply("dog", "animal",
                        listOf(VoiceMessage(true, transcript)))), null) {}
                    Log.i("VoiceInference", "Reply: $reply")
                    assertTrue("No explanation: $reply", reply.length > 20)
                } finally { runtime.unload() }
            }
        }
    }

    @Test fun selectedE4bIntroducesTheWordThroughAssistant() = runBlocking {
        assumeTrue("Enable with -e voiceInference true",
            InstrumentationRegistry.getArguments().getString("voiceInference") == "true")
        val app = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as GemmaBenchApplication
        withTimeout(30_000) { while (app.lab.state.value.busy) delay(50) }
        assumeTrue("Install multimodal E4B first", app.lab.state.value.models.any {
            it.slot == ModelSlot.E4B && it.metadata?.audioSupported == true
        })
        val savedFamily = app.preferences.state.value.llmFamily
        try {
            app.preferences.setLlmFamily(ModelSlot.E4B)
            withTimeout(10_000) { while (app.lab.state.value.preferredLlmFamily != ModelSlot.E4B) delay(50) }
            var streamed = ""
            val reply = withTimeout(600_000) {
                app.lab.voiceText(VoicePrompts.reply("dog", "animal", emptyList())) { streamed = it }
            }
            Log.i("VoiceInference", "Selected E4B introduction: $reply")
            assertTrue("No explanation: $reply", reply.length > 20)
            assertTrue("Reply should introduce dog: $reply", reply.contains("dog", ignoreCase = true))
            assertEquals(reply, streamed)
        } finally {
            if (savedFamily == null) {
                val saved = app.preferences.state.value
                app.preferences.reset()
                saved.speechEngine?.let(app.preferences::setSpeechEngine)
                app.preferences.setAssistantShowTextInput(saved.assistantShowTextInput)
            } else app.preferences.setLlmFamily(savedFamily)
        }
    }
}
