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
import java.util.concurrent.CopyOnWriteArrayList
import java.io.File

/** Opt-in real Gemma audio test: -e voiceInference true, with an installed multimodal E2B/E4B. */
@RunWith(AndroidJUnit4::class)
class VoiceInferenceInstrumentedTest {
    @Test fun backgroundCaptionGroundsTextOnlyFollowup() = runBlocking {
        assumeTrue("Enable with -e wordImageCaptionInference true",
            InstrumentationRegistry.getArguments().getString("wordImageCaptionInference") == "true")
        val app = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as GemmaBenchApplication
        withTimeout(30_000) { while (app.lab.state.value.busy) delay(50) }
        assumeTrue("Install a multimodal Gemma model first", app.lab.state.value.models.any {
            it.metadata?.visionSupported == true && it.metadata.audioSupported == true
        })
        val jpeg = requireNotNull(BundledWordImages(app.assets).jpeg("dog", "animal"))
        val directory = File(app.cacheDir, "caption-validation-${System.nanoTime()}")
        val cache = WordImageCaptionCache(directory)
        val requests = CopyOnWriteArrayList<Pair<String, Boolean>>()
        val spoken = CopyOnWriteArrayList<String>()
        val captionResult = CompletableDeferred<String>()
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                requests += prompt to (jpeg != null)
                val start = android.os.SystemClock.elapsedRealtime()
                try {
                    var result: String? = null
                    while (result == null) {
                        try { result = app.lab.voiceText(prompt, audio, jpeg, onText) }
                        catch (error: IllegalStateException) {
                            if (!error.message.orEmpty().startsWith("The local model is busy.")) throw error
                            delay(100)
                        }
                    }
                    Log.i("VoiceInference", "${if (jpeg != null) "Background caption" else "Text reply"}: " +
                        "${android.os.SystemClock.elapsedRealtime() - start} ms; $result")
                    if (jpeg != null) captionResult.complete(result)
                    return result
                } catch (error: Exception) {
                    captionResult.completeExceptionally(error)
                    throw error
                }
            }
            override fun cancel() = app.lab.cancelVoiceInference()
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? = awaitCancellation()
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) { spoken += text }
            override fun stop() {}
        }
        val agent = withContext(Dispatchers.Main) {
            VoiceAgentController(model, input, output, image = { _, _ -> jpeg }, captionCache = cache)
        }
        try {
            withContext(Dispatchers.Main) { agent.start("dog", "animal", true) }
            val raw = withTimeout(600_000) { captionResult.await() }
            val description = requireNotNull(WordImageCaption.parse(raw)) { "Incomplete caption: $raw" }
            withTimeout(10_000) {
                while (cache.read(jpeg, VoicePrompts.caption("dog", "animal")) == null) delay(50)
            }
            assertFalse("The introduction must not send an image", requests.first().second)
            assertTrue("Only the background request sees the image", requests[1].second)
            withContext(Dispatchers.Main) { agent.send("What does the dog in this picture look like?") }
            withTimeout(600_000) {
                while (agent.state.value.messages.count { !it.child } < 2) delay(50)
            }
            assertEquals(3, requests.size)
            assertFalse(requests.last().second)
            assertTrue(requests.last().first.contains(description.context))
            assertFalse(agent.state.value.messages.any { it.text.contains("Caption:") || it.text.contains("Summary:") })
            assertFalse(spoken.any { it.contains("Caption:") || it.contains("Summary:") })
        } finally {
            withContext(Dispatchers.Main) { agent.close() }
            directory.deleteRecursively()
        }
    }

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
                app.preferences.setAssistantIncludeImage(saved.assistantIncludeImage)
            } else app.preferences.setLlmFamily(savedFamily)
        }
    }
}
