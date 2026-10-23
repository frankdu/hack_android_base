package dev.gemmabench

import android.os.Build
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import dev.gemmabench.data.KidsEnglishDictionary
import dev.gemmabench.tts.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class DictionarySpeechInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test fun generatedSentencesHaveNonSilentAudioAndValidWordTiming() = runBlocking {
        val text = "A butterfly is an insect with colorful wings."
        val store = SpeechModelStore(context)
        LocalSpeech(context).use { engine ->
            for (option in listOf(SpeechOption.KITTEN, SpeechOption.KOKORO)) {
                val file = store.prepare(option) {}
                val audio = withContext(Dispatchers.Default) { engine.generate(option, text, file) }
                assertTrue("${option.label}: missing speech", audio.durationMs > 1000)
                assertTrue(audio.samples.any { kotlin.math.abs(it) > 0.01f })
                assertEquals(listOf("A", "butterfly", "is", "an", "insect", "with", "colorful", "wings"),
                    audio.words.map { text.substring(it.start, it.end) })
                assertTrue(audio.words.zipWithNext().all { (a, b) -> a.endMs <= b.startMs })
                assertTrue(audio.words.all { it.startMs >= 0 && it.endMs <= audio.durationMs })
                assertEquals(option == SpeechOption.KITTEN, audio.exactTiming)
                Log.i("VoiceAgent", "${option.label}: ${audio.durationMs} ms audio; aligned=${audio.exactTiming}; ${audio.words}")
            }
        }
    }

    @Test fun generatedSentenceUsesSavedVoiceAndPublishesHighlights() {
        val app = context.applicationContext as GemmaBenchApplication
        val saved = app.preferences.state.value
        lateinit var controller: SpeechController
        instrumentation.runOnMainSync {
            app.preferences.setSpeechEngine(SpeechOption.KITTEN)
            controller = SpeechController(context)
        }
        try {
            val ranges = mutableListOf<Pair<Int, Int>>()
            runBlocking {
                withContext(Dispatchers.Main) {
                    withTimeout(120_000) {
                        controller.speakText("Dogs bark to talk to each other.") { start, end ->
                            assertTrue("Playback animation must be active during speech", controller.state.value.isSpeaking)
                            ranges += start to end
                        }
                    }
                }
            }
            assertTrue("No word highlighting during playback", ranges.size >= 5)
            assertNull(controller.state.value.activeWord)
            assertFalse("Playback animation must stop with speech", controller.state.value.isSpeaking)
            assertTrue(controller.state.value.results.keys.any { it.second == SpeechOption.KITTEN })
            assertFalse(controller.state.value.results.keys.any { it.second == SpeechOption.KOKORO })
        } finally {
            instrumentation.runOnMainSync {
                controller.close(); app.preferences.reset()
                saved.speechEngine?.let(app.preferences::setSpeechEngine)
                saved.llmFamily?.let(app.preferences::setLlmFamily)
            }
        }
    }

    @Test fun everyDictionaryWordHasBothPronunciations() {
        val source = context.assets.open(KidsEnglishDictionary.ASSET_NAME).use { it.readBytes() }
        val tokens = context.assets.open("tts/dictionary-tokens.json").bufferedReader().use {
            JsonParser.parseReader(it).asJsonObject
        }
        val hash = MessageDigest.getInstance("SHA-256").digest(source).joinToString("") { "%02x".format(it) }
        assertEquals("Regenerate tokens after dictionary edits", hash, tokens["dictionarySha256"].asString)
        val words = KidsEnglishDictionary.parse(source.toString(Charsets.UTF_8).reader()).flatMap { it.items }.toSet()
        assertEquals(words, tokens.getAsJsonObject("words").keySet())
        for (word in words) for (model in listOf("kokoro", "kitten")) {
            val ids = tokens.getAsJsonObject("words").getAsJsonObject(word).getAsJsonArray(model)
            assertTrue("$model: $word", ids.size() in 1..510)
        }
    }

    /** Real models, CPU inference, non-silent audio, cold/warm timings, and WAV output. */
    @Test fun localModelsSpeakOnDevice() = runBlocking {
        val results = mutableListOf<Map<String, Any>>()
        val store = SpeechModelStore(context)
        LocalSpeech(context).use { engine ->
            for (option in listOf(SpeechOption.KOKORO, SpeechOption.KITTEN)) {
                val model = store.prepare(option) { if (it % 25 == 0) Log.i("DictionaryTTS", "${option.label}: $it%") }
                for (word in listOf("dog", "dog", "butterfly", "ice cream")) {
                    val audio = withContext(Dispatchers.Default) { engine.generate(option, word, model) }
                    assertTrue("${option.label} $word is too short", audio.durationMs > 100)
                    val rms = sqrt(audio.samples.sumOf { it.toDouble() * it } / audio.samples.size)
                    assertTrue("${option.label} $word is silent", rms > 0.001)
                    val record = mapOf("device" to Build.MODEL, "model" to option.label, "word" to word,
                        "loadMs" to audio.loadMs, "synthesisMs" to audio.generationMs,
                        "audioMs" to audio.durationMs, "rms" to rms)
                    results += record
                    Log.i("DictionaryTTS", record.toString())
                    writeSpeechWav(File(context.filesDir, "tts/${option.name.lowercase()}-sample.wav"), audio.samples)
                }
            }
        }
        File(context.filesDir, "tts/benchmark.json").writeText(GsonBuilder().setPrettyPrinting().create().toJson(results))
    }

    @Test fun localOptionsPlayAndCancellationClearsActiveWord() {
        ActivityScenario.launch(MainActivity::class.java).use {
            lateinit var controller: SpeechController
            instrumentation.runOnMainSync { controller = SpeechController(context) }
            try {
                for (option in listOf(SpeechOption.KOKORO, SpeechOption.KITTEN)) {
                    instrumentation.runOnMainSync { controller.speak("dog", option) }
                    runBlocking { withTimeout(180_000) { controller.state.first { it.activeWord == null } } }
                    val result = controller.state.value.results["dog" to option].orEmpty()
                    assertTrue("${option.label} playback failed: $result", result.contains("ms"))
                    Log.i("DictionaryTTS", "${option.label}: $result")
                }
                instrumentation.runOnMainSync {
                    controller.speak("butterfly", SpeechOption.KOKORO)
                    controller.stop()
                }
                assertNull(controller.state.value.activeWord)
            } finally { instrumentation.runOnMainSync { controller.close() } }
        }
    }

    @Test fun systemVoicePlays() {
        ActivityScenario.launch(MainActivity::class.java).use {
            lateinit var controller: SpeechController
            instrumentation.runOnMainSync { controller = SpeechController(context); controller.speak("dog", SpeechOption.SYSTEM) }
            try {
                runBlocking { withTimeout(30_000) { controller.state.first { it.activeWord == null } } }
                val result = controller.state.value.results["dog" to SpeechOption.SYSTEM].orEmpty()
                assertTrue("System voice failed: $result", result.contains("ms to speech"))
                Log.i("DictionaryTTS", "Android: $result")
            } finally { instrumentation.runOnMainSync { controller.close() } }
        }
    }

    @Test fun defaultPlaybackUsesSavedVoiceAndCancellationDoesNotTryAnotherEngine() {
        val app = context.applicationContext as GemmaBenchApplication
        val saved = app.preferences.state.value
        ActivityScenario.launch(MainActivity::class.java).use {
            lateinit var controller: SpeechController
            var controllerCreated = false
            try {
                instrumentation.runOnMainSync {
                    app.preferences.setSpeechEngine(SpeechOption.KOKORO)
                    controller = SpeechController(context)
                    controllerCreated = true
                    controller.speakDefault("dog")
                }
                runBlocking { withTimeout(180_000) { controller.state.first { it.activeWord == null } } }
                assertTrue("Preferred voice failed: ${controller.state.value}",
                    controller.state.value.results["dog" to SpeechOption.KOKORO].orEmpty().contains("ms synthesis"))
                assertNull(controller.state.value.results["dog" to SpeechOption.KITTEN])
                instrumentation.runOnMainSync { controller.speakDefault("butterfly"); controller.stop() }
                assertNull(controller.state.value.activeWord)
                assertNull(controller.state.value.results["butterfly" to SpeechOption.KITTEN])
            } finally {
                instrumentation.runOnMainSync {
                    if (controllerCreated) controller.close()
                    app.preferences.reset()
                    saved.speechEngine?.let(app.preferences::setSpeechEngine)
                    saved.llmFamily?.let(app.preferences::setLlmFamily)
                }
            }
        }
    }

    @Test fun automaticPlaybackUsesFirstWorkingEngine() {
        val app = context.applicationContext as GemmaBenchApplication
        val saved = app.preferences.state.value
        ActivityScenario.launch(MainActivity::class.java).use {
            var controller: SpeechController? = null
            try {
                instrumentation.runOnMainSync {
                    app.preferences.reset()
                    controller = SpeechController(context).also { it.speakDefault("dog") }
                }
                val speech = requireNotNull(controller)
                runBlocking { withTimeout(90_000) { speech.state.first { it.activeWord == null } } }
                val state = speech.state.value
                val played = listOf(SpeechOption.SYSTEM, SpeechOption.KITTEN, SpeechOption.KOKORO).filter {
                    state.results["dog" to it].orEmpty().contains("ms")
                }
                assertEquals("Exactly one engine should play: $state", 1, played.size)
                val engine = played.single()
                assertEquals("Default should retain the working engine: $state", engine, state.defaultOption)
                if (engine != SpeechOption.SYSTEM) {
                    assertTrue("System must be unavailable before local fallback: $state", SpeechOption.SYSTEM in state.unavailableOptions)
                }
                Log.i("DictionaryTTS", "Automatic playback: $engine; ${state.results}")
            } finally {
                instrumentation.runOnMainSync {
                    controller?.close()
                    app.preferences.reset()
                    saved.speechEngine?.let(app.preferences::setSpeechEngine)
                    saved.llmFamily?.let(app.preferences::setLlmFamily)
                }
            }
        }
    }
}
