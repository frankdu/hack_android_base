package dev.gemmabench.voice

import dev.gemmabench.tts.*
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

@OptIn(ExperimentalCoroutinesApi::class)
class VoiceAgentTest {
    private val dispatcher = StandardTestDispatcher()
    @Before fun setup() { Dispatchers.setMain(dispatcher) }
    @After fun cleanup() { Dispatchers.resetMain() }

    @Test fun speechWaitsForSentenceBoundariesAndKeepsFinalFragment() {
        val splitter = SpeechSentences()
        assertEquals(emptyList<String>(), splitter.append("A dog is a friendly"))
        assertEquals(listOf("A dog is a friendly animal."), splitter.append(" animal. What would"))
        assertEquals(listOf("What would you like to know?"), splitter.append(" you like to know?"))
        assertEquals(listOf("Hello"), splitter.append("Hello", finished = true))
        assertEquals(emptyList<String>(), splitter.append("", finished = true))
        val long = "wonderful ".repeat(60)
        val chunks = SpeechSentences().append(long, finished = true)
        assertTrue(chunks.all { it.length <= 240 })
        assertEquals(long.trim(), chunks.joinToString(" "))
    }

    @Test fun highlightingRangesPreserveDisplayedPunctuationAndContractions() {
        val text = "It's a butterfly, isn’t it?"
        assertEquals(listOf("It's", "a", "butterfly", "isn’t", "it"), wordRanges(text).map { text.substring(it) })
        val times = estimatedWordTimes(text, 5000)
        assertEquals(0L, times.first().startMs)
        assertEquals(5000L, times.last().endMs)
        assertTrue(times.zipWithNext().all { (a, b) -> a.endMs <= b.startMs })
    }

    @Test fun recordedWavDeclaresActualRateAndPcmSize() {
        val pcm = ByteArray(32_000)
        val wav = pcmWav(pcm)
        val buffer = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", wav.copyOfRange(0, 4).toString(Charsets.US_ASCII))
        assertEquals(16_000, buffer.getInt(24))
        assertEquals(32_000, buffer.getInt(40))
        assertEquals(pcm.size + 44, wav.size)
    }

    @Test fun microphoneWaitsForSpeechThenGemmaTranscribesAndAnswersWithHistory() = runTest(dispatcher) {
        for (includeImage in listOf(true, false)) {
            val events = mutableListOf<String>()
            val prompts = mutableListOf<String>()
            val images = mutableListOf<ByteArray?>()
            val captions = mutableListOf<String>()
            val illustration = byteArrayOf(7, 8, 9)
            var imageLoads = 0
            var captures = 0
            val model = object : VoiceModel {
                override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                    if (prompt == VoicePrompts.caption("dog", "animal")) {
                        assertArrayEquals(illustration, jpeg)
                        assertNull(audio)
                        captions += prompt
                        return "Caption: A brown dog sits.\nSummary: The dog has a red collar and a white background."
                    }
                    prompts += prompt
                    images += jpeg
                    if (audio != null) { events += "transcribe"; return "Why do dogs bark?" }
                    val reply = if (prompts.size == 1) "A dog is a friendly animal. What would you like to know?" else "Dogs bark to communicate."
                    onText(reply)
                    return reply
                }
                override fun cancel() { events += "cancelModel" }
            }
            val input = object : VoiceInput {
                override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? {
                    events += "listen"
                    return if (captures++ == 0) byteArrayOf(1) else null
                }
                override fun cancel() {}
                override fun finish() {}
            }
            val speech = object : VoiceOutput {
                override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) {
                    events += "speak:$text"; onRange(0, 1); delay(100); events += "finished"
                }
                override fun stop() {}
            }
            VoiceAgentController(model, input, speech, image = { word, category ->
                assertEquals("dog", word)
                assertEquals("animal", category)
                imageLoads++
                illustration
            }, includeImage = { includeImage }).use { agent ->
                agent.start("dog", "animal", true)
                runCurrent()
                assertFalse(events.contains("listen"))
                assertEquals(0, agent.state.value.highlight!!.first)
                advanceUntilIdle()
                assertEquals(listOf("speak:A dog is a friendly animal.", "finished", "speak:What would you like to know?", "finished",
                    "listen", "transcribe", "speak:Dogs bark to communicate.", "finished", "listen"), events)
                assertTrue(prompts.last().contains("Child: Why do dogs bark?"))
                assertTrue(prompts.last().contains("Gemma: A dog is a friendly animal."))
                assertEquals(VoicePhase.PAUSED, agent.state.value.phase)
                assertNull(agent.state.value.highlight)
                assertEquals(3, agent.state.value.messages.size)
                assertEquals(1, imageLoads)
                assertEquals(if (includeImage) 0 else 1, captions.size)
                assertEquals(3, images.size)
                assertArrayEquals(if (includeImage) illustration else null, images[0])
                assertNull(images[1]) // Speech transcription gets audio only.
                assertArrayEquals(if (includeImage) illustration else null, images[2])
                assertEquals(includeImage, prompts.first().contains("attached illustration"))
                assertEquals(includeImage, prompts.last().contains("attached illustration"))
                assertFalse(prompts.first().contains("<image_description>"))
                assertEquals(!includeImage, prompts.last().contains("red collar"))
                assertFalse(agent.state.value.messages.any { it.text.contains("Caption:") || it.text.contains("red collar") })
            }
        }
    }

    @Test fun foregroundPreemptsCaptionWaitsForCleanupAndRetriesSilently() = runTest(dispatcher) {
        val prompts = mutableListOf<String>()
        val spoken = mutableListOf<String>()
        var captionAttempts = 0
        var modelActive = false
        var cleanedUp = false
        val description = "Caption: A brown dog sits.\nSummary: It wears a red collar."
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                assertFalse("Native requests must never overlap", modelActive)
                modelActive = true
                val isCaption = jpeg != null
                try {
                    if (isCaption) {
                        captionAttempts++
                        onText("Caption: Never show or speak this partial result.")
                        if (captionAttempts == 1) awaitCancellation()
                        return description
                    }
                    prompts += prompt
                    val reply = "Dogs are friendly."
                    onText(reply)
                    return reply
                } finally {
                    if (isCaption && captionAttempts == 1) {
                        withContext(NonCancellable) { delay(50); cleanedUp = true }
                    }
                    modelActive = false
                }
            }
            override fun cancel() {}
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
        VoiceAgentController(model, input, output, image = { _, _ -> byteArrayOf(1) }).use { agent ->
            agent.start("dog", "animal", true)
            runCurrent()
            assertEquals(1, prompts.size)
            assertEquals(1, captionAttempts)
            assertFalse(prompts.first().contains("<image_description>"))
            agent.send("What color is it?")
            runCurrent()
            assertEquals("Wait for caption JNI cleanup before using Gemma again", 1, prompts.size)
            advanceUntilIdle()
            assertTrue(cleanedUp)
            assertEquals(2, prompts.size)
            assertEquals(2, captionAttempts)
            assertFalse(prompts.last().contains("<image_description>"))
            agent.send("What is around its neck?")
            advanceUntilIdle()
            assertTrue(prompts.last().contains(description))
            assertEquals(2, captionAttempts)
            assertEquals(listOf("Dogs are friendly.", "Dogs are friendly.", "Dogs are friendly."), spoken)
            assertFalse(agent.state.value.messages.any { it.text.contains("Caption:") })
        }
    }

    @Test fun changingWordsDiscardsLateCaptionAndWaitsForOldNativeRequest() = runTest(dispatcher) {
        val prompts = mutableListOf<String>()
        var modelActive = false
        var cancellations = 0
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                assertFalse(modelActive)
                modelActive = true
                try {
                    if (jpeg != null) {
                        if (jpeg[0] == 1.toByte()) {
                            // Simulate a native encoder that returns a result despite a stop request.
                            withContext(NonCancellable) { delay(100) }
                            return "Caption: Old dog.\nSummary: A red collar."
                        }
                        return "Caption: New cat.\nSummary: A blue cushion."
                    }
                    prompts += prompt
                    return "Cats are animals."
                } finally { modelActive = false }
            }
            override fun cancel() { cancellations++ }
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? = null
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) {}
            override fun stop() {}
        }
        VoiceAgentController(model, input, output,
            image = { word, _ -> byteArrayOf(if (word == "dog") 1 else 2) }).use { agent ->
            agent.start("dog", "animal", false)
            runCurrent()
            agent.start("cat", "animal", false)
            advanceUntilIdle()
            assertTrue(cancellations > 0)
            agent.send("What is in the picture?")
            advanceUntilIdle()
            assertTrue(prompts.single().contains("blue cushion"))
            assertFalse(prompts.single().contains("red collar"))
            assertEquals("cat", agent.state.value.word)
        }
    }

    @Test fun failedCaptionLeavesConversationUsableAndCanRetry() = runTest(dispatcher) {
        var attempts = 0
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                if (jpeg != null) { attempts++; error("Vision unavailable") }
                return "Dogs are friendly."
            }
            override fun cancel() {}
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? = null
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) {}
            override fun stop() {}
        }
        VoiceAgentController(model, input, output, image = { _, _ -> byteArrayOf(1) }).use { agent ->
            agent.start("dog", "animal", true)
            advanceUntilIdle()
            agent.send("Why do dogs bark?")
            advanceUntilIdle()
            assertEquals(2, attempts)
            assertEquals(VoicePhase.PAUSED, agent.state.value.phase)
            assertEquals(3, agent.state.value.messages.size)
            assertFalse(agent.state.value.detail.orEmpty().contains("Vision unavailable"))
        }
    }

    @Test fun cancelStopsInferenceAndPreventsLateSpeechOrRecording() = runTest(dispatcher) {
        var cancelled = false
        var speech = false
        var recorded = false
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                delay(1000); onText("Hello."); return "Hello."
            }
            override fun cancel() { cancelled = true }
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? { recorded = true; return null }
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) { speech = true }
            override fun stop() {}
        }
        VoiceAgentController(model, input, output).use { agent ->
            agent.start("dog", "animal", true); runCurrent(); agent.cancel(); advanceUntilIdle()
            assertTrue(cancelled); assertFalse(speech); assertFalse(recorded)
            assertEquals(VoicePhase.PAUSED, agent.state.value.phase)
            agent.end(); assertNull(agent.state.value.word)
        }
    }

    @Test fun doneSubmitsRecordingButCancelDiscardsItAndStopsListening() = runTest(dispatcher) {
        for (submit in listOf(true, false)) {
            var transcriptions = 0
            var recordings = 0
            var replies = 0
            var recording = CompletableDeferred<ByteArray?>()
            val model = object : VoiceModel {
                override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                    if (audio != null) { transcriptions++; return "Why do dogs bark?" }
                    return "Dogs bark to communicate."
                }
                override fun cancel() {}
            }
            val input = object : VoiceInput {
                override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? {
                    recordings++
                    recording = CompletableDeferred()
                    onLevel(0.5f)
                    return recording.await()
                }
                override fun finish() { recording.complete(byteArrayOf(1)) }
                override fun cancel() {}
            }
            val output = object : VoiceOutput {
                override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) { replies++ }
                override fun stop() {}
            }
            VoiceAgentController(model, input, output).use { agent ->
                agent.start("dog", "animal", true)
                advanceUntilIdle()
                assertEquals(VoicePhase.LISTENING, agent.state.value.phase)
                if (submit) agent.finishListening() else agent.cancel()
                advanceUntilIdle()
                assertEquals(if (submit) 1 else 0, transcriptions)
                assertEquals(if (submit) 2 else 1, replies)
                assertEquals(if (submit) 2 else 1, recordings)
                assertEquals(if (submit) VoicePhase.LISTENING else VoicePhase.PAUSED, agent.state.value.phase)
                assertEquals(if (submit) 3 else 1, agent.state.value.messages.size)
                if (!submit) {
                    assertEquals(0f, agent.state.value.level)
                    assertEquals("Tap Talk to continue.", agent.state.value.detail)
                }
            }
        }
    }

    @Test fun cancelDuringTranscriptionNeverStartsAReplyOrReopensTheMicrophone() = runTest(dispatcher) {
        var cancelled = false
        var recordings = 0
        var replies = 0
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                if (audio != null) { awaitCancellation() }
                return "A dog is a friendly animal."
            }
            override fun cancel() { cancelled = true }
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? { recordings++; return byteArrayOf(1) }
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) { replies++ }
            override fun stop() {}
        }
        VoiceAgentController(model, input, output).use { agent ->
            agent.start("dog", "animal", true)
            advanceUntilIdle()
            assertEquals(VoicePhase.TRANSCRIBING, agent.state.value.phase)
            agent.cancel()
            advanceUntilIdle()
            assertTrue(cancelled)
            assertEquals(1, recordings)
            assertEquals(1, replies)
            assertEquals(1, agent.state.value.messages.size)
            assertEquals(VoicePhase.PAUSED, agent.state.value.phase)
        }
    }

    @Test fun stopSpeakingCancelsQueuedSpeechAndTalkStartsANewTurn() = runTest(dispatcher) {
        var spoken = 0
        var recordings = 0
        var stopped = false
        val model = object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, jpeg: ByteArray?, onText: (String) -> Unit): String {
                val reply = "Dogs bark to communicate. They also wag their tails."
                onText(reply)
                return reply
            }
            override fun cancel() {}
        }
        val input = object : VoiceInput {
            override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? { recordings++; awaitCancellation() }
            override fun finish() {}
            override fun cancel() {}
        }
        val output = object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) {
                spoken++
                onRange(0, 4)
                awaitCancellation()
            }
            override fun stop() { stopped = true }
        }
        VoiceAgentController(model, input, output).use { agent ->
            // Keep a completed child turn so Talk continues by listening instead of repeating the intro.
            agent.start("dog", "animal", false)
            agent.send("Why do dogs bark?")
            runCurrent()
            assertEquals(VoicePhase.SPEAKING, agent.state.value.phase)
            assertTrue(agent.state.value.speakingText.isNotEmpty())
            assertNotNull(agent.state.value.highlight)
            stopped = false
            agent.cancel()
            advanceUntilIdle()
            assertTrue(stopped)
            assertEquals(1, spoken)
            assertEquals(0, recordings)
            assertEquals(VoicePhase.PAUSED, agent.state.value.phase)
            assertEquals("", agent.state.value.speakingText)
            assertNull(agent.state.value.highlight)
            assertEquals(listOf(VoiceMessage(true, "Why do dogs bark?")), agent.state.value.messages)
            agent.listen()
            advanceUntilIdle()
            assertEquals(VoicePhase.LISTENING, agent.state.value.phase)
            assertEquals(1, recordings)
            assertEquals(1, spoken)
        }
    }

    @Test fun deniedPermissionAndInaudibleInputNeverInventAQuestion() {
        assertNull(VoicePrompts.transcript("[inaudible]"))
        assertNull(VoicePrompts.transcript(""))
        assertEquals("Hello", VoicePrompts.transcript("\"Hello\""))
        assertTrue(VoicePrompts.wantsToStop("Stop!"))
        assertFalse(VoicePrompts.wantsToStop("Why do dogs stop barking?"))
        val intro = VoicePrompts.reply("dog", "animal", emptyList())
        assertTrue(intro.contains("introduce the word's concept in one short sentence"))
        assertTrue(intro.contains("What would you like to know about dog?"))
    }
}
