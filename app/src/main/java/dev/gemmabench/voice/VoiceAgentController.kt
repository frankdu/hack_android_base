package dev.gemmabench.voice

import android.content.Context
import dev.gemmabench.LabController
import dev.gemmabench.tts.SpeechController
import dev.gemmabench.tts.SpeechSentences
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

enum class VoicePhase { THINKING, SPEAKING, LISTENING, TRANSCRIBING, PAUSED, ERROR }
data class VoiceAgentState(
    val word: String? = null, val category: String = "", val phase: VoicePhase = VoicePhase.PAUSED,
    val messages: List<VoiceMessage> = emptyList(), val speakingText: String = "",
    val highlight: IntRange? = null, val level: Float = 0f, val detail: String? = null,
)

/** Half-duplex agent: Gemma ASR → bounded dialogue → sentence speech → next listening turn.
 * Model output is the speak tool's argument, executed sentence by sentence as tokens arrive.
 * Session text and audio are never persisted. A generation guards all late callbacks.
 */
interface VoiceModel {
    suspend fun generate(prompt: String, audio: ByteArray?, onText: (String) -> Unit): String
    fun cancel()
}
interface VoiceInput {
    suspend fun capture(onLevel: (Float) -> Unit): ByteArray?
    fun finish()
    fun cancel()
}
interface VoiceOutput {
    suspend fun speakText(text: String, onRange: (Int, Int) -> Unit)
    fun stop()
}

class VoiceAgentController(private val model: VoiceModel, private val recorder: VoiceInput,
                           private val speech: VoiceOutput) : AutoCloseable {
    constructor(context: Context, lab: LabController, speech: SpeechController) : this(
        object : VoiceModel {
            override suspend fun generate(prompt: String, audio: ByteArray?, onText: (String) -> Unit) = lab.voiceText(prompt, audio, onText)
            override fun cancel() = lab.cancelVoiceInference()
        }, VoiceRecorder(context.applicationContext), object : VoiceOutput {
            override suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) = speech.speakText(text, onRange)
            override fun stop() = speech.stop()
        })
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val mutable = MutableStateFlow(VoiceAgentState())
    val state = mutable.asStateFlow()
    private var job: Job? = null
    @Volatile private var generation = 0L
    private var inferenceActive = false

    fun start(word: String, category: String, microphoneAllowed: Boolean) {
        end()
        mutable.value = VoiceAgentState(word = word, category = category)
        if (!microphoneAllowed) {
            mutable.update { it.copy(detail = "Allow microphone access in Android app settings, or type a question below.") }
            return
        }
        runTurn(intro = true)
    }

    fun listen() { if (state.value.word != null) runTurn(intro = state.value.messages.isEmpty()) }
    fun send(text: String) { if (text.isNotBlank() && state.value.word != null) runTurn(question = text.trim().take(600)) }
    fun finishListening() { recorder.finish() }

    private fun runTurn(intro: Boolean = false, question: String? = null) {
        val previous = job
        cancel()
        val id = generation
        job = scope.launch {
            try {
                previous?.join()
                var next = question
                var opening = intro
                while (isActive && id == generation) {
                    if (!opening && next == null) {
                        delay(250) // Let the speaker's audio tail fade before opening the microphone.
                        mutable.update { it.copy(phase = VoicePhase.LISTENING, detail = null, speakingText = "", level = 0f) }
                        val audio = recorder.capture { level -> if (id == generation) mutable.update { it.copy(level = level) } }
                        if (audio == null) {
                            mutable.update { it.copy(phase = VoicePhase.PAUSED, level = 0f, detail = "I didn’t catch that. Tap Talk when you’re ready.") }
                            break
                        }
                        mutable.update { it.copy(phase = VoicePhase.TRANSCRIBING, level = 0f) }
                        next = VoicePrompts.transcript(infer(VoicePrompts.TRANSCRIBE, audio))
                        if (next == null) {
                            mutable.update { it.copy(phase = VoicePhase.PAUSED, detail = "I couldn’t understand that. Tap Talk and try again.") }
                            break
                        }
                    }
                    next?.let { question ->
                        mutable.update { it.copy(messages = (it.messages + VoiceMessage(true, question)).takeLast(12)) }
                    }
                    val stopAfterReply = next?.let(VoicePrompts::wantsToStop) == true
                    mutable.update { it.copy(phase = VoicePhase.THINKING, detail = null, speakingText = "", highlight = null) }
                    val snapshot = state.value
                    sayReply(VoicePrompts.reply(snapshot.word!!, snapshot.category, snapshot.messages), id)
                    if (stopAfterReply) {
                        mutable.update { it.copy(phase = VoicePhase.PAUSED, detail = "Tap Talk whenever you want to explore more.") }
                        break
                    }
                    opening = false
                    next = null
                }
            } catch (cancelled: CancellationException) {
                if (id == generation) mutable.update { it.copy(phase = VoicePhase.PAUSED) }
                throw cancelled
            }
            catch (error: Exception) {
                if (id == generation) mutable.update { it.copy(phase = VoicePhase.ERROR, detail = error.message ?: "Let’s try that again.", highlight = null, level = 0f) }
            } finally {
                if (id == generation) mutable.update { it.copy(highlight = null, level = 0f) }
            }
        }
    }

    private suspend fun infer(prompt: String, audio: ByteArray? = null, onText: (String) -> Unit = {}): String {
        inferenceActive = true
        try { return model.generate(prompt, audio, onText) } finally { inferenceActive = false }
    }

    private suspend fun sayReply(prompt: String, id: Long) = coroutineScope {
        // At most 192 output tokens are produced, bounding this queue to one short reply.
        val sentences = Channel<String>(Channel.UNLIMITED)
        val producer = async {
            var previous = ""
            val splitter = SpeechSentences()
            try {
                val result = infer(prompt, onText = { full ->
                    if (id == generation && full.startsWith(previous)) {
                        splitter.append(full.substring(previous.length)).forEach { sentences.trySend(it) }
                        previous = full
                    }
                })
                splitter.append(result.removePrefix(previous), finished = true).forEach { sentences.send(it) }
                check(result.isNotBlank()) { "Gemma didn’t return a reply. Tap Talk to try again." }
            } finally { sentences.close() }
        }
        var spoken = ""
        for (sentence in sentences) {
            ensureActive()
            mutable.update { it.copy(phase = VoicePhase.SPEAKING, speakingText = sentence, highlight = null) }
            speech.speakText(sentence) { start, end ->
                if (id == generation) mutable.update { it.copy(highlight = start until end) }
            }
            spoken = listOf(spoken, sentence).filter(String::isNotEmpty).joinToString(" ")
            mutable.update { it.copy(highlight = null) }
        }
        producer.await()
        mutable.update { it.copy(messages = (it.messages + VoiceMessage(false, spoken)).takeLast(12), speakingText = "", highlight = null) }
    }

    /** Discard the active turn and stop automatic listening; Talk starts a new turn. */
    fun cancel() {
        generation++
        if (inferenceActive) model.cancel()
        job?.cancel()
        job = null
        recorder.cancel()
        speech.stop()
        mutable.update { it.copy(phase = VoicePhase.PAUSED, speakingText = "", highlight = null,
            level = 0f, detail = "Tap Talk to continue.") }
    }
    fun end() { cancel(); mutable.value = VoiceAgentState() }
    override fun close() { end(); scope.cancel() }
}
