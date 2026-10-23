package dev.gemmabench.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import dev.gemmabench.GemmaBenchApplication
import dev.gemmabench.settings.AppPreferencesStore
import dev.gemmabench.settings.speechCandidates
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

enum class SystemVoiceAvailability { CHECKING, AVAILABLE, UNAVAILABLE }

data class SpeechState(
    val activeWord: String? = null,
    val activeOption: SpeechOption? = null,
    val results: Map<Pair<String, SpeechOption>, String> = emptyMap(),
    val ready: Set<SpeechOption> = emptySet(),
    val preferredOption: SpeechOption? = null,
    val systemVoice: SystemVoiceAvailability = SystemVoiceAvailability.CHECKING,
    val systemVoiceDetail: String = "Checking for an offline English voice…",
    val failedOptions: Set<SpeechOption> = emptySet(),
    val lastError: Pair<String, String>? = null,
    val isSpeaking: Boolean = false,
) {
    val unavailableOptions: Set<SpeechOption> get() = failedOptions +
        (if (systemVoice == SystemVoiceAvailability.UNAVAILABLE) setOf(SpeechOption.SYSTEM) else emptySet())
    val defaultOption: SpeechOption? get() = speechCandidates(preferredOption, unavailableOptions).firstOrNull()
}

/** Activity-owned playback: a new tap cancels the previous request across all engines. */
class SpeechController(context: Context,
    private val preferences: AppPreferencesStore = (context.applicationContext as GemmaBenchApplication).preferences,
) : AutoCloseable {
    private val context = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val local = LocalSpeech(this.context)
    private val store = SpeechModelStore(this.context)
    private val mutableState = MutableStateFlow(SpeechState(ready = SpeechOption.entries.filter {
        it != SpeechOption.SYSTEM && it.file(this.context).length() == it.bytes
    }.toSet(), preferredOption = preferences.state.value.speechEngine))
    val state = mutableState.asStateFlow()
    private var job: Deferred<Unit>? = null
    private var request = 0L
    private var tts: TextToSpeech? = null
    private var ttsReady: CompletableDeferred<Int>? = null
    private var systemProbe: Job? = null
    private val manager = this.context.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()
    private val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        .setAudioAttributes(attributes).setOnAudioFocusChangeListener { change ->
            if (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT) stop()
        }.build()

    init {
        scope.launch { preferences.state.collect { prefs -> mutableState.update { it.copy(preferredOption = prefs.speechEngine) } } }
        refreshSystemAvailability()
    }

    fun selectPreferred(engine: SpeechOption) {
        stop()
        preferences.setSpeechEngine(engine)
        mutableState.update { it.copy(preferredOption = engine, failedOptions = it.failedOptions - engine) }
        if (engine == SpeechOption.SYSTEM) refreshSystemAvailability()
    }

    fun resetDefaults() {
        stop()
        preferences.reset()
        mutableState.update { it.copy(preferredOption = null, failedOptions = emptySet()) }
        refreshSystemAvailability()
    }

    fun refreshSystemAvailability() {
        if (systemProbe?.isActive == true || mutableState.value.activeWord != null) return
        systemProbe = scope.launch {
            mutableState.update { it.copy(systemVoice = SystemVoiceAvailability.CHECKING) }
            try { systemEngine() }
            catch (cancelled: CancellationException) {
                if (cancelled !is TimeoutCancellationException) throw cancelled
                systemUnavailable("System voice did not initialize. Using a local voice.")
            } catch (error: Exception) { systemUnavailable(error.message ?: "System voice unavailable.") }
        }
    }

    private fun systemUnavailable(message: String) {
        resetSystemEngine()
        mutableState.update { it.copy(systemVoice = SystemVoiceAvailability.UNAVAILABLE,
            systemVoiceDetail = message, ready = it.ready - SpeechOption.SYSTEM,
            failedOptions = it.failedOptions + SpeechOption.SYSTEM) }
    }

    fun speak(word: String, option: SpeechOption) { start(word, option) }
    fun speakDefault(word: String) { start(word, null) }

    /** Awaitable speech output used by the agent; never record until this completes. */
    suspend fun speakText(text: String, onRange: (Int, Int) -> Unit) {
        val task = start(text, null, onRange)
        try { task.await() } finally { if (job === task) stop() }
    }

    private fun start(word: String, explicit: SpeechOption?, onRange: ((Int, Int) -> Unit)? = null): Deferred<Unit> {
        stop()
        val id = ++request
        mutableState.update { it.copy(activeWord = word, activeOption = explicit ?: it.defaultOption,
            lastError = null,
            failedOptions = if (explicit == null) it.failedOptions else it.failedOptions - explicit) }
        val task = scope.async {
            try {
                if (explicit == null && preferences.state.value.speechEngine in listOf(null, SpeechOption.SYSTEM)) systemProbe?.join()
                val options = explicit?.let(::listOf) ?: speechCandidates(
                    preferences.state.value.speechEngine, mutableState.value.unavailableOptions)
                if (options.isEmpty()) {
                    mutableState.update { it.copy(results = it.results + ((word to SpeechOption.SYSTEM) to
                        "No voice available. Check Settings.")) }
                }
                var failure: Exception? = null
                for (option in options) {
                    ensureActive()
                    mutableState.update { it.copy(activeOption = option) }
                    fun report(message: String) {
                        if (id == request) mutableState.update { it.copy(results = it.results + ((word to option) to message)) }
                    }
                    try {
                        performSpeech(word, option, id, ::report, onRange)
                        mutableState.update { it.copy(failedOptions = it.failedOptions - option) }
                        return@async
                    } catch (error: TimeoutCancellationException) {
                        mutableState.update { it.copy(isSpeaking = false) }
                        failure = error
                        if (option == SpeechOption.SYSTEM) systemUnavailable("System voice timed out.")
                        report("Speech timed out. Tap to retry.")
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (error: Exception) {
                        mutableState.update { it.copy(isSpeaking = false) }
                        failure = error
                        if (error is SpeechOutputUnavailable) {
                            report(error.message ?: "Audio is busy. Try again.")
                            mutableState.update { it.copy(lastError = word to error.message.orEmpty()) }
                            throw error
                        }
                        if (option == SpeechOption.SYSTEM) systemUnavailable(error.message ?: "System voice failed.")
                        report(error.message ?: "Playback failed. Tap to retry.")
                    }
                    mutableState.update { it.copy(failedOptions = it.failedOptions + option) }
                }
                val message = failure?.message ?: "No voice available. Check Settings."
                mutableState.update { it.copy(lastError = word to message) }
                throw IllegalStateException(message, failure)
            } finally {
                if (id == request) {
                    manager.abandonAudioFocusRequest(focus)
                    mutableState.value = mutableState.value.copy(activeWord = null, activeOption = null, isSpeaking = false)
                }
            }
        }
        job = task
        return task
    }

    private suspend fun performSpeech(word: String, option: SpeechOption, id: Long, report: (String) -> Unit, onRange: ((Int, Int) -> Unit)?) {
        if (option == SpeechOption.SYSTEM) {
            systemProbe?.join()
            report("Opening voice…")
            speakSystem(word, id, report, onRange)
            return
        }
        report(if (option in mutableState.value.ready) "Preparing…" else "Downloading…")
        val file = store.prepare(option) { percent -> scope.launch { if (id == request) report("Download $percent%") } }
        currentCoroutineContext().ensureActive()
        mutableState.update { it.copy(ready = it.ready + option) }
        report("Generating…")
        val wav = File.createTempFile("word-", ".wav", context.cacheDir)
        try {
            val audio = withContext(Dispatchers.IO) {
                local.generate(option, word, file).also { writeSpeechWav(wav, it.samples) }
            }
            currentCoroutineContext().ensureActive()
            report("${audio.generationMs} ms synthesis" + (if (audio.loadMs > 10) " · ${audio.loadMs} ms load" else ""))
            play(wav, audio.words, onRange)
        } finally { wav.delete() }
    }

    private fun resetSystemEngine() {
        tts?.shutdown()
        tts = null
        ttsReady = null
    }

    private suspend fun systemEngine(): TextToSpeech {
        if (tts == null) {
            val initialized = CompletableDeferred<Int>()
            ttsReady = initialized
            tts = TextToSpeech(context) { status -> initialized.complete(status) }
        }
        check(withTimeout(15_000) { ttsReady!!.await() } == TextToSpeech.SUCCESS) {
            "Android voice unavailable. Install an English TTS engine in Android settings."
        }
        val engine = tts!!
        val voice = engine.voices.orEmpty().filter {
            it.locale.language == "en" && !it.isNetworkConnectionRequired &&
                TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features.orEmpty()
        }.sortedWith(compareByDescending<android.speech.tts.Voice> { it.locale.country == "US" }
            .thenByDescending { it.quality }.thenBy { it.latency }.thenBy { it.name }).firstOrNull()
            ?: error("No offline English voice. Download English voice data in Android TTS settings.")
        check(engine.setVoice(voice) == TextToSpeech.SUCCESS) { "Could not select the offline English voice." }
        engine.setSpeechRate(1f)
        engine.setPitch(1f)
        engine.setAudioAttributes(attributes)
        mutableState.update { it.copy(systemVoice = SystemVoiceAvailability.AVAILABLE,
            systemVoiceDetail = "Installed offline English voice", ready = it.ready + SpeechOption.SYSTEM,
            failedOptions = it.failedOptions - SpeechOption.SYSTEM) }
        return engine
    }

    private suspend fun speakSystem(word: String, id: Long, report: (String) -> Unit, onRange: ((Int, Int) -> Unit)?) {
        val engine = systemEngine()
        val completion = CompletableDeferred<Unit>()
        val started = CompletableDeferred<Unit>()
        val start = SystemClock.elapsedRealtime()
        var hasRanges = false
        var fallback: Job? = null
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                if (utteranceId != id.toString()) return
                started.complete(Unit)
                val elapsed = SystemClock.elapsedRealtime() - start
                scope.launch {
                    if (id != request) return@launch
                    mutableState.update { it.copy(isSpeaking = true) }
                    if (id == request) report("$elapsed ms to speech")
                    // Some installed engines omit range callbacks. Use a timed fallback until one arrives.
                    fallback = scope.launch {
                        for (timing in estimatedWordTimes(word, wordRanges(word).size * 400L)) {
                            delay((timing.startMs - (SystemClock.elapsedRealtime() - start - elapsed)).coerceAtLeast(0))
                            if (id == request && !hasRanges) onRange?.invoke(timing.start, timing.end)
                        }
                    }
                }
            }
            override fun onRangeStart(utteranceId: String?, start: Int, end: Int, frame: Int) {
                if (utteranceId != id.toString()) return
                scope.launch {
                    if (id == request) { hasRanges = true; onRange?.invoke(start, end) }
                }
            }
            override fun onDone(utteranceId: String?) {
                if (utteranceId == id.toString()) { started.complete(Unit); completion.complete(Unit) }
            }
            @Deprecated("Platform callback")
            override fun onError(utteranceId: String?) {
                if (utteranceId == id.toString()) {
                    val error = IllegalStateException("Android speech failed. Check voice data.")
                    started.completeExceptionally(error); completion.completeExceptionally(error)
                }
            }
            override fun onError(utteranceId: String?, errorCode: Int) {
                if (utteranceId == id.toString()) {
                    val error = IllegalStateException("Android speech error $errorCode. Check voice data.")
                    started.completeExceptionally(error); completion.completeExceptionally(error)
                }
            }
        })
        requestFocus()
        try {
            check(engine.speak(word, TextToSpeech.QUEUE_FLUSH, Bundle(), id.toString()) == TextToSpeech.SUCCESS) {
                "Android speech could not start."
            }
            withTimeout(8_000) { started.await() }
            withTimeout(120_000) { completion.await() }
        } finally { fallback?.cancel(); if (id == request) engine.stop() }
    }

    private fun requestFocus() {
        if (manager.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            throw SpeechOutputUnavailable()
        }
    }

    private class SpeechOutputUnavailable : IllegalStateException("Audio is busy. Try again after the current call or playback.")

    private suspend fun play(wav: File, words: List<SpokenWord>, onRange: ((Int, Int) -> Unit)?) {
        requestFocus()
        val player = MediaPlayer()
        var highlighter: Job? = null
        try {
            suspendCancellableCoroutine { continuation ->
                player.setAudioAttributes(attributes)
                player.setDataSource(wav.path)
                player.setOnCompletionListener { if (continuation.isActive) continuation.resume(Unit) }
                player.setOnErrorListener { _, what, extra ->
                    if (continuation.isActive) continuation.resumeWithException(IllegalStateException("Audio playback error $what/$extra"))
                    true
                }
                player.setOnPreparedListener {
                    if (continuation.isActive) {
                        it.start()
                        mutableState.update { state -> state.copy(isSpeaking = true) }
                        highlighter = scope.launch {
                            var previous: SpokenWord? = null
                            while (isActive && player.isPlaying) {
                                val current = words.lastOrNull { timing -> timing.startMs <= player.currentPosition }
                                if (current != null && current != previous) {
                                    onRange?.invoke(current.start, current.end)
                                    previous = current
                                }
                                delay(25)
                            }
                        }
                    }
                }
                player.prepareAsync()
            }
        } finally { highlighter?.cancel(); player.release() }
    }

    fun stop() {
        request++
        job?.cancel()
        job = null
        tts?.stop()
        manager.abandonAudioFocusRequest(focus)
        val current = mutableState.value
        val key = current.activeWord?.let { word -> current.activeOption?.let { word to it } }
        mutableState.value = current.copy(activeWord = null, activeOption = null, isSpeaking = false,
            results = if (key != null && current.results[key]?.contains("ms") != true)
                current.results + (key to "Stopped · tap to play") else current.results)
    }

    override fun close() {
        stop()
        scope.cancel()
        tts?.shutdown()
        // Native generation must finish before its session can be freed.
        CoroutineScope(Dispatchers.IO).launch { local.close() }
    }
}
