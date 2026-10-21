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
    private var job: Job? = null
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

    fun speak(word: String, option: SpeechOption) = start(word, option)
    fun speakDefault(word: String) = start(word, null)

    private fun start(word: String, explicit: SpeechOption?) {
        stop()
        val id = ++request
        mutableState.update { it.copy(activeWord = word, activeOption = explicit ?: it.defaultOption,
            failedOptions = if (explicit == null) it.failedOptions else it.failedOptions - explicit) }
        job = scope.launch {
            try {
                if (explicit == null && preferences.state.value.speechEngine in listOf(null, SpeechOption.SYSTEM)) systemProbe?.join()
                val options = explicit?.let(::listOf) ?: speechCandidates(
                    preferences.state.value.speechEngine, mutableState.value.unavailableOptions)
                if (options.isEmpty()) {
                    mutableState.update { it.copy(results = it.results + ((word to SpeechOption.SYSTEM) to
                        "No voice available. Retry a comparison button or check Settings.")) }
                }
                for (option in options) {
                    ensureActive()
                    mutableState.update { it.copy(activeOption = option) }
                    fun report(message: String) {
                        if (id == request) mutableState.update { it.copy(results = it.results + ((word to option) to message)) }
                    }
                    try {
                        performSpeech(word, option, id, ::report)
                        mutableState.update { it.copy(failedOptions = it.failedOptions - option) }
                        return@launch
                    } catch (_: TimeoutCancellationException) {
                        if (option == SpeechOption.SYSTEM) systemUnavailable("System voice timed out.")
                        report("Speech timed out. Tap to retry.")
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (error: Exception) {
                        if (error is SpeechOutputUnavailable) {
                            report(error.message ?: "Audio is busy. Try again.")
                            return@launch
                        }
                        if (option == SpeechOption.SYSTEM) systemUnavailable(error.message ?: "System voice failed.")
                        report(error.message ?: "Playback failed. Tap to retry.")
                    }
                    mutableState.update { it.copy(failedOptions = it.failedOptions + option) }
                }
            } finally {
                if (id == request) {
                    manager.abandonAudioFocusRequest(focus)
                    mutableState.value = mutableState.value.copy(activeWord = null, activeOption = null)
                }
            }
        }
    }

    private suspend fun performSpeech(word: String, option: SpeechOption, id: Long, report: (String) -> Unit) {
        if (option == SpeechOption.SYSTEM) {
            systemProbe?.join()
            report("Opening voice…")
            speakSystem(word, id, report)
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
            play(wav)
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

    private suspend fun speakSystem(word: String, id: Long, report: (String) -> Unit) {
        val engine = systemEngine()
        val completion = CompletableDeferred<Unit>()
        val started = CompletableDeferred<Unit>()
        val start = SystemClock.elapsedRealtime()
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                if (utteranceId != id.toString()) return
                started.complete(Unit)
                val elapsed = SystemClock.elapsedRealtime() - start
                scope.launch { if (id == request) report("$elapsed ms to speech") }
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
            withTimeout(30_000) { completion.await() }
        } finally { engine.stop() }
    }

    private fun requestFocus() {
        if (manager.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            throw SpeechOutputUnavailable()
        }
    }

    private class SpeechOutputUnavailable : IllegalStateException("Audio is busy. Try again after the current call or playback.")

    private suspend fun play(wav: File) {
        requestFocus()
        val player = MediaPlayer()
        try {
            suspendCancellableCoroutine { continuation ->
                player.setAudioAttributes(attributes)
                player.setDataSource(wav.path)
                player.setOnCompletionListener { if (continuation.isActive) continuation.resume(Unit) }
                player.setOnErrorListener { _, what, extra ->
                    if (continuation.isActive) continuation.resumeWithException(IllegalStateException("Audio playback error $what/$extra"))
                    true
                }
                player.setOnPreparedListener { if (continuation.isActive) it.start() }
                player.prepareAsync()
            }
        } finally { player.release() }
    }

    fun stop() {
        request++
        job?.cancel()
        job = null
        tts?.stop()
        manager.abandonAudioFocusRequest(focus)
        val current = mutableState.value
        val key = current.activeWord?.let { word -> current.activeOption?.let { word to it } }
        mutableState.value = current.copy(activeWord = null, activeOption = null,
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
