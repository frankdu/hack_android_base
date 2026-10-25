package dev.gemmabench.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.os.Build
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.security.MessageDigest

enum class SpeechOption(val label: String, val bytes: Long, val sha256: String, val url: String) {
    SYSTEM("Android", 0, "", ""),
    KOKORO("Kokoro", 92_361_116, "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx/model_quantized.onnx"),
    KITTEN("Kitten", 24_369_971, "f7b0afcbee92870b32b8e0276d855b954dc25470c9f051b376ac7eee537c76fc",
        "https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/84781d74e29ee25217551556398b42f80593a813/kitten_tts_nano_v0_8.onnx");

    fun file(context: Context) = File(context.filesDir, "tts/${name.lowercase()}.onnx")
}

/** A completed download is size/SHA-256 checked before atomic promotion. */
class SpeechModelStore(private val context: Context) {
    private companion object { val lock = Mutex() }
    suspend fun prepare(option: SpeechOption, progress: (Int) -> Unit): File = lock.withLock { withContext(Dispatchers.IO) {
        require(option != SpeechOption.SYSTEM)
        val target = option.file(context)
        if (target.isFile && target.length() == option.bytes) return@withContext target
        target.parentFile!!.mkdirs()
        val partial = File(target.path + ".part")
        val connection = URI(option.url).toURL().openConnection() as HttpURLConnection
        connection.connectTimeout = 20_000
        connection.readTimeout = 20_000
        try {
            check(connection.responseCode == 200) { "Download failed (HTTP ${connection.responseCode}). Tap to retry." }
            val digest = MessageDigest.getInstance("SHA-256")
            connection.inputStream.use { input ->
                partial.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var count = 0L
                    var lastPercent = -1
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        count += read
                        check(count <= option.bytes) { "Unexpected model size. Tap to retry." }
                        digest.update(buffer, 0, read)
                        output.write(buffer, 0, read)
                        val percent = (count * 100 / option.bytes).toInt()
                        if (percent != lastPercent) { progress(percent); lastPercent = percent }
                    }
                    check(count == option.bytes) { "Incomplete model download. Tap to retry." }
                }
            }
            check(digest.digest().joinToString("") { "%02x".format(it) } == option.sha256) {
                "Model verification failed. Tap to retry."
            }
            currentCoroutineContext().ensureActive()
            check(partial.renameTo(target)) { "Could not save model. Check free storage and retry." }
            target
        } finally {
            connection.disconnect()
            partial.delete()
        }
    } }
}

data class SpeechAudio(val samples: FloatArray, val loadMs: Long, val generationMs: Long,
                       val words: List<SpokenWord> = emptyList(), val exactTiming: Boolean = false) {
    val durationMs get() = samples.size * 1000L / 24_000
}

/** Serializes session ownership; only one model remains loaded to bound memory use. */
class LocalSpeech(private val context: Context) : AutoCloseable {
    private val pronunciation = EnglishPronunciation(context)
    private val environment by lazy {
        OrtEnvironment.getEnvironment().apply { setTelemetry(false) }
    }
    private val words by lazy {
        context.assets.open("tts/dictionary-tokens.json").bufferedReader().use {
            JsonParser.parseReader(it).asJsonObject.getAsJsonObject("words")
        }
    }
    private var session: OrtSession? = null
    private var loaded: SpeechOption? = null
    private var styles = FloatArray(0)
    private var closed = false

    @Synchronized fun generate(option: SpeechOption, word: String, file: File): SpeechAudio {
        check(!closed) { "Speech engine closed" }
        require(option != SpeechOption.SYSTEM)
        val entry = words.getAsJsonObject(word)
        val tokens = if (entry == null) pronunciation.tokenize(word, option) else null
        val ids = tokens?.ids ?: entry!!.getAsJsonArray(option.name.lowercase()).map { it.asLong }.toLongArray()
        require(ids.isNotEmpty() && ids.size <= 510) { "Unsupported pronunciation length" }
        val loadStart = SystemClock.elapsedRealtime()
        if (loaded != option) {
            session?.close()
            session = null
            loaded = null
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(2)
                options.setInterOpNumThreads(1)
                // ARM emulators can advertise SME2 although streaming SME is unavailable.
                if (Build.HARDWARE in setOf("ranchu", "goldfish")) {
                    options.addConfigEntry("mlas.disable_kleidiai", "1")
                }
                session = environment.createSession(file.path, options)
            }
            val data = context.assets.open("tts/${option.name.lowercase()}-voice.bin").use { it.readBytes() }
            val floats = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            styles = FloatArray(floats.remaining()).also(floats::get)
            loaded = option
        }
        val loadMs = SystemClock.elapsedRealtime() - loadStart
        val styleIndex = if (option == SpeechOption.KOKORO) ids.size else
            (tokens?.styleIndex ?: entry!!["kittenStyle"].asInt).coerceAtMost(styles.size / 256 - 1)
        val offset = styleIndex * 256
        require(offset + 256 <= styles.size) { "Invalid voice style" }
        val padded = longArrayOf(0) + ids + longArrayOf(0)
        val start = SystemClock.elapsedRealtime()
        var durations: LongArray? = null
        val audio = OnnxTensor.createTensor(environment, LongBuffer.wrap(padded), longArrayOf(1, padded.size.toLong())).use { input ->
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(styles.copyOfRange(offset, offset + 256)), longArrayOf(1, 256)).use { style ->
                OnnxTensor.createTensor(environment, FloatBuffer.wrap(floatArrayOf(if (option == SpeechOption.KITTEN) 0.8f else 1f)), longArrayOf(1)).use { speed ->
                    session!!.run(mapOf("input_ids" to input, "style" to style, "speed" to speed)).use { result ->
                        if (option == SpeechOption.KITTEN) {
                            val buffer = (result[1] as OnnxTensor).longBuffer
                            durations = LongArray(buffer.remaining()).also(buffer::get)
                        }
                        val buffer = (result[0] as OnnxTensor).floatBuffer
                        FloatArray(buffer.remaining()).also(buffer::get)
                    }
                }
            }
        }
        // Kitten's reference implementation removes its 5000-sample trailing padding.
        val samples = if (option == SpeechOption.KITTEN && audio.size > 5000) audio.copyOf(audio.size - 5000) else audio
        check(samples.isNotEmpty() && samples.all { it.isFinite() }) { "Speech synthesis produced invalid audio" }
        val durationMs = samples.size * 1000L / 24_000
        val alignment = durations?.takeIf { it.size == padded.size }?.let { frames ->
            tokens?.words?.map { (chars, phones) ->
                val begin = frames.take(phones.first + 1).sum()
                val end = begin + phones.sumOf { frames[it + 1] }
                // Kitten exports 40 Hz phoneme durations; trimming affects only the audio tail.
                SpokenWord(chars.first, chars.last + 1, (begin * 25).coerceAtMost(durationMs),
                    (end * 25).coerceAtMost(durationMs))
            }
        }
        return SpeechAudio(samples, loadMs, SystemClock.elapsedRealtime() - start,
            alignment ?: estimatedWordTimes(word, durationMs), alignment != null)
    }

    @Synchronized override fun close() {
        closed = true
        session?.close()
        session = null
        styles = FloatArray(0)
    }
}

fun writeSpeechWav(file: File, samples: FloatArray) {
    val dataSize = samples.size * 2
    val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        .put("RIFF".toByteArray()).putInt(36 + dataSize).put("WAVEfmt ".toByteArray())
        .putInt(16).putShort(1).putShort(1).putInt(24_000).putInt(48_000)
        .putShort(2).putShort(16).put("data".toByteArray()).putInt(dataSize).array()
    file.outputStream().buffered().use { output ->
        output.write(header)
        val pcm = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach { pcm.putShort((it.coerceIn(-1f, 1f) * 32767).toInt().toShort()) }
        output.write(pcm.array())
    }
}
