package dev.gemmabench.voice

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/** One bounded turn, held only in memory. Silence ends a turn; TTS never runs with the mic open. */
class VoiceRecorder(private val context: Context) : VoiceInput {
    @Volatile private var active: AudioRecord? = null
    private val done = AtomicBoolean(false)
    override fun finish() { done.set(true) }
    override fun cancel() { done.set(true); runCatching { active?.stop() } }

    override suspend fun capture(onLevel: (Float) -> Unit): ByteArray? = withContext(Dispatchers.IO) {
        check(context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            "Allow microphone access to talk with Gemma."
        }
        done.set(false)
        val minimum = AudioRecord.getMinBufferSize(16_000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0) { "This device has no available microphone." }
        val recorder = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, 16_000,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minimum, 6400))
        active = recorder
        try {
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "The microphone is busy. Please try again." }
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "Could not start the microphone." }
            val pcm = ByteArrayOutputStream()
            val frame = ShortArray(320)
            var samples = 0
            var voicedSamples = 0
            var lastVoice = 0
            while (!done.get() && samples < 16_000 * 20) {
                currentCoroutineContext().ensureActive()
                val count = recorder.read(frame, 0, frame.size)
                if (done.get()) break
                check(count > 0) { "Microphone interrupted. Tap Talk to try again." }
                val rms = sqrt((0 until count).sumOf { frame[it].toDouble() * frame[it] } / count)
                onLevel((rms / 4000).toFloat().coerceIn(0f, 1f))
                if (rms > 350) { voicedSamples += count; lastVoice = samples + count }
                for (i in 0 until count) { pcm.write(frame[i].toInt() and 255); pcm.write(frame[i].toInt() shr 8 and 255) }
                samples += count
                if (voicedSamples >= 4800 && samples - lastVoice > 22_400) break
                if (voicedSamples < 4800 && samples >= 128_000) break
            }
            currentCoroutineContext().ensureActive()
            if (voicedSamples < 4800) null else pcmWav(pcm.toByteArray())
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
            if (active === recorder) active = null
        }
    }
}

internal fun pcmWav(pcm: ByteArray): ByteArray = ByteBuffer.allocate(44 + pcm.size).order(ByteOrder.LITTLE_ENDIAN)
    .put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVEfmt ".toByteArray())
    .putInt(16).putShort(1).putShort(1).putInt(16_000).putInt(32_000)
    .putShort(2).putShort(16).put("data".toByteArray()).putInt(pcm.size).put(pcm).array()
