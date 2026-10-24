package dev.gemmabench.imagegen

import android.os.SystemClock
import com.google.gson.JsonParser
import dev.gemmabench.imagegen.bonsai.BonsaiMath
import dev.gemmabench.imagegen.iris.QwenTokenizer
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/** Host ABI for litert-community/FLUX.2-klein-4B-LiteRT revision 223f4c9 (256px). */
object CommunityFlux {
    fun generate(hostJson: String, dir: File, prompt: String, seed: Long, steps: Int, method: ImageMethod,
        progress: (Int, Int) -> Unit,
        run: (String, List<FloatArray>, Int) -> List<FloatArray>): PortableImagePipeline.Output {
        require(steps == 4)
        val start = SystemClock.elapsedRealtime()
        val host = JsonParser.parseString(hostJson).asJsonObject
        require(host["format"].asString == "community-flux-host-v1" && host["width"].asInt == 256)
        fun array(name: String) = host.getAsJsonArray(name).map { it.asFloat }.toFloatArray()
        val tokenizer = QwenTokenizer(File(dir, "tokenizer"), "qwen_special.txt")
        val raw = tokenizer.encode("<|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        val ids = IntArray(512) { if (it < raw.size) raw[it] else 151643 }
        val real = raw.size.coerceAtMost(512)
        var hidden = embeddings(File(dir, "tokenizer/qwen_embed_fp16.bin"), ids)
        val mask = encoderMask(real)
        val (encCos, encSin) = encoderRope()
        val taps = ArrayList<FloatArray>()
        repeat(3) { i ->
            hidden = run("ke_enc$i.tflite", listOf(hidden, mask, encCos, encSin), 1).single()
            require(hidden.size == 512 * 2560)
            taps.add(hidden)
        }
        val context = FloatArray(512 * 7680)
        for (token in 0 until 512) for (tap in 0..2)
            System.arraycopy(taps[tap], token * 2560, context, token * 7680 + tap * 2560, 2560)
        taps.clear()
        val encodeEnd = SystemClock.elapsedRealtime()
        val (jointCos, jointSin) = jointRope()
        val random = Random(seed)
        val latent = FloatArray(256 * 128) { random.nextGaussian().toFloat() }
        val sigmas = array("sigmas")
        repeat(steps) { step ->
            progress(step, steps)
            val temb = host.getAsJsonArray("temb")[step].asJsonArray.map { it.asFloat }.toFloatArray()
            val prep = run("kc_prep.tflite", listOf(latent, context, temb), 5)
            var image = prep[0]; var text = prep[1]
            repeat(2) { i ->
                val outputs = run("kc_double$i.tflite", listOf(image, text, jointCos, jointSin, prep[2], prep[3]), 2)
                image = outputs[0]; text = outputs[1]
            }
            var joint = text + image
            repeat(4) { i -> joint = run("kc_single$i.tflite", listOf(joint, jointCos, jointSin, prep[4]), 1).single() }
            val v = run("kc_final.tflite", listOf(joint, temb), 1).single()
            require(v.size == latent.size)
            val ds = sigmas[step + 1] - sigmas[step]
            for (i in latent.indices) latent[i] += ds * v[i]
        }
        val denoiseEnd = SystemClock.elapsedRealtime(); progress(steps, steps)
        val decoded = run("kv_vae.tflite", listOf(BonsaiMath.unpatchify(latent, array("bnScale"), array("bnShift"), 16)), 1).single()
        return PortableImagePipeline.Output(PortableImagePipeline.pixels(decoded, 256),
            "LiteRT ${if (method == ImageMethod.GPU) "FP32 GPU" else "CPU (4 threads)"} + CPU host",
            encodeEnd - start, denoiseEnd - encodeEnd, SystemClock.elapsedRealtime() - denoiseEnd)
    }

    fun encoderMask(real: Int): FloatArray {
        require(real in 1..512)
        // Pre-expand across all 32 heads: the published graphs require this layout.
        val out = FloatArray(32 * 512 * 512)
        for (head in 0 until 32) for (q in 0 until 512) for (k in 0 until 512)
            if (k > q || k >= real) out[(head * 512 + q) * 512 + k] = -Float.MAX_VALUE
        return out
    }

    fun encoderRope(): Pair<FloatArray, FloatArray> {
        val c = FloatArray(512 * 128); val s = FloatArray(c.size)
        for (pos in 0 until 512) for (j in 0 until 64) {
            val angle = pos / 1_000_000.0.pow(j / 64.0)
            for (half in 0..1) {
                c[pos * 128 + half * 64 + j] = cos(angle).toFloat()
                s[pos * 128 + half * 64 + j] = sin(angle).toFloat()
            }
        }
        return c to s
    }

    fun jointRope(): Pair<FloatArray, FloatArray> {
        val c = FloatArray(768 * 64); val s = FloatArray(c.size)
        for (token in 0 until 768) {
            val pos = if (token < 512) intArrayOf(0, 0, 0, token)
                else intArrayOf(0, (token - 512) / 16, (token - 512) % 16, 0)
            for (axis in 0..3) for (j in 0 until 16) {
                val angle = pos[axis] / 2000.0.pow(j / 16.0)
                c[token * 64 + axis * 16 + j] = cos(angle).toFloat()
                s[token * 64 + axis * 16 + j] = sin(angle).toFloat()
            }
        }
        return c to s
    }

    private fun embeddings(file: File, ids: IntArray): FloatArray = RandomAccessFile(file, "r").use { input ->
        val table = input.channel.map(FileChannel.MapMode.READ_ONLY, 0, input.length()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        FloatArray(ids.size * 2560) { i -> halfToFloat(table.get(ids[i / 2560] * 2560 + i % 2560)) }
    }

    fun halfToFloat(value: Short): Float {
        val bits = value.toInt() and 65535
        val sign = if (bits and 32768 != 0) -1f else 1f
        val exponent = (bits ushr 10) and 31; val fraction = bits and 1023
        return when (exponent) {
            0 -> sign * fraction * 2.0.pow(-24).toFloat()
            31 -> if (fraction == 0) sign * Float.POSITIVE_INFINITY else Float.NaN
            else -> Float.fromBits(((bits and 32768) shl 16) or ((exponent + 112) shl 23) or (fraction shl 13))
        }
    }
}
