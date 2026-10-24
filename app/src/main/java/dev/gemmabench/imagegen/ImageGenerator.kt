package dev.gemmabench.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import com.google.gson.GsonBuilder
import java.io.File
import java.util.concurrent.CancellationException

class ImageGenerator(private val context: Context, private val store: ImageModelStore) {
    private val gson = GsonBuilder().setPrettyPrinting().create()
    private fun outputDirectory() = File(context.filesDir, "word-images").apply { mkdirs() }
    fun results(): List<ImageResult> = outputDirectory().listFiles().orEmpty()
        .filter { it.extension == "json" }.sortedByDescending { it.name }.take(24)
        .mapNotNull { file -> runCatching { gson.fromJson(file.readText(), ImageResult::class.java) }.getOrNull() }
        .filter { File(it.imagePath).isFile }

    fun generate(model: ImageModel, word: String, category: String, prompt: String, seed: Long,
        steps: Int, method: ImageMethod = ImageMethod.CPU, outputDirectory: File? = null,
        progress: (Int, Int) -> Unit): ImageResult {
        if (method == ImageMethod.TPU) return generateLiteRt(model, word, category, prompt, seed, steps, outputDirectory, progress)
        require(model.id in store.ready()) { "Download and verify all ${model.name} files first." }
        require(word.isNotBlank() && category.isNotBlank() && prompt.isNotBlank()) { "Enter a word, category and prompt." }
        require(prompt.length <= 2000) { "Use a prompt shorter than 2000 characters." }
        require(steps in 1..4 && seed >= 0) { "Steps must be 1–4 and seed must be a non-negative integer." }
        val totalStart = SystemClock.elapsedRealtime()
        val paths = model.files.map { store.file(it).absolutePath }
        val nativeStart = SystemClock.elapsedRealtime()
        var denoiseStart = nativeStart
        var denoiseEnd = nativeStart
        val pixels = DiffusionNative.generateOnBackend(paths[0], paths.getOrElse(1) { "" }, paths.getOrElse(2) { "" },
            prompt, model.flux, method == ImageMethod.GPU, steps, seed, Runtime.getRuntime().availableProcessors().coerceIn(1, 4)) { step, total, _ ->
                if (total == steps) {
                    if (step == 0) denoiseStart = SystemClock.elapsedRealtime()
                    if (step == total) denoiseEnd = SystemClock.elapsedRealtime()
                    progress(step, total)
                }
            } ?: throw CancellationException("Image generation stopped.")
        val nativeMs = SystemClock.elapsedRealtime() - nativeStart
        check(pixels.size == 512 * 512) { "Unexpected image dimensions from native runtime." }
        check(pixels.any { it != pixels.first() }) { "The backend returned a uniform blank image; this device/backend may be incompatible." }
        val timestamp = System.currentTimeMillis()
        val directory = outputDirectory?.apply { mkdirs() } ?: outputDirectory()
        val file = File(directory, "$timestamp-${model.id}-${method.name}.png")
        val bitmap = Bitmap.createBitmap(pixels, 512, 512, Bitmap.Config.ARGB_8888)
        try { file.outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) } }
        finally { bitmap.recycle() }
        val result = ImageResult(model.id, model.name, word.trim(), category.trim(), prompt, seed, steps,
            SystemClock.elapsedRealtime() - totalStart, nativeMs, highWaterRssKb(), file.absolutePath, timestamp,
            model.files.associate { it.filename to it.sha256 }, "${Build.MANUFACTURER} ${Build.MODEL} · ${Build.SOC_MODEL}",
            backend = if (method == ImageMethod.GPU) "Vulkan GPU denoiser (text + VAE on CPU)" else "ARM CPU",
            loadAndEncodeMs = denoiseStart - nativeStart, denoiseMs = (denoiseEnd - denoiseStart).coerceAtLeast(0),
            decodeAndSaveMs = (SystemClock.elapsedRealtime() - denoiseEnd).coerceAtLeast(0))
        File(directory, file.nameWithoutExtension + ".json").writeText(gson.toJson(result))
        return result
    }
    private fun generateLiteRt(model: ImageModel, word: String, category: String, prompt: String, seed: Long,
        steps: Int, outputDirectory: File?, progress: (Int, Int) -> Unit): ImageResult {
        require(word.isNotBlank() && category.isNotBlank() && prompt.isNotBlank() && prompt.length <= 2000)
        require(seed >= 0)
        val start = SystemClock.elapsedRealtime()
        val output = LiteRtImageExecutor(context).generate(model, prompt, seed, steps, progress)
        val inferenceMs = SystemClock.elapsedRealtime() - start
        check(output.pixels.any { it != output.pixels.first() }) { "LiteRT returned a uniform blank image." }
        val timestamp = System.currentTimeMillis()
        val directory = outputDirectory?.apply { mkdirs() } ?: outputDirectory()
        val file = File(directory, "$timestamp-${model.id}-TPU.png")
        val bitmap = Bitmap.createBitmap(output.pixels, 512, 512, Bitmap.Config.ARGB_8888)
        try { file.outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) } }
        finally { bitmap.recycle() }
        val result = ImageResult(model.id, model.name, word.trim(), category.trim(), prompt, seed, steps,
            SystemClock.elapsedRealtime() - start, inferenceMs, highWaterRssKb(), file.absolutePath, timestamp,
            output.hashes, "${Build.MANUFACTURER} ${Build.MODEL} · ${Build.SOC_MODEL}", backend = output.backend,
            runtime = "Google LiteRT 2.3.0 · word-image-litert-v1 · java.util.Random Gaussian noise",
            loadAndEncodeMs = output.encodeMs, denoiseMs = output.denoiseMs,
            decodeAndSaveMs = output.decodeMs + SystemClock.elapsedRealtime() - start - inferenceMs)
        File(directory, file.nameWithoutExtension + ".json").writeText(gson.toJson(result))
        return result
    }
    private fun highWaterRssKb(): Long = runCatching {
        File("/proc/self/status").useLines { lines ->
            lines.first { it.startsWith("VmHWM:") }.substringAfter(":").trim().substringBefore(" ").toLong()
        }
    }.getOrElse { Debug.getNativeHeapAllocatedSize() / 1024 }
}
