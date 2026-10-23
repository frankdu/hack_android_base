package dev.gemmabench.data

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.StatFs
import android.provider.OpenableColumns
import android.util.AtomicFile
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import com.google.ai.edge.litertlm.ModelInfo
import com.google.ai.edge.litertlm.ModelType
import com.google.ai.edge.litertlm.Modality
import dev.gemmabench.BuildConfig
import dev.gemmabench.benchmark.*
import dev.gemmabench.download.DownloadSpec
import dev.gemmabench.download.ModelFileVerifier
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest

class LabStorage(private val context: Context) {
    val gson = GsonBuilder().setPrettyPrinting().create()
    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
    private val manifest = File(context.filesDir, "models.json")
    private val resultsFile = File(context.filesDir, "results.json")

    private fun downloadedModelsDir(): File = requireNotNull(context.getExternalFilesDir("downloaded-models")) {
        "Downloaded model storage is unavailable."
    }

    fun file(model: ModelArtifact) = File(if (model.downloaded) downloadedModelsDir() else modelsDir, model.filename)

    fun models(): List<ModelArtifact> = readList<ModelArtifact>(manifest)
        .filter { it.filename == it.slot.filename && file(it).length() == it.bytes && it.bytes > 0 }

    fun results(): List<BenchmarkResult> = readList(resultsFile)

    /** Upgrade existing manifests so pickers can distinguish text-only and multimodal artifacts. */
    fun refreshModelMetadata(): List<ModelArtifact> {
        val installed = models()
        val updated = installed.map { model ->
            if (model.metadata?.visionSupported == null || model.metadata.audioSupported == null)
                model.copy(metadata = inspectModel(file(model))) else model
        }
        if (updated != installed) writeJson(manifest, updated)
        return updated
    }

    fun saveResults(results: List<BenchmarkResult>) = writeJson(resultsFile, results.take(100))

    /** Verify and promote in place. There is no second multi-GB import copy. */
    fun installDownload(spec: DownloadSpec, source: File, onProgress: (Long) -> Unit): ModelArtifact {
        val destination = File(downloadedModelsDir(), spec.slot.filename)
        // Recover a process exit after rename but before the private manifest was committed.
        val candidate = if (source.isFile) source else destination
        val hash = ModelFileVerifier.verify(candidate, spec.bytes, spec.sha256, onProgress)
        val metadata = inspectModel(candidate)
        val artifact = ModelArtifact(spec.slot, spec.slot.filename, spec.bytes, hash, metadata, downloaded = true)
        if (candidate != destination) check(candidate.renameTo(destination)) { "Could not install the downloaded model." }
        writeJson(manifest, models().filterNot { it.slot == spec.slot } + artifact)
        return artifact
    }

    fun importModel(slot: ModelSlot, uri: Uri, onProgress: (String) -> Unit): ModelArtifact {
        val resolver = context.contentResolver
        var size: Long? = null
        resolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use {
            if (it.moveToFirst() && !it.isNull(0)) size = it.getLong(0)
        }
        if (size != null) check(StatFs(modelsDir.absolutePath).availableBytes > size + 128L * 1024 * 1024) {
            "Not enough space for a local copy of this model."
        }
        return resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open the selected file." }
            copyModel(slot, input, onProgress)
        }
    }

    /** ADB staging uses app external storage; imported artifacts live in private storage. */
    fun importStaged(onProgress: (String) -> Unit): List<ModelArtifact> {
        val staging = context.getExternalFilesDir("models") ?: return emptyList()
        val imported = mutableListOf<ModelArtifact>()
        ModelSlot.entries.forEach { slot ->
            val source = File(staging, slot.filename)
            if (source.isFile) {
                imported += FileInputStream(source).use { copyModel(slot, it, onProgress) }
                // Leave the staging file under user control; it can be removed after a successful import.
            }
        }
        return imported
    }

    private fun copyModel(slot: ModelSlot, input: java.io.InputStream, onProgress: (String) -> Unit): ModelArtifact {
        val temporary = File(modelsDir, "${slot.filename}.part")
        val digest = MessageDigest.getInstance("SHA-256")
        var copied = 0L
        var lastReport = 0L
        try {
            FileOutputStream(temporary).use { out ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    out.write(buffer, 0, count)
                    digest.update(buffer, 0, count)
                    copied += count
                    if (copied - lastReport >= 32L * 1024 * 1024) {
                        onProgress("Importing ${slot.label} · ${copied / (1024 * 1024)} MiB")
                        lastReport = copied
                    }
                }
                out.fd.sync()
            }
            check(copied > 0) { "The selected model is empty." }
            onProgress("Checking ${slot.label} model metadata…")
            val metadata = inspectModel(temporary)
            val artifact = ModelArtifact(slot, slot.filename, copied,
                digest.digest().joinToString("") { "%02x".format(it) }, metadata)
            check(temporary.renameTo(file(artifact))) { "Could not finish the model import." }
            val updated = models().filterNot { it.slot == slot } + artifact
            writeJson(manifest, updated)
            return artifact
        } finally {
            temporary.delete()
        }
    }

    private fun inspectModel(file: File): ArtifactMetadata = try {
        ModelInfo.from(file.absolutePath).use { info ->
            require(info.modelType == ModelType.LLM) { "Choose a generative .litertlm model, not an embedding model." }
            ArtifactMetadata(
                info.supportedBackends(Modality.TEXT).map { it.name }, info.maxContextTokens(),
                info.isDynamicContext(), info.minRuntimeVersion(),
                (info as? ModelInfo.Llm)?.hasSpeculativeDecodingSupport() == true,
                info.inputModalities().vision, info.supportedBackends(Modality.VISION).map { it.name },
                info.inputModalities().audio, info.supportedBackends(Modality.AUDIO).map { it.name },
            )
        }
    } catch (e: LinkageError) {
        throw IllegalStateException("Cannot load the ARM64 LiteRT-LM native library: ${e.message}", e)
    }

    fun device(): DeviceInfo {
        val info = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(info)
        return DeviceInfo(
            Build.MANUFACTURER, Build.MODEL, Build.SOC_MODEL, Build.VERSION.RELEASE,
            Build.VERSION.SDK_INT, Build.FINGERPRINT, Build.SUPPORTED_ABIS.toList(),
            info.totalMem / (1024 * 1024), BuildConfig.VERSION_NAME, BuildConfig.LITERT_LM_VERSION,
        )
    }

    fun snapshot(): EnvironmentSnapshot {
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val temp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        val process = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        return EnvironmentSnapshot(
            context.getSystemService(PowerManager::class.java).currentThermalStatus,
            if (level >= 0 && scale > 0) level * 100 / scale else null,
            if (temp == Int.MIN_VALUE) null else temp / 10.0,
            (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0,
            process.totalPss / 1024.0,
            memory.availMem / (1024 * 1024),
        )
    }

    fun exportJson(results: List<BenchmarkResult>): String = gson.toJson(mapOf(
        "schemaVersion" to 1,
        "exportedAt" to java.time.Instant.now().toString(),
        "tokenCounts" to "native LiteRT-LM; no character estimates",
        "results" to results,
    ))

    private inline fun <reified T> readList(file: File): List<T> {
        if (!file.exists()) return emptyList()
        return try {
            AtomicFile(file).openRead().bufferedReader().use {
                gson.fromJson<List<T>>(it, object : TypeToken<List<T>>() {}.type) ?: emptyList()
            }
        } catch (e: Exception) {
            // Preserve malformed files rather than overwriting them silently.
            throw IllegalStateException("Cannot read ${file.name}: ${e.message}", e)
        }
    }

    private fun writeJson(file: File, value: Any) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            stream.write(gson.toJson(value).toByteArray(Charsets.UTF_8))
            atomic.finishWrite(stream)
        } catch (e: Exception) {
            atomic.failWrite(stream)
            throw e
        }
    }
}
