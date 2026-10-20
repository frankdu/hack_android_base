package dev.gemmabench

import android.app.Application
import android.net.Uri
import android.os.Build
import dev.gemmabench.benchmark.*
import dev.gemmabench.data.CsvExport
import dev.gemmabench.data.LabStorage
import dev.gemmabench.runtime.LiteRtRuntime
import dev.gemmabench.download.*
import java.util.concurrent.Executors
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class GemmaBenchApplication : Application() {
    val lab by lazy { LabController(this) }
}

data class LabState(
    val models: List<ModelArtifact> = emptyList(),
    val results: List<BenchmarkResult> = emptyList(),
    val current: BenchmarkResult? = null,
    val busy: Boolean = false,
    val runningInference: Boolean = false,
    val stopRequested: Boolean = false,
    val progress: String = "Download a model to begin.",
    val output: String = "",
    val error: String? = null,
    val device: DeviceInfo? = null,
    val tensorLibraries: Boolean = false,
    val engineRetained: Boolean = false,
    val downloads: List<ModelDownload> = emptyList(),
) {
    val downloadsActive: Boolean get() = downloads.any { it.active }
    val unavailable: Boolean get() = busy || downloadsActive
}

/** Application-owned worker survives rotation. JNI resources are only closed on this worker. */
class LabController(private val app: Application) {
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "gemma-lab") }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + worker)
    private val storage = LabStorage(app)
    private val downloads = SystemModelDownloads(app)
    private val runtime = LiteRtRuntime(app, storage::file)
    private val runner = BenchmarkRunner(runtime, storage::device, storage::snapshot)
    private val mutable = MutableStateFlow(LabState(busy = true))
    val state = mutable.asStateFlow()

    init {
        scope.launch {
            try {
                mutable.update { it.copy(
                    models = storage.models(), results = storage.results(), device = storage.device(),
                    tensorLibraries = runtime.tensorLibrariesAvailable, progress = "Ready · all inference stays on this device.",
                    downloads = downloads.statuses(),
                ) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
            while (isActive) {
                try { refreshDownloads() } catch (e: Exception) { fail(e) }
                delay(1000)
            }
        }
    }

    private fun begin(message: String, inference: Boolean = false): Boolean {
        if (mutable.value.unavailable) return false
        mutable.update { it.copy(
            busy = true, runningInference = inference, stopRequested = false,
            progress = message, error = null, current = null, output = "",
        ) }
        return true
    }

    fun downloadModel(slot: ModelSlot, wifiOnly: Boolean) {
        // More than one system download may run; native model operations stay serialized.
        if (state.value.busy || state.value.downloads.any { it.spec.slot == slot && it.active }) return
        mutable.update { it.copy(busy = true, error = null, progress = "Starting ${slot.label} download…") }
        scope.launch {
            try {
                runtime.unload()
                downloads.enqueue(ModelCatalog.forSlot(slot), wifiOnly)
                mutable.update { it.copy(downloads = downloads.statuses(), engineRetained = false,
                    progress = "Model download started. You can leave the app; Android manages the transfer.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun cancelDownload(id: Long) {
        val transfer = state.value.downloads.firstOrNull { it.id == id } ?: return
        if (transfer.phase == TransferPhase.VERIFYING || state.value.busy) return
        mutable.update { it.copy(downloads = it.downloads.map { d ->
            if (d.id == id) d.copy(phase = TransferPhase.CANCELING, detail = "Canceling…") else d
        }) }
        scope.launch {
            try {
                downloads.cancel(id)
                mutable.update { it.copy(downloads = downloads.statuses(), progress = "Download canceled.") }
            } catch (e: Exception) { fail(e) }
        }
    }

    private fun refreshDownloads() {
        if (state.value.busy) return
        var statuses = downloads.statuses()
        mutable.update { it.copy(downloads = statuses) }
        for (transfer in statuses.filter { it.phase == TransferPhase.VERIFYING }) {
            mutable.update { it.copy(busy = true, engineRetained = false, progress = "Verifying ${transfer.spec.slot.label}…") }
            try {
                runtime.unload()
                storage.installDownload(transfer.spec, downloads.temporaryFile(transfer.spec)) { checked ->
                    mutable.update { it.copy(downloads = it.downloads.map { d ->
                        if (d.id == transfer.id) d.copy(detail = "Verifying checksum · ${checked / (1024 * 1024)} MiB checked") else d
                    }) }
                }
                mutable.update { it.copy(models = storage.models(), progress = "${transfer.spec.slot.label} is ready to benchmark.") }
                downloads.installed(transfer.id)
            } catch (e: Exception) {
                downloads.markFailed(transfer.id, e.message ?: "Could not verify the model. Tap Retry.")
                mutable.update { it.copy(progress = "Model verification failed. Retry from Models.") }
            } finally {
                statuses = downloads.statuses()
                mutable.update { it.copy(busy = false, downloads = statuses) }
            }
        }
    }

    fun importModel(slot: ModelSlot, uri: Uri) {
        if (!begin("Preparing local model copy…")) return
        scope.launch {
            try {
                runtime.unload()
                storage.importModel(slot, uri) { message -> mutable.update { it.copy(progress = message) } }
                mutable.update { it.copy(models = storage.models(), progress = "${slot.label} imported and fingerprinted.", engineRetained = false) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun importStaged() {
        if (!begin("Checking files staged by ADB…")) return
        scope.launch {
            try {
                runtime.unload()
                val imported = storage.importStaged { message -> mutable.update { it.copy(progress = message) } }
                mutable.update { it.copy(
                    models = storage.models(), engineRetained = false,
                    progress = if (imported.isEmpty()) "No staged models found. See scripts/push-model.sh." else "Imported ${imported.size} staged model(s).",
                ) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun run(slots: Set<ModelSlot>, settings: RunSettings) {
        if (!begin("Preparing run…", inference = true)) return
        runner.reset()
        scope.launch {
            try {
                settings.validate()
                val models = slots.sortedBy { it.ordinal }.map { slot ->
                    state.value.models.firstOrNull { it.slot == slot }
                        ?: error("Download ${slot.label} from Models first.")
                }
                require(models.isNotEmpty()) { "Select at least one model." }
                for (model in models) {
                    if (state.value.stopRequested) break
                    val result = try {
                        validateBackend(model, settings)
                        runner.run(
                            model, settings,
                            onProgress = { message -> mutable.update { it.copy(progress = message, output = "") } },
                            onText = { text -> mutable.update { it.copy(output = text) } },
                            onSample = { result -> mutable.update { it.copy(current = result) } },
                        )
                    } catch (e: Exception) {
                        BenchmarkResult(
                            startedAt = java.time.Instant.now().toString(), model = model,
                            settings = settings, device = storage.device(), samples = emptyList(),
                            status = "failed", error = e.message ?: e.javaClass.simpleName,
                        )
                    }
                    val updated = (listOf(result) + state.value.results).take(100)
                    storage.saveResults(updated)
                    mutable.update { it.copy(results = updated, current = result) }
                    if (result.status == "failed") {
                        runtime.unload()
                        mutable.update { it.copy(error = result.error) }
                    }
                }
                mutable.update { it.copy(
                    progress = if (it.stopRequested) "Stopped · completed samples saved." else "Finished · results saved on device.",
                    engineRetained = settings.workload == Workload.PROMPT && !it.stopRequested && it.current?.status == "completed",
                ) }
            } catch (e: Exception) { fail(e) }
            finally {
                if (state.value.stopRequested) {
                    runtime.unload()
                    mutable.update { it.copy(engineRetained = false) }
                }
                finish()
            }
        }
    }

    fun stop() {
        if (!state.value.runningInference) return
        mutable.update { it.copy(stopRequested = true, progress = "Stopping… Fixed-token tests stop after the current native call.") }
        try { runner.stop() } catch (e: Exception) { fail(e) }
    }

    fun unload() {
        if (!begin("Releasing model memory…")) return
        scope.launch {
            try {
                runtime.unload()
                mutable.update { it.copy(engineRetained = false, progress = "Model unloaded. Runtime cache files are retained.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun export(uri: Uri, csv: Boolean) {
        if (!begin("Exporting local results…")) return
        scope.launch {
            try {
                val text = if (csv) CsvExport.encode(state.value.results) else storage.exportJson(state.value.results)
                app.contentResolver.openOutputStream(uri, "wt").use {
                    requireNotNull(it) { "Cannot write to that destination." }
                    it.write(text.toByteArray(Charsets.UTF_8))
                }
                mutable.update { it.copy(progress = "Exported ${state.value.results.size} result(s).") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    private fun validateBackend(model: ModelArtifact, settings: RunSettings) {
        require(model.slot.tpu == (settings.accelerator == Accelerator.TPU)) {
            "CPU/GPU require standard models. TPU requires the matching E2B Tensor variant."
        }
        model.metadata?.let { metadata ->
            val backendName = if (settings.accelerator == Accelerator.TPU) "NPU" else settings.accelerator.name
            require(metadata.textBackends.isEmpty() || backendName in metadata.textBackends) {
                "This artifact declares ${metadata.textBackends.joinToString()} support, not $backendName. Import a compatible variant from the official files."
            }
            require(!settings.speculativeDecoding || metadata.supportsSpeculativeDecoding) {
                "This artifact does not contain a speculative-decoding drafter. Use the baseline or import a compatible variant."
            }
            if (settings.workload == Workload.PROMPT && metadata.maxContextTokens > 0) {
                require(settings.contextTokens <= metadata.maxContextTokens) {
                    "This artifact supports at most ${metadata.maxContextTokens} context tokens."
                }
                require(metadata.dynamicContext || settings.contextTokens == metadata.maxContextTokens) {
                    "This artifact has a fixed ${metadata.maxContextTokens}-token context; set the prompt context to that value."
                }
            }
        }
        if (settings.accelerator == Accelerator.TPU) {
            require(runtime.tensorLibrariesAvailable) { "Bundle matching Tensor native libraries before using TPU. See docs/BENCHMARKING.md." }
            val soc = Build.SOC_MODEL.uppercase().replace(" ", "").replace("_", "")
            val target = if (model.slot == ModelSlot.E2B_G5) "G5" else "G6"
            require(soc.contains(target)) { "This artifact targets Tensor $target; this device reports ${Build.SOC_MODEL}." }
        }
    }

    private fun finish() { mutable.update { it.copy(busy = false, runningInference = false) } }
    private fun fail(e: Exception) {
        mutable.update { it.copy(error = e.message ?: e.javaClass.simpleName, progress = "Action failed. Details below.") }
    }
}
