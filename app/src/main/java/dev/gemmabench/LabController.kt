package dev.gemmabench

import android.app.Application
import android.net.Uri
import android.util.Log
import dev.gemmabench.benchmark.*
import dev.gemmabench.data.CsvExport
import dev.gemmabench.data.LabStorage
import dev.gemmabench.runtime.LiteRtRuntime
import dev.gemmabench.download.*
import dev.gemmabench.photos.*
import com.google.gson.GsonBuilder
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
    val tpuSupport: TpuSupport = TpuSupport(),
    val engineRetained: Boolean = false,
    val downloads: List<ModelDownload> = emptyList(),
    val photos: PhotoDemoState = PhotoDemoState(),
    val photoBatchProgress: PhotoBatchProgress? = null,
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
    private val photoLibrary = PhotoLibrary(app)
    private val runner = BenchmarkRunner(runtime, storage::device, storage::snapshot)
    private val mutable = MutableStateFlow(LabState(busy = true))
    val state = mutable.asStateFlow()

    init {
        scope.launch {
            try {
                photoLibrary.clearPreviews()
                mutable.update { it.copy(
                    models = storage.refreshModelMetadata(), results = storage.results(), device = storage.device(),
                    tpuSupport = runtime.tpuSupport,
                    progress = "Ready · all inference stays on this device.",
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
                if (slot.tpu) {
                    require(runtime.tpuSupport.available) { runtime.tpuSupport.reason }
                    require(slot == runtime.tpuSupport.slot) { "Download the TPU model matching this device's Tensor generation." }
                }
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

    fun analyzePhotos(slot: ModelSlot, accelerator: Accelerator, batchSize: Int, resume: Boolean = false) {
        if (!begin("Preparing photo batch…", inference = true)) return
        runtime.resetStop()
        mutable.update { it.copy(engineRetained = false,
            photoBatchProgress = if (resume) it.photos.batchProgress else PhotoBatchProgress()) }
        scope.launch {
            try {
                require(batchSize in 1..PhotoDemoState.MAX_BATCH_SIZE) { "Batch size must be 1–${PhotoDemoState.MAX_BATCH_SIZE}." }
                check(PhotoPermissions.access(app) != PhotoAccess.DENIED) { "Grant photo access first." }
                val model = state.value.models.firstOrNull { it.slot == slot } ?: error("Download ${slot.label} from Models first.")
                model.unavailableReason(accelerator, runtime.tpuSupport, photos = true)?.let { error(it) }
                val metadata = model.metadata
                val contextTokens = metadata?.maxContextTokens?.takeIf { it > 0 }?.let {
                    if (metadata.dynamicContext) minOf(4096, it) else it
                } ?: 4096
                val settings = RunSettings(
                    accelerator = accelerator, workload = Workload.PROMPT, contextTokens = contextTokens,
                    outputTokens = minOf(1024, contextTokens - 320), repetitions = 1, warmups = 0, cooldownSeconds = 0, prompt = PhotoObjects.PROMPT,
                )
                settings.validate()
                validateBackend(model, settings)
                if (resume) {
                    require(state.value.photos.modelSha256 == model.sha256 && state.value.photos.textBackend == accelerator.name) {
                        "Resume with the same model and backend, or start over."
                    }
                } else {
                    val page = photoLibrary.nextBatch(state.value.photos.beforeId, batchSize)
                    photoLibrary.clearPreviews()
                    mutable.update { it.copy(photos = PhotoDemoState(
                        results = page.photos.map { photo -> PhotoResult(photo) },
                        beforeId = page.photos.lastOrNull()?.id ?: it.photos.beforeId,
                        hasMore = page.hasMore, batchNumber = it.photos.batchNumber + 1,
                        modelLabel = model.slot.label, modelSha256 = model.sha256,
                        modelSlot = slot, accessScope = PhotoPermissions.access(app),
                        textBackend = accelerator.name, batchSize = batchSize,
                        unsupportedSkipped = page.unsupportedSkipped,
                    )) }
                }
                mutable.update { it.copy(photoBatchProgress = it.photos.batchProgress) }
                val batch = state.value.photos.results
                for ((index, result) in batch.withIndex()) {
                    if (state.value.stopRequested) break
                    if (result.status != PhotoStatus.PENDING) continue
                    mutable.update { it.copy(progress = "Batch ${it.photos.batchNumber} · photo ${index + 1}/${batch.size} · ${result.photo.name}", output = "") }
                    updatePhoto(index) { it.copy(status = PhotoStatus.ANALYZING, error = null) }
                    var stage = "image_decode"
                    try {
                        val prepared = photoLibrary.prepare(result.photo)
                        updatePhoto(index) { it.copy(previewPath = prepared.file.absolutePath, width = prepared.width, height = prepared.height) }
                        if (state.value.stopRequested) break
                        val jpeg = prepared.file.readBytes()
                        val objects = analyzePhotoResponse(settings,
                            infer = { request ->
                                stage = "inference"
                                updatePhoto(index) { it.copy(attempts = it.attempts + 1, outputTokenLimit = request.outputTokens) }
                                runtime.analyzePhoto(model, request, jpeg) { text -> mutable.update { it.copy(output = text) } }
                            },
                            onMeasurement = { attempt, request, measurement ->
                                stage = "response_validation"
                                mutable.update { it.copy(photos = it.photos.copy(visionBackend = runtime.photoVisionBackend)) }
                                updatePhoto(index) { it.copy(rawOutput = measurement.output.orEmpty(),
                                    elapsedMs = (it.elapsedMs ?: 0.0) + measurement.elapsedMs, attempts = attempt,
                                    decodeTokens = measurement.metrics.decodeTokens, outputTokenLimit = request.outputTokens) }
                            },
                            onRetry = {
                                mutable.update { it.copy(progress = "Photo ${index + 1}/${batch.size} · retrying incomplete JSON with a shorter object list…") }
                                Log.w("GemmaPhotos", "photo=${index + 1}/${batch.size} model=${slot.name} text=$accelerator retry=malformed_json")
                            },
                        )
                        updatePhoto(index) { it.copy(status = PhotoStatus.COMPLETED, objects = objects) }
                    } catch (e: CancellationException) {
                        if (!state.value.stopRequested) throw e
                        break
                    } catch (e: Exception) {
                        if (state.value.stopRequested) break
                        val kind = if (e is PhotoResponseException) "model_json" else stage
                        val failed = state.value.photos.results[index]
                        Log.w("GemmaPhotos", "photo=${index + 1}/${batch.size} model=${slot.name} text=$accelerator stage=$kind " +
                            "attempts=${failed.attempts} decoded=${failed.decodeTokens} limit=${failed.outputTokenLimit} " +
                            "chars=${failed.rawOutput.length} exception=${e.javaClass.simpleName}")
                        updatePhoto(index) { it.copy(status = PhotoStatus.FAILED, errorKind = kind, error = e.message ?: e.javaClass.simpleName) }
                    }
                }
                mutable.update { it.copy(progress = when {
                    it.stopRequested -> "Photo analysis stopped. Resume this batch or start over."
                    batch.isEmpty() -> "No more accessible photos. Grant access to more photos or start over."
                    else -> "Batch ${it.photos.batchNumber} finished · ${it.photos.results.count { r -> r.status == PhotoStatus.COMPLETED }}/${batch.size} photos analyzed."
                }) }
            } catch (e: Exception) { fail(e) }
            finally {
                mutable.update { it.copy(photos = it.photos.copy(results = it.photos.results.map { r ->
                    if (r.status == PhotoStatus.ANALYZING) r.copy(status = PhotoStatus.PENDING) else r
                })) }
                // Keep photo memory bounded between batches; the runtime disk cache is retained.
                runtime.unload()
                mutable.update { it.copy(engineRetained = false) }
                finish()
            }
        }
    }

    private fun updatePhoto(index: Int, transform: (PhotoResult) -> PhotoResult) {
        mutable.update { state ->
            val photos = state.photos.copy(results = state.photos.results.mapIndexed { i, result ->
                if (i == index) transform(result) else result
            })
            state.copy(photos = photos, photoBatchProgress = photos.batchProgress)
        }
    }

    fun resetPhotos() {
        if (!begin("Photo demo reset. Ready for the newest accessible photos.")) return
        scope.launch {
            try {
                photoLibrary.clearPreviews()
                mutable.update { it.copy(photos = PhotoDemoState()) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun exportPhotos(uri: Uri) {
        if (!begin("Exporting photo object results…")) return
        scope.launch {
            try {
                val photos = state.value.photos
                val report = mapOf(
                    "model" to photos.modelLabel, "sha256" to photos.modelSha256,
                    "textBackend" to photos.textBackend, "visionBackend" to photos.visionBackend,
                    "batch" to photos.batchNumber, "batchSize" to photos.batchSize,
                    "unsupportedImagesSkipped" to photos.unsupportedSkipped,
                    "coordinates" to "Estimated boxes [left, top, right, bottom], 0..1000, origin top-left of upright image",
                    "photos" to photos.results.map { result -> mapOf(
                        "photo" to result.photo, "status" to result.status,
                        "uprightInputWidth" to result.width, "uprightInputHeight" to result.height,
                        "objects" to result.objects, "elapsedMs" to result.elapsedMs,
                        "rawOutput" to result.rawOutput, "error" to result.error,
                        "attempts" to result.attempts, "decodeTokens" to result.decodeTokens,
                        "outputTokenLimit" to result.outputTokenLimit, "errorKind" to result.errorKind,
                    ) },
                )
                app.contentResolver.openOutputStream(uri, "wt").use {
                    requireNotNull(it) { "Cannot write to that destination." }
                    it.write(GsonBuilder().setPrettyPrinting().create().toJson(report).toByteArray(Charsets.UTF_8))
                }
                mutable.update { it.copy(progress = "Exported ${photos.results.size} photo result(s).") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun stop() {
        if (!state.value.runningInference) return
        mutable.update { it.copy(stopRequested = true, progress = "Stopping… Waiting for the active native call to return.") }
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
        model.unavailableReason(settings.accelerator, runtime.tpuSupport)?.let { error(it) }
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
    }

    private fun finish() { mutable.update { it.copy(busy = false, runningInference = false, photoBatchProgress = null) } }
    private fun fail(e: Exception) {
        mutable.update { it.copy(error = e.message ?: e.javaClass.simpleName, progress = "Action failed. Details below.") }
    }
}
