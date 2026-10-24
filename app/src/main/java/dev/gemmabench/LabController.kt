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
import dev.gemmabench.people.*
import dev.gemmabench.settings.*
import dev.gemmabench.voice.voiceContextTokens
import dev.gemmabench.imagegen.*
import com.google.gson.GsonBuilder
import java.util.concurrent.Executors
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class GemmaBenchApplication : Application() {
    val preferences by lazy { AppPreferencesStore(this) }
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
    val people: PeopleState = PeopleState(),
    val preferredLlmFamily: ModelSlot? = null,
    val images: ImageLabState = ImageLabState(),
    val imageGenerating: Boolean = false,
    val imageBenchmark: ImageBenchmarkState = ImageBenchmarkState(),
    val imageBenchmarkRunning: Boolean = false,
) {
    val defaultLlm: LocalLlmChoice? get() = selectLocalLlm(models, tpuSupport, preferredLlmFamily)
    val downloadsActive: Boolean get() = downloads.any { it.active }
    val unavailable: Boolean get() = busy || downloadsActive || images.downloading
}

/** Application-owned worker survives rotation. JNI resources are only closed on this worker. */
class LabController(private val app: Application) {
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "gemma-lab") }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + worker)
    private val storage = LabStorage(app)
    private val downloads = SystemModelDownloads(app)
    private val runtime = LiteRtRuntime(app, storage::file)
    private val imageModels = ImageModelStore(app)
    private val imageGenerator = ImageGenerator(app, imageModels)
    private val imageBenchmarkStore = ImageBenchmarkStore(app)
    private val imageBenchmarkClient = ImageBenchmarkClient(app)
    private val imageBenchmarkRunner = ImageBenchmarkRunner(imageBenchmarkStore, { run, sample ->
        val started = android.os.SystemClock.elapsedRealtime()
        val result = imageBenchmarkClient.generate(sample, imageBenchmarkStore.images(run.id), run.timeoutMinutes, retainWorker = run.retainsEngine) { step, steps ->
            mutable.update { it.copy(imageBenchmark = it.imageBenchmark.copy(step = step, steps = steps),
                progress = if (step >= steps) "Benchmark · decoding ${sample.input.word} · ${sample.method.label}"
                    else "Benchmark · ${sample.input.word} · ${sample.method.label} · step $step / $steps") }
        }
        result to (android.os.SystemClock.elapsedRealtime() - started)
    }, imageBenchmarkClient::stop, releaseCombination = imageBenchmarkClient::release)
    private val photoLibrary = PhotoLibrary(app)
    private val photoCache by lazy { PhotoAnalysisCache(PhotoCacheDatabase.getInstance(app).photoAnalysis()) }
    private val peopleRepository by lazy { PeopleRepository(PhotoCacheDatabase.getInstance(app)) }
    private val faceEngine by lazy { FaceEngine(app) }
    private val peopleRefreshGeneration = java.util.concurrent.atomic.AtomicInteger()
    private val runner = BenchmarkRunner(runtime, storage::device, storage::snapshot)
    private val mutable = MutableStateFlow(LabState(busy = true))
    val state = mutable.asStateFlow()

    init {
        scope.launch {
            (app as GemmaBenchApplication).preferences.state.collect { preferences ->
                mutable.update { it.copy(preferredLlmFamily = preferences.llmFamily) }
            }
        }
        scope.launch {
            try {
                photoLibrary.clearPreviews()
                mutable.update { it.copy(
                    models = storage.refreshModelMetadata(), results = storage.results(), device = storage.device(),
                    tpuSupport = runtime.tpuSupport,
                    progress = "Ready · all inference stays on this device.",
                    downloads = downloads.statuses(),
                    images = ImageLabState(ready = imageModels.ready(), results = imageGenerator.results()),
                    imageBenchmark = ImageBenchmarkState(runs = imageBenchmarkStore.runs(recover = true), methods = imageMethodSupport(app)),
                ) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
            while (isActive) {
                try { refreshDownloads() } catch (e: Exception) { fail(e) }
                try { refreshImageDownloads() } catch (e: Exception) { fail(e) }
                delay(1000)
            }
        }
    }

    private fun updateImageBenchmark(run: ImageBenchmarkRun) {
        mutable.update { it.copy(imageBenchmark = it.imageBenchmark.copy(
            runs = (listOf(run) + it.imageBenchmark.runs.filterNot { old -> old.id == run.id }).sortedByDescending { r -> r.startedAtEpochMs },
            step = 0, steps = 0), progress = "Image benchmark · ${run.successful} generated · ${run.failed} failed · ${run.pending} remaining") }
    }

    fun startImageBenchmark(modelIds: Set<String>, methods: Set<ImageMethod>, timeoutMinutes: Int) {
        if (!begin("Preparing 30-word image benchmark…", inference = true)) return
        imageBenchmarkRunner.resetStop()
        imageBenchmarkClient.resetStop()
        mutable.update { it.copy(imageBenchmarkRunning = true) }
        scope.launch {
            try {
                val selected = ImageModels.all.filter { it.id in modelIds }
                require(selected.isNotEmpty()) { "Select at least one model." }
                val available = imageMethodSupport(app)
                val requested = available.filter { it.method in methods }
                require(requested.any { it.available && it.method in methods }) { "Select an available image generation method." }
                require(selected.any { model -> requested.any { it.method in methods && it.supports(model.id) } }) {
                    "Install at least one selected model/backend combination first."
                }
                runtime.unload()
                val now = System.currentTimeMillis()
                val plan = newImageBenchmarkRun(java.util.UUID.randomUUID().toString(), now,
                    "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · ${android.os.Build.SOC_MODEL} · ${android.os.Build.FINGERPRINT}",
                    selected, requested, timeoutMinutes, includeUnavailable = selected.all { it.pipeline != ImagePipeline.NATIVE })
                val bundleHashes = plan.samples.filter { it.status == ImageSampleStatus.PENDING &&
                    (it.method == ImageMethod.TPU || ImageModels.get(it.modelId).pipeline == ImagePipeline.LITERT_PLAN) }
                    .distinctBy { it.modelId to it.method }.associate { sample ->
                        val dir = LiteRtImageBundles(app, portable = sample.method != ImageMethod.TPU).load(sample.modelId).first
                        (sample.modelId to sample.method) to LiteRtImageBundles.sha256(java.io.File(dir, "manifest.json"))
                    }
                val run = plan.copy(samples = plan.samples.map { sample ->
                    sample.copy(bundleManifestSha256 = bundleHashes[sample.modelId to sample.method])
                })
                imageBenchmarkStore.save(run)
                updateImageBenchmark(run)
                mutable.update { it.copy(engineRetained = false, imageBenchmark = it.imageBenchmark.copy(activeRunId = run.id, methods = available)) }
                imageBenchmarkRunner.run(run.id, ::updateImageBenchmark)
            } catch (e: Exception) { fail(e) }
            finally {
                mutable.update { it.copy(imageBenchmarkRunning = false, imageBenchmark = it.imageBenchmark.copy(activeRunId = null)) }
                finish()
            }
        }
    }

    fun resumeImageBenchmark(id: String) {
        if (!begin("Resuming image benchmark…", inference = true)) return
        imageBenchmarkRunner.resetStop()
        imageBenchmarkClient.resetStop()
        mutable.update { it.copy(imageBenchmarkRunning = true) }
        scope.launch {
            try {
                val run = imageBenchmarkStore.load(id)
                require(run.resumable) { "This run has no pending samples." }
                val pending = run.samples.filter { it.status == ImageSampleStatus.PENDING }
                val available = imageMethodSupport(app)
                require(pending.all { sample -> available.any { it.method == sample.method && it.supports(sample.modelId) } }) { "A required method is unavailable on this device/build." }
                pending.filter { it.method == ImageMethod.TPU || ImageModels.get(it.modelId).pipeline == ImagePipeline.LITERT_PLAN }.distinctBy { it.modelId to it.method }.forEach { sample ->
                    val directory = LiteRtImageBundles(app, portable = sample.method != ImageMethod.TPU).load(sample.modelId).first
                    require(sample.bundleManifestSha256 == LiteRtImageBundles.sha256(java.io.File(directory, "manifest.json"))) {
                        "The LiteRT bundle changed. Start a new benchmark to preserve comparable results."
                    }
                }
                require(run.device.endsWith(android.os.Build.FINGERPRINT)) { "The device build changed. Start a new comparison to keep measurements consistent." }
                runtime.unload()
                mutable.update { it.copy(engineRetained = false, imageBenchmark = it.imageBenchmark.copy(activeRunId = id)) }
                imageBenchmarkRunner.run(id, ::updateImageBenchmark)
            } catch (e: Exception) { fail(e) }
            finally {
                mutable.update { it.copy(imageBenchmarkRunning = false, imageBenchmark = it.imageBenchmark.copy(activeRunId = null)) }
                finish()
            }
        }
    }

    fun rateImageBenchmark(id: String, sampleId: String, quality: Int?) {
        if (state.value.busy) return
        scope.launch {
            try { updateImageBenchmark(imageBenchmarkStore.rate(id, sampleId, quality)) }
            catch (e: Exception) { fail(e) }
        }
    }

    fun exportImageBenchmark(uri: Uri, id: String) {
        if (!begin("Exporting image benchmark…")) return
        scope.launch {
            try {
                app.contentResolver.openOutputStream(uri, "wt").use {
                    requireNotNull(it).write(imageBenchmarkStore.json(id).toByteArray(Charsets.UTF_8))
                }
                mutable.update { it.copy(progress = "Image benchmark exported.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun verifyImageTpuBundles() {
        if (!begin("Verifying LiteRT image bundles…")) return
        scope.launch {
            try {
                val message = LiteRtImageBundles(app, portable = true).verify() + "\n" +
                    if (LiteRtImageBundles(app).soc != null) LiteRtImageBundles(app).verify() else "Tensor TPU bundles require G5/G6."
                mutable.update { it.copy(progress = message, images = it.images.copy(ready = imageModels.ready()), imageBenchmark = it.imageBenchmark.copy(methods = imageMethodSupport(app))) }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    private fun refreshImageDownloads() {
        synchronized(this) {
            if (state.value.busy) return
            mutable.update { it.copy(busy = true) }
        }
        try {
            val transfers = imageModels.refresh { filename ->
                mutable.update { it.copy(busy = true, progress = "Verifying $filename…") }
            }
            val ready = imageModels.ready()
            mutable.update { it.copy(imageBenchmark = it.imageBenchmark.copy(methods = imageMethodSupport(app)), images = it.images.copy(ready = ready, transfers = transfers),
                progress = if (ready != it.images.ready && transfers.none { t -> t.active })
                    "Image model files verified and ready." else it.progress) }
        } finally { finish() }
    }

    fun downloadImageModel(id: String, wifiOnly: Boolean) {
        if (!begin("Starting image model download…")) return
        scope.launch {
            try {
                runtime.unload()
                val model = ImageModels.get(id)
                imageModels.enqueue(model, wifiOnly)
                val transfers = imageModels.refresh { }
                mutable.update { it.copy(engineRetained = false, images = it.images.copy(transfers = transfers),
                    progress = "${model.name} download started. Android manages the transfer.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun cancelImageDownload(id: String) {
        if (state.value.busy) return
        scope.launch {
            try {
                imageModels.cancel(id)
                val transfers = imageModels.refresh { }
                mutable.update { it.copy(images = it.images.copy(transfers = transfers), progress = "Image download canceled.") }
            } catch (e: Exception) { fail(e) }
        }
    }

    fun generateWordImage(id: String, word: String, category: String, prompt: String, seed: Long, steps: Int, method: ImageMethod = ImageMethod.CPU) {
        if (!begin("Loading image model · ${method.label}…", inference = true)) return
        imageBenchmarkClient.resetStop()
        try { DiffusionNative.resetStop() }
        catch (e: LinkageError) {
            mutable.update { it.copy(error = "Native image runtime unavailable: ${e.message}") }
            finish()
            return
        }
        mutable.update { it.copy(imageGenerating = true, images = it.images.copy(step = 0, totalSteps = steps)) }
        scope.launch {
            try {
                runtime.unload()
                mutable.update { it.copy(engineRetained = false) }
                val onProgress: (Int, Int) -> Unit = { step, total ->
                    mutable.update { it.copy(progress = if (step >= total) "Decoding and saving image… · ${method.label}"
                        else "Generating image · step $step / $total · ${method.label}",
                        images = it.images.copy(step = step, totalSteps = total)) }
                }
                val model = ImageModels.get(id)
                val result = if (method == ImageMethod.TPU || model.pipeline != ImagePipeline.NATIVE) {
                    val input = ImageBenchmarkPrompt(word, category, prompt, seed)
                    val sample = ImageBenchmarkSample("manual", id, model.name, method, input, steps)
                    imageBenchmarkClient.generate(sample, java.io.File(app.filesDir, "word-images"), 20, progress = onProgress)
                } else imageGenerator.generate(model, word, category, prompt, seed, steps, method, progress = onProgress)
                mutable.update { it.copy(images = it.images.copy(results = (listOf(result) + it.images.results).take(24)),
                    progress = "${result.modelName} image saved · ${result.totalMs / 1000.0} s including model load.") }
            } catch (_: CancellationException) {
                mutable.update { it.copy(progress = "Image generation stopped.") }
            } catch (e: Exception) { fail(e) }
            finally {
                mutable.update { it.copy(imageGenerating = false) }
                finish()
            }
        }
    }

    fun exportImageResults(uri: Uri) {
        if (!begin("Exporting image research results…")) return
        scope.launch {
            try {
                app.contentResolver.openOutputStream(uri, "wt").use {
                    requireNotNull(it).write(GsonBuilder().setPrettyPrinting().create()
                        .toJson(state.value.images.results).toByteArray(Charsets.UTF_8))
                }
                mutable.update { it.copy(progress = "Image research results exported.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun exportWordImage(uri: Uri, path: String) {
        if (!begin("Saving generated image…")) return
        scope.launch {
            try {
                require(state.value.images.results.any { it.imagePath == path } ||
                    state.value.imageBenchmark.runs.any { run -> run.samples.any { it.result?.imagePath == path } })
                app.contentResolver.openOutputStream(uri, "wt").use { output ->
                    requireNotNull(output)
                    java.io.File(path).inputStream().use { it.copyTo(output) }
                }
                mutable.update { it.copy(progress = "Generated image saved.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    @Synchronized private fun begin(message: String, inference: Boolean = false): Boolean {
        if (mutable.value.unavailable) return false
        mutable.update { it.copy(
            busy = true, runningInference = inference, stopRequested = false,
            progress = message, error = null, current = null, output = "",
        ) }
        return true
    }

    /** Reserves the existing worker; no second multi-GB engine competes with lab inference. */
    suspend fun voiceText(prompt: String, audio: ByteArray? = null, onText: (String) -> Unit = {}): String {
        check(begin("Word assistant is thinking…", inference = true)) {
            "The local model is busy. Wait for Research Lab to finish, then try again."
        }
        try {
            return withContext(worker) {
                currentCoroutineContext().ensureActive()
                runtime.resetStop()
                val snapshot = state.value
                val choice = selectLocalLlm(snapshot.models, snapshot.tpuSupport, snapshot.preferredLlmFamily, audio = true)
                    ?: error("Download a multimodal Gemma model in Settings → Research Lab to talk together.")
                val model = state.value.models.first { it.slot == choice.slot }
                val settings = RunSettings(accelerator = choice.accelerator, workload = Workload.PROMPT,
                    contextTokens = model.metadata.voiceContextTokens(),
                    outputTokens = if (audio == null) 192 else 128, prompt = prompt)
                validateBackend(model, settings)
                runtime.voiceText(model, settings, audio, onText)
            }
        } finally {
            // JNI cleanup stays on its owner even when the Activity cancels the request.
            withContext(NonCancellable + worker) { finish() }
        }
    }

    fun cancelVoiceInference() { runtime.requestStop() }

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
                val cacheRequest = PhotoCacheRequest.create(model, settings,
                    visionAccelerator(metadata?.visionBackends.orEmpty(), accelerator, runtime.tpuSupport.available)?.name,
                    BuildConfig.LITERT_LM_VERSION)
                if (resume) {
                    require(state.value.photos.modelSha256 == model.sha256 && state.value.photos.textBackend == accelerator.name) {
                        "Resume with the same model and backend, or start over."
                    }
                } else {
                    val page = photoLibrary.nextBatch(state.value.photos.beforeId, batchSize)
                    for (skipped in page.skippedPhotos) {
                        photoCache.save(cacheRequest, null, PhotoResult(skipped.photo,
                            status = PhotoStatus.SKIPPED, skippedReason = skipped.reason))
                    }
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
                    updatePhoto(index) { PhotoResult(it.photo, status = PhotoStatus.ANALYZING) }
                    var stage = "image_decode"
                    var inputSha256: String? = null
                    try {
                        val prepared = photoLibrary.prepare(result.photo)
                        updatePhoto(index) { it.copy(previewPath = prepared.file.absolutePath, width = prepared.width, height = prepared.height) }
                        if (state.value.stopRequested) break
                        val jpeg = prepared.file.readBytes()
                        inputSha256 = PhotoCacheRequest.sha256(jpeg)
                        stage = "cache_lookup"
                        val cached = photoCache.completed(cacheRequest.key(result.photo, inputSha256), result.photo)
                        if (cached != null) {
                            updatePhoto(index) { cached.copy(previewPath = prepared.file.absolutePath) }
                            mutable.update { it.copy(photos = it.photos.copy(visionBackend = cached.visionBackend)) }
                            continue
                        }
                        if (state.value.stopRequested) break
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
                                    decodeTokens = measurement.metrics.decodeTokens, outputTokenLimit = request.outputTokens,
                                    measurements = it.measurements + PhotoAttempt(attempt, request.outputTokens, measurement),
                                    visionBackend = runtime.photoVisionBackend) }
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
                        if (stage == "cache_lookup") throw e
                        val kind = if (e is PhotoResponseException) "model_json" else stage
                        val failed = state.value.photos.results[index]
                        Log.w("GemmaPhotos", "photo=${index + 1}/${batch.size} model=${slot.name} text=$accelerator stage=$kind " +
                            "attempts=${failed.attempts} decoded=${failed.decodeTokens} limit=${failed.outputTokenLimit} " +
                            "chars=${failed.rawOutput.length} exception=${e.javaClass.simpleName}")
                        updatePhoto(index) { it.copy(status = PhotoStatus.FAILED, errorKind = kind, error = e.message ?: e.javaClass.simpleName) }
                    }
                    val saved = photoCache.save(cacheRequest, inputSha256, state.value.photos.results[index])
                    updatePhoto(index) { saved }
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

    private fun accessiblePeoplePhotos(): Map<String, String> = if (PhotoPermissions.access(app) == PhotoAccess.DENIED) emptyMap()
        else photoLibrary.accessiblePhotos().associate { it.uri to it.sourceRevision }

    private suspend fun refreshPeopleGallery() {
        val (groups, unassigned) = peopleRepository.gallery(accessiblePeoplePhotos())
        mutable.update { it.copy(people = it.people.copy(groups = groups, unassigned = unassigned,
            calibrated = faceEngine.policy().validated, accessScope = PhotoPermissions.access(app), refreshing = false)) }
    }

    /** Hide stale permission-scoped content immediately; refresh on the serialized worker. */
    fun refreshPeople() {
        val generation = peopleRefreshGeneration.incrementAndGet()
        mutable.update { it.copy(people = it.people.copy(groups = emptyList(), unassigned = emptyList(), refreshing = true)) }
        scope.launch {
            while (state.value.unavailable) { delay(100); if (peopleRefreshGeneration.get() != generation) return@launch }
            if (peopleRefreshGeneration.get() != generation || !begin("Refreshing people…")) return@launch
            try {
                mutable.update { it.copy(people = PeopleState(refreshing = true)) }
                refreshPeopleGallery()
                mutable.update { it.copy(progress = "People gallery refreshed.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun findPeople(batchSize: Int, resume: Boolean = false) {
        if (!begin("Preparing people batch…", inference = true)) return
        mutable.update { it.copy(photoBatchProgress = null) }
        scope.launch {
            try {
                require(batchSize in 1..PhotoDemoState.MAX_BATCH_SIZE)
                check(PhotoPermissions.access(app) != PhotoAccess.DENIED) { "Grant photo access first." }
                runtime.unload()
                mutable.update { it.copy(engineRetained = false) }
                val accessible = accessiblePeoplePhotos()
                if (!resume) {
                    val page = photoLibrary.nextBatch(state.value.people.beforeId, batchSize)
                    mutable.update { it.copy(people = it.people.copy(results = page.photos.map(::PeopleBatchResult),
                        beforeId = page.photos.lastOrNull()?.id ?: it.people.beforeId, hasMore = page.hasMore,
                        batchNumber = it.people.batchNumber + 1)) }
                }
                val policy = faceEngine.policy()
                val batch = state.value.people.results
                for ((index, result) in batch.withIndex()) {
                    if (state.value.stopRequested) break
                    if (result.status == "COMPLETED") continue
                    mutable.update { it.copy(progress = "Finding people · photo ${index + 1}/${batch.size}") }
                    val updated = try {
                        if (peopleRepository.cached(result.photo)) {
                            PeopleBatchResult(result.photo, "COMPLETED", PhotoCacheDatabase.getInstance(app).people().facesForPhoto(result.photo.uri).size, cached = true)
                        } else {
                            val faces = faceEngine.analyze(result.photo.uri)
                            if (state.value.stopRequested) break
                            peopleRepository.save(result.photo, faces, accessible, policy)
                            PeopleBatchResult(result.photo, "COMPLETED", faces.size)
                        }
                    } catch (e: Exception) {
                        if (e is CancellationException) throw e
                        val message = e.message ?: "Face analysis failed."
                        peopleRepository.failed(result.photo, message)
                        PeopleBatchResult(result.photo, "FAILED", error = message)
                    }
                    mutable.update { it.copy(people = it.people.copy(results = it.people.results.mapIndexed { i, r -> if (i == index) updated else r })) }
                }
                refreshPeopleGallery()
                mutable.update { it.copy(progress = if (it.stopRequested) "People scan stopped. Resume this batch to continue."
                    else "People batch finished · ${it.people.results.count { r -> r.status == "COMPLETED" }}/${batch.size} photos processed.") }
            } catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    private fun editPeople(message: String, action: suspend PeopleRepository.() -> Unit) {
        if (!begin(message)) return
        scope.launch {
            try { peopleRepository.action(); refreshPeopleGallery(); mutable.update { it.copy(progress = message) } }
            catch (e: Exception) { fail(e) }
            finally { finish() }
        }
    }

    fun renamePerson(id: String, name: String) = editPeople("Person name saved.") { rename(id, name) }
    fun mergePeople(source: String, target: String) = editPeople("Person groups merged.") { merge(source, target) }
    fun movePersonFace(faceId: String, target: String?) = editPeople("Face moved.") { move(faceId, target) }
    fun clearPeople() = editPeople("Person data cleared.") {
        clear()
        mutable.update { it.copy(people = PeopleState()) }
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
                        "skippedReason" to result.skippedReason, "measurements" to result.measurements,
                        "cacheKey" to result.cacheKey, "fromCache" to result.fromCache,
                        "computedAtEpochMs" to result.computedAtEpochMs,
                        "visionBackend" to result.visionBackend,
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
        try {
            if (state.value.imageBenchmarkRunning) imageBenchmarkRunner.stop()
            else if (state.value.imageGenerating) {
                imageBenchmarkClient.stop()
                DiffusionNative.requestStop()
            } else runner.stop()
        } catch (e: Exception) { fail(e) }
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
