package dev.gemmabench.runtime

import android.content.Context
import android.os.Build
import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.BenchmarkInfo
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ModelInfo
import com.google.ai.edge.litertlm.Modality
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.ai.edge.litertlm.ResponseFormat
import com.google.ai.edge.litertlm.benchmark
import dev.gemmabench.benchmark.*
import dev.gemmabench.photos.PhotoObjects
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.flow.collect

/** The only place depending on LiteRT-LM, reusable when the lab moves into dev settings. */
@OptIn(ExperimentalApi::class)
class LiteRtRuntime(private val context: Context, private val modelFile: (ModelArtifact) -> File) : InferenceRuntime {
    private var engine: Engine? = null
    private var engineKey: String? = null
    var photoVisionBackend: String? = null
        private set
    private val stopped = AtomicBoolean(false)
    private val conversationLock = Any()
    private var activeConversation: Conversation? = null

    val tensorLibrariesAvailable: Boolean
        get() = File(context.applicationInfo.nativeLibraryDir, "libLiteRtDispatch_GoogleTensor.so").isFile

    val tpuSupport: TpuSupport by lazy {
        val initial = TpuSupport.check(Build.SOC_MODEL, Build.SUPPORTED_ABIS.toList(), tensorLibrariesAvailable)
        if (!initial.available) initial else {
            val error = try {
                System.loadLibrary("edgetpu_util")
                System.loadLibrary("edgetpu_litert")
                System.load(File(context.applicationInfo.nativeLibraryDir, "libLiteRtDispatch_GoogleTensor.so").absolutePath)
                null
            } catch (e: LinkageError) { e.message ?: e.javaClass.simpleName }
              catch (e: SecurityException) { e.message ?: e.javaClass.simpleName }
            TpuSupport.check(Build.SOC_MODEL, Build.SUPPORTED_ABIS.toList(), tensorLibrariesAvailable, error)
        }
    }

    private fun backend(accelerator: Accelerator): Backend = when (accelerator) {
        Accelerator.CPU -> Backend.CPU()
        Accelerator.GPU -> Backend.GPU()
        Accelerator.TPU -> {
            check(tpuSupport.available) { tpuSupport.reason }
            Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        }
    }

    override fun resetStop() { stopped.set(false) }

    override fun requestStop() {
        stopped.set(true)
        synchronized(conversationLock) { activeConversation?.cancelProcess() }
    }

    override fun unload() {
        engine?.let { if (it.isInitialized()) it.close() }
        engine = null
        engineKey = null
        photoVisionBackend = null
    }

    override suspend fun measure(
        model: ModelArtifact,
        settings: RunSettings,
        onText: (String) -> Unit,
    ): Measurement = try {
        measureNative(model, settings, onText)
    } catch (e: LinkageError) {
        throw IllegalStateException("Cannot load native inference libraries: ${e.message}", e)
    }

    /** Shares worker/cancellation with text runs, but uses a distinct engine with vision enabled. */
    suspend fun analyzePhoto(model: ModelArtifact, settings: RunSettings, jpeg: ByteArray, onText: (String) -> Unit): Measurement = try {
        measureNative(model, settings, onText, jpeg)
    } catch (e: LinkageError) {
        throw IllegalStateException("Cannot load native vision libraries: ${e.message}", e)
    }

    private suspend fun measureNative(
        model: ModelArtifact,
        settings: RunSettings,
        onText: (String) -> Unit,
        jpeg: ByteArray? = null,
    ): Measurement {
        check(!stopped.get()) { "Run stopped." }
        val file = modelFile(model)
        check(file.isFile && file.length() == model.bytes) { "Model file is missing or changed; import it again." }
        check(model.slot.tpu == (settings.accelerator == Accelerator.TPU)) {
            "Select a CPU/GPU model for CPU or GPU, or the matching Tensor model for TPU."
        }
        model.unavailableReason(settings.accelerator, tpuSupport, photos = jpeg != null)?.let { error(it) }
        if (settings.workload == Workload.FIXED_TOKENS) {
            unload()
            ExperimentalFlags.enableBenchmark = true
            ExperimentalFlags.enableSpeculativeDecoding = settings.speculativeDecoding
            val start = SystemClock.elapsedRealtimeNanos()
            val info = benchmark(
                modelPath = file.absolutePath,
                backend = backend(settings.accelerator),
                prefillTokens = settings.prefillTokens,
                decodeTokens = settings.outputTokens,
                cacheDir = cache(model, settings),
                prompt = settings.prompt,
            )
            return Measurement(info.metrics(), elapsedMs(start))
        }

        val key = "${model.sha256}:${settings.accelerator}:${settings.contextTokens}:${settings.speculativeDecoding}:vision=${jpeg != null}"
        val reused = engine != null && key == engineKey
        var loadMs: Double? = null
        if (!reused) {
            unload()
            var vision: Backend? = null
            if (jpeg != null) {
                ModelInfo.from(file.absolutePath).use { info ->
                    require(info.inputModalities().vision) { "This model has no vision encoder. Import a multimodal Gemma model." }
                    val supported = info.supportedBackends(Modality.VISION).map { it.name }
                    val selected = visionAccelerator(supported, settings.accelerator, tpuSupport.available)
                        ?: error("This vision encoder requires an unavailable backend: ${supported.joinToString()}.")
                    photoVisionBackend = selected.name
                    vision = backend(selected)
                }
            }
            ExperimentalFlags.enableBenchmark = true
            ExperimentalFlags.enableSpeculativeDecoding = settings.speculativeDecoding
            val created = Engine(EngineConfig(
                modelPath = file.absolutePath,
                backend = backend(settings.accelerator),
                visionBackend = vision,
                maxNumImages = if (jpeg != null) 1 else null,
                maxNumTokens = settings.contextTokens,
                cacheDir = cache(model, settings),
            ))
            val start = SystemClock.elapsedRealtimeNanos()
            try {
                created.initialize()
            } catch (e: Exception) {
                if (created.isInitialized()) created.close()
                throw e
            }
            loadMs = elapsedMs(start)
            engine = created
            engineKey = key
        }
        if (stopped.get()) throw CancellationException("Run stopped during initialization.")
        val conversationStart = SystemClock.elapsedRealtimeNanos()
        val conversation = engine!!.createConversation(ConversationConfig(
            // Fresh conversation for every request: no history or prefix reuse in prompt benchmarks.
            samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0),
            maxOutputToken = settings.outputTokens,
            thinkingConfig = ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0),
            enableSpeculativeDecoding = settings.speculativeDecoding,
            enableResponseFormat = jpeg != null,
        ))
        val setupMs = elapsedMs(conversationStart)
        synchronized(conversationLock) { activeConversation = conversation }
        try {
            if (stopped.get()) throw CancellationException("Run stopped.")
            val output = StringBuilder()
            var firstVisibleMs: Double? = null
            var lastPublish = 0L
            val requestStart = SystemClock.elapsedRealtimeNanos()
            val contents = if (jpeg == null) Contents.of(settings.prompt)
                else Contents.of(Content.ImageBytes(jpeg), Content.Text(settings.prompt))
            conversation.sendMessageAsync(contents,
                responseFormat = if (jpeg != null) ResponseFormat.json(PhotoObjects.SCHEMA) else null,
            ).bufferInferenceChunks().collect { message ->
                val text = message.toString()
                if (text.isNotEmpty()) {
                    if (firstVisibleMs == null) firstVisibleMs = elapsedMs(requestStart)
                    output.append(text)
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastPublish >= 100) {
                        onText(output.toString())
                        lastPublish = now
                    }
                }
            }
            val elapsed = elapsedMs(requestStart)
            if (stopped.get()) throw CancellationException("Run stopped.")
            onText(output.toString())
            return Measurement(
                metrics = conversation.getBenchmarkInfo().metrics(),
                elapsedMs = elapsed,
                engineLoadMs = loadMs,
                conversationSetupMs = setupMs,
                firstVisibleTextMs = firstVisibleMs,
                engineReused = reused,
                output = output.toString(),
            )
        } finally {
            synchronized(conversationLock) { activeConversation = null }
            conversation.close()
        }
    }

    private fun cache(model: ModelArtifact, settings: RunSettings): String =
        File(context.cacheDir, "litert/${model.sha256}/${settings.accelerator}").apply { mkdirs() }.absolutePath

    private fun BenchmarkInfo.metrics() = NativeMetrics(
        initTimeInSecond.positiveOrNull(), timeToFirstTokenInSecond.positiveOrNull(),
        lastPrefillTokenCount, lastDecodeTokenCount,
        lastPrefillTokensPerSecond.positiveOrNull(), lastDecodeTokensPerSecond.positiveOrNull(),
    )

    private fun elapsedMs(startNanos: Long): Double = (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0
}
