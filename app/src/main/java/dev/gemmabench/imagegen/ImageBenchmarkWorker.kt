package dev.gemmabench.imagegen

import android.app.Service
import android.content.*
import android.content.pm.PackageManager
import android.os.*
import com.google.gson.Gson
import kotlinx.coroutines.*
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

private const val RUN = 1
private const val PROGRESS = 2
private const val RESULT = 3
private const val READY = 4
private const val STOP = 5

/** A native driver abort/hang is contained in :imagebenchmark, never the UI/manifest process. */
class ImageBenchmarkWorkerService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private val started = AtomicBoolean(false)
    private var combination: Pair<String, ImageMethod>? = null
    private var verifiedBundleHash: String? = null
    private val generator by lazy { ImageGenerator(this, ImageModelStore(this)) }
    private val messenger = Messenger(Handler(Looper.getMainLooper()) { message ->
        when (message.what) {
            STOP -> { Process.killProcess(Process.myPid()); true }
            RUN -> {
                val reply = message.replyTo
                if (!started.compareAndSet(false, true)) return@Handler true
                fun send(what: Int, bundle: Bundle) = runCatching {
                    reply.send(Message.obtain(null, what).apply { data = bundle })
                }
                send(READY, Bundle().apply { putInt("pid", Process.myPid()) })
                val request = Bundle(message.data)
                Thread({
                    try {
                        val model = ImageModels.get(requireNotNull(request.getString("model")))
                        val method = ImageMethod.valueOf(requireNotNull(request.getString("method")))
                        val key = model.id to method
                        check(combination == null || combination == key) { "Unload the worker before changing model/backend." }
                        combination = key
                        val dir = File(requireNotNull(request.getString("directory"))).canonicalFile
                        val benchmarkRoot = File(filesDir, "image-benchmarks").canonicalPath + File.separator
                        require(dir.path.startsWith(benchmarkRoot) || dir == File(filesDir, "word-images").canonicalFile)
                        request.getString("bundleManifestSha256")?.let { expected ->
                            if (verifiedBundleHash == null) {
                                val bundle = LiteRtImageBundles(this, portable = method != ImageMethod.TPU).load(model.id).first
                                require(expected == LiteRtImageBundles.sha256(File(bundle, "manifest.json"))) { "LiteRT bundle changed after this run was planned." }
                                verifiedBundleHash = expected
                            }
                            require(verifiedBundleHash == expected) { "LiteRT bundle changed within the session." }
                        }
                        if (model.pipeline == ImagePipeline.NATIVE && method != ImageMethod.TPU) DiffusionNative.resetStop()
                        val result = generator.generate(model,
                            requireNotNull(request.getString("word")), requireNotNull(request.getString("category")),
                            requireNotNull(request.getString("prompt")), request.getLong("seed"), request.getInt("steps"), method, dir, retainEngine = request.getBoolean("retainWorker")) { step, total ->
                            send(PROGRESS, Bundle().apply { putInt("step", step); putInt("total", total) })
                        }
                        started.set(false)
                        send(RESULT, Bundle().apply { putString("result", Gson().toJson(result)) })
                    } catch (e: Throwable) {
                        send(RESULT, Bundle().apply { putString("error", e.message ?: e.javaClass.simpleName) })
                    } finally {
                        if (!request.getBoolean("retainWorker")) {
                            handler.postDelayed({ Process.killProcess(Process.myPid()) }, 100)
                        }
                    }
                }, "image-benchmark-native").start()
                true
            }
            else -> false
        }
    })
    override fun onBind(intent: Intent?): IBinder = messenger.binder
    override fun onUnbind(intent: Intent?): Boolean {
        handler.postDelayed({ Process.killProcess(Process.myPid()) }, 100)
        return false
    }
}

class ImageWorkerException(message: String, val methodUnavailable: Boolean = false) : Exception(message)

/** One bound worker per combination; release waits for process death before another can load. */
class ImageBenchmarkClient(private val context: Context) {
    private val stopped = AtomicBoolean(false)
    @Volatile private var activePid: Int? = null
    @Volatile private var activeMessenger: Messenger? = null
    @Volatile private var activeContinuation: CancellableContinuation<ImageResult>? = null
    @Volatile private var binder: IBinder? = null
    private var connection: ServiceConnection? = null
    private var bound = false
    private var combination: Pair<String, ImageMethod>? = null
    fun resetStop() { stopped.set(false) }
    fun stop() {
        stopped.set(true)
        activeContinuation?.cancel(CancellationException("Image worker stopped."))
        terminateWorker()
    }
    private fun terminateWorker() {
        val pid = activePid
        if (pid != null && pid != Process.myPid()) runCatching { Process.killProcess(pid) }
        else runCatching { activeMessenger?.send(Message.obtain(null, STOP)) }
    }
    suspend fun release() = withContext(NonCancellable) {
        val deadline = SystemClock.elapsedRealtime() + 3_000
        // A canceled bind can still connect asynchronously. Observe that binder before unbinding,
        // otherwise a later bind could reuse a process whose termination is still pending.
        while (bound && binder == null && SystemClock.elapsedRealtime() < deadline) delay(20)
        terminateWorker()
        val workerBinder = binder
        val connected = !bound || workerBinder != null
        while (workerBinder?.isBinderAlive == true && SystemClock.elapsedRealtime() < deadline) delay(20)
        // Keep the binding on failure so a retry can still observe and terminate a late connection.
        check(connected && workerBinder?.isBinderAlive != true) { "Image worker did not exit; refusing to load another model." }
        if (bound) connection?.let { runCatching { context.unbindService(it) } }
        bound = false
        connection = null
        binder = null
        activePid = null
        activeMessenger = null
        activeContinuation = null
        combination = null
    }
    suspend fun generate(sample: ImageBenchmarkSample, directory: File, timeoutMinutes: Int,
        retainWorker: Boolean = false, progress: (Int, Int) -> Unit): ImageResult {
        val key = sample.modelId to sample.method
        if (combination != null && (combination != key || activeMessenger == null || binder?.isBinderAlive != true)) release()
        var success = false
        try {
            val result = withTimeout(timeoutMinutes * 60_000L) {
                if (stopped.get()) throw CancellationException("Image worker stopped before loading.")
                suspendCancellableCoroutine { continuation ->
                    activeContinuation = continuation
                    if (stopped.get()) { continuation.cancel(); return@suspendCancellableCoroutine }
                    val reply = Messenger(Handler(Looper.getMainLooper()) { message ->
                        when (message.what) {
                            READY -> activePid = message.data.getInt("pid")
                            PROGRESS -> if (continuation.isActive) progress(message.data.getInt("step"), message.data.getInt("total"))
                            RESULT -> if (continuation.isActive) {
                                val error = message.data.getString("error")
                                if (error != null) continuation.resumeWithException(ImageWorkerException(error))
                                else runCatching {
                                    Gson().fromJson(requireNotNull(message.data.getString("result")), ImageResult::class.java)
                                }.fold({ continuation.resume(it) }, { continuation.resumeWithException(it) })
                            }
                        }
                        true
                    })
                    fun send() {
                        if (!continuation.isActive) return
                        runCatching {
                            activeMessenger!!.send(Message.obtain(null, RUN).apply {
                                replyTo = reply
                                data = Bundle().apply {
                                    putString("model", sample.modelId); putString("method", sample.method.name)
                                    putBoolean("retainWorker", retainWorker)
                                    sample.bundleManifestSha256?.let { putString("bundleManifestSha256", it) }
                                    putString("word", sample.input.word); putString("category", sample.input.category)
                                    putString("prompt", sample.input.prompt); putLong("seed", sample.input.seed)
                                    putInt("steps", sample.steps); putString("directory", directory.path)
                                }
                            })
                        }.onFailure { if (continuation.isActive) continuation.resumeWithException(it) }
                    }
                    continuation.invokeOnCancellation { terminateWorker() }
                    if (activeMessenger != null) {
                        send()
                    } else {
                        combination = key
                        val serviceConnection = object : ServiceConnection {
                            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                                binder = service
                                activeMessenger = Messenger(service)
                                if (stopped.get() || !continuation.isActive) terminateWorker() else send()
                            }
                            override fun onServiceDisconnected(name: ComponentName?) {
                                activeContinuation?.let { pending ->
                                    if (pending.isActive) pending.resumeWithException(ImageWorkerException(
                                        "Native ${sample.method.label} worker exited before returning an image (driver crash or memory limit).", true))
                                }
                            }
                            override fun onBindingDied(name: ComponentName?) = onServiceDisconnected(name)
                            override fun onNullBinding(name: ComponentName?) = onServiceDisconnected(name)
                        }
                        connection = serviceConnection
                        bound = context.bindService(Intent(context, ImageBenchmarkWorkerService::class.java), serviceConnection, Context.BIND_AUTO_CREATE)
                        if (!bound && continuation.isActive) continuation.resumeWithException(ImageWorkerException("Could not start image worker."))
                    }
                }
            }
            success = true
            return result
        } finally {
            activeContinuation = null
            if (!retainWorker || !success || stopped.get()) release()
        }
    }
}

fun imageMethodSupport(context: Context): List<ImageMethodSupport> {
    val cpu = Build.SUPPORTED_ABIS.contains("arm64-v8a")
    val nativeGpu = context.packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x00401000) &&
        runCatching { DiffusionNative.vulkanBuilt() }.getOrDefault(false)
    val hardware = LiteRtImageHardware.available(context)
    val bundles = LiteRtImageBundles(context)
    return imageBackendSupport(ImageModels.all, ImageModelStore(context).ready(), bundles.installed(), cpu,
        com.google.ai.edge.litert.Accelerator.GPU in hardware,
        com.google.ai.edge.litert.Accelerator.NPU in hardware, nativeGpu, bundles.soc)
}
