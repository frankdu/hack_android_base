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
                        val dir = File(requireNotNull(request.getString("directory"))).canonicalFile
                        require(dir.path.startsWith(File(filesDir, "image-benchmarks").canonicalPath + File.separator))
                        DiffusionNative.resetStop()
                        val result = ImageGenerator(this, ImageModelStore(this)).generate(model,
                            requireNotNull(request.getString("word")), requireNotNull(request.getString("category")),
                            requireNotNull(request.getString("prompt")), request.getLong("seed"), request.getInt("steps"), method, dir) { step, total ->
                            send(PROGRESS, Bundle().apply { putInt("step", step); putInt("total", total) })
                        }
                        send(RESULT, Bundle().apply { putString("result", Gson().toJson(result)) })
                    } catch (e: Throwable) {
                        send(RESULT, Bundle().apply { putString("error", e.message ?: e.javaClass.simpleName) })
                    } finally {
                        // Fresh process for the next sample: release driver state and native high-water marks.
                        handler.postDelayed({ Process.killProcess(Process.myPid()) }, 100)
                    }
                }, "image-benchmark-native").start()
                true
            }
            else -> false
        }
    })
    override fun onBind(intent: Intent?): IBinder = messenger.binder
    override fun onUnbind(intent: Intent?): Boolean {
        if (started.get()) handler.postDelayed({ Process.killProcess(Process.myPid()) }, 100)
        return false
    }
}

class ImageWorkerException(message: String, val methodUnavailable: Boolean = false) : Exception(message)

class ImageBenchmarkClient(private val context: Context) {
    private val stopped = AtomicBoolean(false)
    @Volatile private var activePid: Int? = null
    @Volatile private var activeMessenger: Messenger? = null
    @Volatile private var activeContinuation: CancellableContinuation<ImageResult>? = null
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
    suspend fun generate(sample: ImageBenchmarkSample, directory: File, timeoutMinutes: Int,
        progress: (Int, Int) -> Unit): ImageResult = withTimeout(timeoutMinutes * 60_000L) {
        if (stopped.get()) throw CancellationException("Image worker stopped before loading.")
        var binder: IBinder? = null
        var connection: ServiceConnection? = null
        var bound = false
        try {
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
                val serviceConnection = object : ServiceConnection {
                    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                        if (!continuation.isActive) return
                        binder = service
                        activeMessenger = Messenger(service)
                        runCatching {
                            activeMessenger!!.send(Message.obtain(null, RUN).apply {
                                replyTo = reply
                                data = Bundle().apply {
                                    putString("model", sample.modelId); putString("method", sample.method.name)
                                    putString("word", sample.input.word); putString("category", sample.input.category)
                                    putString("prompt", sample.input.prompt); putLong("seed", sample.input.seed)
                                    putInt("steps", sample.steps); putString("directory", directory.path)
                                }
                            })
                        }.onFailure { if (continuation.isActive) continuation.resumeWithException(it) }
                    }
                    override fun onServiceDisconnected(name: ComponentName?) {
                        if (continuation.isActive) continuation.resumeWithException(
                            ImageWorkerException("Native ${sample.method.label} worker exited before returning an image (driver crash or memory limit).", true))
                    }
                    override fun onBindingDied(name: ComponentName?) = onServiceDisconnected(name)
                    override fun onNullBinding(name: ComponentName?) = onServiceDisconnected(name)
                }
                connection = serviceConnection
                bound = context.bindService(Intent(context, ImageBenchmarkWorkerService::class.java), serviceConnection, Context.BIND_AUTO_CREATE)
                if (!bound && continuation.isActive) continuation.resumeWithException(ImageWorkerException("Could not start image worker."))
                continuation.invokeOnCancellation { terminateWorker() }
            }
        } finally {
            // Ensure a subsequent bind cannot reuse a driver context scheduled for termination.
            terminateWorker()
            withContext(NonCancellable) {
                val deadline = SystemClock.elapsedRealtime() + 3_000
                while (binder?.isBinderAlive == true && SystemClock.elapsedRealtime() < deadline) delay(20)
                if (bound) connection?.let { runCatching { context.unbindService(it) } }
                activePid = null
                activeMessenger = null
                activeContinuation = null
            }
        }
    }
}

fun imageMethodSupport(context: Context): List<ImageMethodSupport> {
    val cpu = Build.SUPPORTED_ABIS.contains("arm64-v8a")
    val gpu = context.packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x00401000)
    val built = runCatching { DiffusionNative.vulkanBuilt() }.getOrDefault(false)
    return listOf(
        ImageMethodSupport(ImageMethod.CPU, cpu, if (cpu) "ARM64 native runtime" else "Requires ARM64"),
        ImageMethodSupport(ImageMethod.GPU, gpu && built, when {
            !built -> "Vulkan is not included in this build."
            !gpu -> "Device does not advertise Vulkan 1.1 compute support."
            else -> "Experimental Vulkan: driver and model compatibility are checked during generation. No automatic CPU rerouting."
        }),
        ImageMethodSupport(ImageMethod.TPU, false, "No Tensor G5/G6 compiled diffusion artifacts or diffusion TPU backend are installed. Gemma TPU models cannot run this image pipeline."),
    )
}
