package dev.gemmabench.imagegen

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import java.util.concurrent.atomic.AtomicBoolean

interface ImageBenchmarkRepository {
    fun load(id: String): ImageBenchmarkRun
    fun save(run: ImageBenchmarkRun)
}
/** Orchestration is independent of Android/JNI so interruption, stats and failed methods are testable. */
class ImageBenchmarkRunner(
    private val repository: ImageBenchmarkRepository,
    private val generate: suspend (ImageBenchmarkRun, ImageBenchmarkSample) -> Pair<ImageResult, Long>,
    private val cancelNative: () -> Unit,
    private val clock: () -> Long = System::currentTimeMillis,
    private val betweenSamples: suspend () -> Unit = { delay(1000) },
) {
    private val stopped = AtomicBoolean(false)
    fun resetStop() { stopped.set(false) }
    fun stop() { stopped.set(true); cancelNative() }
    suspend fun run(id: String, onUpdate: (ImageBenchmarkRun) -> Unit) {
        fun checkpoint(transform: (ImageBenchmarkRun) -> ImageBenchmarkRun): ImageBenchmarkRun {
            val next = transform(repository.load(id)).copy(updatedAtEpochMs = clock())
            repository.save(next)
            onUpdate(next)
            return next
        }
        checkpoint { it.copy(status = ImageRunStatus.RUNNING, finishedAtEpochMs = null) }
        try {
            while (!stopped.get()) {
                val run = repository.load(id)
                val sample = run.samples.firstOrNull { it.status == ImageSampleStatus.PENDING } ?: break
                checkpoint { it.copy(samples = it.samples.map { s -> if (s.id == sample.id) s.copy(status = ImageSampleStatus.RUNNING, error = null) else s }) }
                try {
                    val (result, elapsed) = generate(run, sample)
                    checkpoint { it.copy(samples = it.samples.map { s -> if (s.id == sample.id) s.copy(
                        status = ImageSampleStatus.SUCCESS, result = result, elapsedMs = elapsed, error = null) else s }) }
                } catch (e: Exception) {
                    if (stopped.get() || (e is CancellationException && e !is kotlinx.coroutines.TimeoutCancellationException)) throw CancellationException("Stopped", e)
                    val message = if (e is kotlinx.coroutines.TimeoutCancellationException) "Per-image timeout (${run.timeoutMinutes} min)." else e.message ?: e.javaClass.simpleName
                    val disableMethod = sample.method == ImageMethod.GPU || (e is ImageWorkerException && e.methodUnavailable)
                    checkpoint { current -> current.copy(samples = current.samples.map { s ->
                        when {
                            s.id == sample.id -> s.copy(status = ImageSampleStatus.FAILED, error = message)
                            disableMethod && s.status == ImageSampleStatus.PENDING && s.method == sample.method &&
                                (sample.method == ImageMethod.GPU || s.modelId == sample.modelId) ->
                                s.copy(status = ImageSampleStatus.SKIPPED, error = "Method stopped after an earlier failure: $message")
                            else -> s
                        }
                    }) }
                }
                if (!stopped.get() && repository.load(id).pending > 0) betweenSamples()
            }
            checkpoint { it.copy(status = if (stopped.get()) ImageRunStatus.CANCELED else ImageRunStatus.COMPLETED,
                finishedAtEpochMs = clock(), samples = it.samples.map { s ->
                    if (s.status == ImageSampleStatus.RUNNING) s.copy(status = ImageSampleStatus.PENDING) else s
                }) }
        } catch (_: CancellationException) {
            checkpoint { it.copy(status = ImageRunStatus.CANCELED, finishedAtEpochMs = clock(), samples = it.samples.map { s ->
                if (s.status == ImageSampleStatus.RUNNING) s.copy(status = ImageSampleStatus.PENDING, error = "Stopped; resume to retry this word.") else s
            }) }
        }
    }
}
