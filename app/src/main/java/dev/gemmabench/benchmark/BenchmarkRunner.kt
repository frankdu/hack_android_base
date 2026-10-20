package dev.gemmabench.benchmark

import java.time.Instant
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.delay

/** Runs on one worker dispatcher. No overlapping engines or global experimental-flag changes. */
class BenchmarkRunner(
    private val runtime: InferenceRuntime,
    private val device: () -> DeviceInfo,
    private val snapshot: () -> EnvironmentSnapshot,
) {
    private val stopped = AtomicBoolean(false)

    fun stop() {
        stopped.set(true)
        runtime.requestStop()
    }

    fun reset() {
        stopped.set(false)
        runtime.resetStop()
    }

    suspend fun run(
        model: ModelArtifact,
        settings: RunSettings,
        onProgress: (String) -> Unit,
        onText: (String) -> Unit = {},
        onSample: (BenchmarkResult) -> Unit = {},
    ): BenchmarkResult {
        settings.validate()
        val startedAt = Instant.now().toString()
        val resultId = java.util.UUID.randomUUID().toString()
        val deviceInfo = device()
        val samples = mutableListOf<RunSample>()
        fun result(status: String, error: String? = null) = BenchmarkResult(
            id = resultId, startedAt = startedAt, model = model, settings = settings, device = deviceInfo,
            samples = samples.toList(), status = status, error = error,
        )
        try {
            repeat(settings.warmups + settings.repetitions) { index ->
                if (stopped.get()) return result("stopped")
                val warmup = index < settings.warmups
                val runIndex = if (warmup) index + 1 else index - settings.warmups + 1
                onProgress("${model.slot.label} · ${if (warmup) "Warm-up" else "Run"} $runIndex / ${if (warmup) settings.warmups else settings.repetitions}")
                val before = snapshot()
                val measurement = runtime.measure(model, settings, onText)
                // A fixed-token native call cannot be interrupted safely. Discard it if stop was requested.
                if (stopped.get()) return result("stopped")
                samples += RunSample(
                    runIndex, warmup, measurement.metrics, measurement.elapsedMs,
                    measurement.engineLoadMs, measurement.conversationSetupMs,
                    measurement.firstVisibleTextMs, measurement.engineReused, measurement.output,
                    before, snapshot(),
                )
                onSample(result("running"))
                if (index < settings.warmups + settings.repetitions - 1) {
                    repeat(settings.cooldownSeconds * 10) {
                        if (stopped.get()) return result("stopped")
                        delay(100)
                    }
                }
            }
            return result("completed")
        } catch (e: CancellationException) {
            if (!stopped.get()) throw e
            return result("stopped")
        } catch (e: Exception) {
            return result(if (stopped.get()) "stopped" else "failed", e.message ?: e.javaClass.simpleName)
        }
    }
}
