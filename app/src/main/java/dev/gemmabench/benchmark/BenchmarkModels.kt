package dev.gemmabench.benchmark

import java.util.UUID

enum class ModelSlot(val label: String, val filename: String, val tpu: Boolean = false) {
    E2B("Gemma 4 E2B", "e2b.litertlm"),
    E4B("Gemma 4 E4B", "e4b.litertlm"),
    E2B_G5("Gemma 4 E2B · Tensor G5", "e2b-g5.litertlm", true),
    E2B_G6("Gemma 4 E2B · Tensor G6", "e2b-g6.litertlm", true),
}

enum class Accelerator { CPU, GPU, TPU }
enum class Workload { FIXED_TOKENS, PROMPT }

data class ModelArtifact(
    val slot: ModelSlot,
    val filename: String,
    val bytes: Long,
    val sha256: String,
    val metadata: ArtifactMetadata? = null,
    // Null in older manifests means the original private import location.
    val downloaded: Boolean = false,
)

data class ArtifactMetadata(
    val textBackends: List<String>,
    val maxContextTokens: Int,
    val dynamicContext: Boolean,
    val minRuntimeVersion: String?,
    val supportsSpeculativeDecoding: Boolean,
    val visionSupported: Boolean? = null,
    val visionBackends: List<String>? = null,
    val audioSupported: Boolean? = null,
    val audioBackends: List<String>? = null,
)

data class RunSettings(
    val accelerator: Accelerator = Accelerator.CPU,
    val workload: Workload = Workload.FIXED_TOKENS,
    val prefillTokens: Int = 256,
    val outputTokens: Int = 256,
    val contextTokens: Int = 2048,
    val repetitions: Int = 3,
    val warmups: Int = 1,
    val cooldownSeconds: Int = 2,
    val speculativeDecoding: Boolean = false,
    val prompt: String = PromptPresets.explain,
) {
    fun validate() {
        require(prefillTokens in 16..4096) { "Input tokens must be between 16 and 4096." }
        require(outputTokens in 16..1024) { "Output tokens must be between 16 and 1024." }
        require(contextTokens in 512..8192) { "Context must be between 512 and 8192 tokens." }
        require(prefillTokens + outputTokens + 64 <= contextTokens) {
            "Context needs space for input, output and template overhead (64 tokens)."
        }
        require(repetitions in 1..10) { "Use 1–10 measured repetitions." }
        require(warmups in 0..3) { "Use 0–3 warm-up runs." }
        require(cooldownSeconds in 0..60) { "Cooldown must be 0–60 seconds." }
        require(prompt.isNotBlank()) { "Enter a prompt." }
        require(prompt.length <= 24000) { "Keep prompts under 24,000 characters." }
        require(!speculativeDecoding || accelerator == Accelerator.GPU) {
            "This prototype enables speculative decoding only for GPU experiments."
        }
    }
}

object PromptPresets {
    const val explain = "Explain how a bicycle stays balanced in about 100 words. Use plain language."
    const val extract = """Extract the event as JSON with title, date, time, location, and items_to_bring.
Message: Our repair workshop is October 24, 2026 at 2 PM in the community garage.
Please bring a bicycle, a helmet, and a water bottle. Reply with JSON only."""
    const val summarize = """Summarize this field note in three short bullet points, then give one next action.
The greenhouse was checked at 9 AM. The north bed soil was dry while the south bed was moist.
Two tomato plants had yellow lower leaves. The water tank was half full. The vent was stuck
closed and the temperature was 32 C. Tools are on the shelf beside the entrance."""
}

data class NativeMetrics(
    val initSeconds: Double?,
    val ttftSeconds: Double?,
    val prefillTokens: Int,
    val decodeTokens: Int,
    val prefillTokensPerSecond: Double?,
    val decodeTokensPerSecond: Double?,
)

fun Double.positiveOrNull(): Double? = takeIf { it.isFinite() && it > 0.0 }

data class EnvironmentSnapshot(
    val thermalStatus: Int,
    val batteryPercent: Int?,
    val batteryTemperatureCelsius: Double?,
    val pluggedIn: Boolean,
    val processPssMb: Double,
    val availableRamMb: Long,
)

data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val soc: String,
    val androidRelease: String,
    val sdk: Int,
    val fingerprint: String,
    val supportedAbis: List<String>,
    val totalRamMb: Long,
    val appVersion: String,
    val runtimeVersion: String,
)

data class RunSample(
    val index: Int,
    val warmup: Boolean,
    val metrics: NativeMetrics,
    val elapsedMs: Double,
    val engineLoadMs: Double?,
    val conversationSetupMs: Double?,
    val firstVisibleTextMs: Double?,
    val engineReused: Boolean,
    val output: String?,
    val before: EnvironmentSnapshot,
    val after: EnvironmentSnapshot,
)

data class BenchmarkResult(
    val id: String = UUID.randomUUID().toString(),
    val startedAt: String,
    val model: ModelArtifact,
    val settings: RunSettings,
    val device: DeviceInfo,
    val samples: List<RunSample>,
    val status: String,
    val error: String? = null,
)

data class MetricSummary(val count: Int, val median: Double, val p95: Double, val min: Double, val max: Double)

fun summarize(values: List<Double?>): MetricSummary? {
    val sorted = values.filterNotNull().filter { it.isFinite() && it > 0 }.sorted()
    if (sorted.isEmpty()) return null
    val middle = sorted.size / 2
    val median = if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2 else sorted[middle]
    val p95Index = (kotlin.math.ceil(sorted.size * 0.95).toInt() - 1).coerceIn(sorted.indices)
    return MetricSummary(sorted.size, median, sorted[p95Index], sorted.first(), sorted.last())
}

fun BenchmarkResult.measuredSamples(): List<RunSample> = samples.filterNot { it.warmup }

/** A per-model failure keeps preceding samples and does not abort the other model. */
interface InferenceRuntime {
    suspend fun measure(model: ModelArtifact, settings: RunSettings, onText: (String) -> Unit): Measurement
    fun requestStop()
    fun resetStop()
    fun unload()
}

data class Measurement(
    val metrics: NativeMetrics,
    val elapsedMs: Double,
    val engineLoadMs: Double? = null,
    val conversationSetupMs: Double? = null,
    val firstVisibleTextMs: Double? = null,
    val engineReused: Boolean = false,
    val output: String? = null,
)
