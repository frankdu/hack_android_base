package dev.gemmabench.benchmark

import dev.gemmabench.data.CsvExport
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class BenchmarkRunnerTest {
    private val artifact = ModelArtifact(ModelSlot.E2B, "e2b.litertlm", 123, "a".repeat(64))
    private val environment = EnvironmentSnapshot(0, 80, 25.0, false, 128.0, 8000)
    private val device = DeviceInfo("Google", "Pixel", "Tensor G5", "17", 37, "test-build", listOf("arm64-v8a"), 12000, "test", "0.18.0")
    private val metrics = NativeMetrics(2.0, 0.3, 256, 128, 800.0, 20.0)

    private inner class FakeRuntime : InferenceRuntime {
        var calls = 0
        var failOn = -1
        var afterCall: () -> Unit = {}
        override suspend fun measure(model: ModelArtifact, settings: RunSettings, onText: (String) -> Unit): Measurement {
            calls++
            if (calls == failOn) error("Backend initialization failed")
            afterCall()
            return Measurement(metrics.copy(decodeTokensPerSecond = calls.toDouble()), 100.0)
        }
        override fun requestStop() {}
        override fun resetStop() {}
        override fun unload() {}
    }

    @Test fun warmupsArePreservedButExcludedFromSummary() = runTest {
        val runtime = FakeRuntime()
        val runner = BenchmarkRunner(runtime, { device }, { environment })
        val partials = mutableListOf<BenchmarkResult>()
        val result = runner.run(artifact, RunSettings(warmups = 1, repetitions = 3, cooldownSeconds = 0), {}, onSample = partials::add)
        assertEquals(4, runtime.calls)
        assertEquals(4, result.samples.size)
        assertEquals(listOf(2.0, 3.0, 4.0), result.measuredSamples().map { it.metrics.decodeTokensPerSecond })
        assertEquals(3.0, summarize(result.measuredSamples().map { it.metrics.decodeTokensPerSecond })!!.median, 0.0)
        assertTrue(partials.all { it.id == result.id })
        assertEquals("completed", result.status)
    }

    @Test fun failurePreservesCompletedSamplesAndError() = runTest {
        val runtime = FakeRuntime().apply { failOn = 3 }
        val runner = BenchmarkRunner(runtime, { device }, { environment })
        val result = runner.run(artifact, RunSettings(warmups = 1, repetitions = 3, cooldownSeconds = 0), {})
        assertEquals("failed", result.status)
        assertEquals(2, result.samples.size)
        assertEquals(1, result.measuredSamples().size)
        assertEquals("Backend initialization failed", result.error)
    }

    @Test fun stopDiscardsInFlightSampleAndPreventsNextRun() = runTest {
        val runtime = FakeRuntime()
        val runner = BenchmarkRunner(runtime, { device }, { environment })
        runtime.afterCall = { if (runtime.calls == 2) runner.stop() }
        val result = runner.run(artifact, RunSettings(warmups = 0, repetitions = 3, cooldownSeconds = 0), {})
        assertEquals("stopped", result.status)
        assertEquals(2, runtime.calls)
        assertEquals(1, result.samples.size)
        runtime.afterCall = {}
        runner.reset()
        assertEquals("completed", runner.run(artifact, RunSettings(warmups = 0, repetitions = 1, cooldownSeconds = 0), {}).status)
    }

    @Test fun zeroWarmupsAndNoCompletedSamplesDoNotInventNumbers() = runTest {
        val runtime = FakeRuntime().apply { failOn = 1 }
        val result = BenchmarkRunner(runtime, { device }, { environment })
            .run(artifact, RunSettings(warmups = 0), {})
        assertTrue(result.samples.isEmpty())
        assertNull(summarize(result.measuredSamples().map { it.metrics.decodeTokensPerSecond }))
        val csv = CsvExport.encode(listOf(result))
        assertEquals(csv.lineSequence().first().split(",").size, csv.lineSequence().drop(1).first().split(",").size)
        assertTrue(csv.contains("Backend initialization failed"))
    }

    @Test fun statisticsRejectMissingInvalidAndZeroNativeMetrics() {
        val summary = summarize(listOf(null, Double.NaN, Double.POSITIVE_INFINITY, -1.0, 0.0, 10.0, 20.0))!!
        assertEquals(2, summary.count)
        assertEquals(15.0, summary.median, 0.0)
        assertEquals(20.0, summary.p95, 0.0)
        assertNull(0.0.positiveOrNull())
        assertNull(Double.NaN.positiveOrNull())
    }

    @Test fun unsafeOrImpossibleSettingsAreRejected() {
        assertThrows(IllegalArgumentException::class.java) { RunSettings(repetitions = 0).validate() }
        assertThrows(IllegalArgumentException::class.java) { RunSettings(prefillTokens = 4096, contextTokens = 2048).validate() }
        assertThrows(IllegalArgumentException::class.java) { RunSettings(speculativeDecoding = true).validate() }
        assertThrows(IllegalArgumentException::class.java) { RunSettings(prompt = " ").validate() }
        RunSettings(accelerator = Accelerator.GPU, speculativeDecoding = true).validate()
    }

    @Test fun csvQuotesMultilinePromptsAndNeutralizesFormulas() {
        assertEquals("\"say \"\"hello\"\"\nnext\"", CsvExport.escape("say \"hello\"\nnext"))
        assertEquals("\"'=SUM(1,2)\"", CsvExport.escape("=SUM(1,2)"))
    }
}
