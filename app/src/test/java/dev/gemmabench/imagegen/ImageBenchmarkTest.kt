package dev.gemmabench.imagegen

import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class ImageBenchmarkTest {
    private val methods = listOf(ImageMethodSupport(ImageMethod.CPU, true, "CPU"),
        ImageMethodSupport(ImageMethod.GPU, true, "GPU"), ImageMethodSupport(ImageMethod.TPU, false, "No compiled artifact"))
    private fun plan() = newImageBenchmarkRun("test-run", 10, "test device", listOf(ImageModels.get("SD_TURBO")), methods, 20)
    private class MemoryRepository(var run: ImageBenchmarkRun) : ImageBenchmarkRepository {
        var checkpoints = 0
        override fun load(id: String) = run.also { require(it.id == id) }
        override fun save(run: ImageBenchmarkRun) { this.run = run; checkpoints++ }
    }
    private fun result(sample: ImageBenchmarkSample) = ImageResult(sample.modelId, sample.modelName,
        sample.input.word, sample.input.category, sample.input.prompt, sample.input.seed, sample.steps,
        100, 90, 0, "/test/${sample.id}.png", 100, emptyMap(), "test", backend = sample.method.label)

    @Test fun fixedThirtyWordInputsArePairedAndHistoricalInputsAreFrozen() {
        assertEquals(30, ImageBenchmarkSuite.prompts.size)
        assertEquals(30, ImageBenchmarkSuite.prompts.map { it.word }.distinct().size)
        assertEquals(30, ImageBenchmarkSuite.prompts.map { it.seed }.distinct().size)
        assertTrue(ImageBenchmarkSuite.prompts.all { it.prompt.isNotBlank() && it.prompt.length < 2000 })
        val run = plan()
        assertEquals(60, run.samples.size)
        assertEquals(60, run.samples.map { it.id }.distinct().size)
        assertEquals(ImageMethod.TPU, run.skippedMethods.single().method)
        run.samples.groupBy { it.input.word }.values.forEach {
            assertEquals(2, it.size)
            assertEquals(it[0].input, it[1].input)
            assertEquals(it[0].steps, it[1].steps)
        }
        assertEquals(ImageMethod.CPU, run.samples[0].method)
        assertEquals(ImageMethod.GPU, run.samples[2].method)
    }

    @Test fun statisticsUseArithmeticMeanAndEvenMedianWithoutOverflow() {
        val stats = requireNotNull(ImageSpeedStats.from(listOf(100, 200, 900, 400)))
        assertEquals(4, stats.count)
        assertEquals(400.0, stats.meanMs, 0.0)
        assertEquals(300.0, stats.medianMs, 0.0)
        assertEquals(100, stats.minMs)
        assertEquals(900, stats.maxMs)
        assertNull(ImageSpeedStats.from(emptyList()))
        assertEquals(42.0, ImageSpeedStats.from(listOf(42))!!.medianMs, 0.0)
        assertTrue(ImageSpeedStats.from(listOf(Long.MAX_VALUE, Long.MAX_VALUE))!!.meanMs > 0)
        assertThrows(IllegalArgumentException::class.java) { ImageSpeedStats.from(listOf(-1)) }
    }

    @Test fun failuresAndUnratedImagesAreExcludedFromTheirRespectiveAverages() {
        val sample = plan().samples.first()
        val group = ImageBenchmarkGroup(sample.modelId, sample.modelName, sample.method, listOf(
            sample.copy(status = ImageSampleStatus.SUCCESS, elapsedMs = 100, quality = 5),
            sample.copy(id = "second", status = ImageSampleStatus.SUCCESS, elapsedMs = 300),
            sample.copy(id = "failed", status = ImageSampleStatus.FAILED, elapsedMs = 10000, quality = 1),
        ))
        assertEquals(2, group.speed!!.count)
        assertEquals(200.0, group.speed!!.meanMs, 0.0)
        assertEquals(5.0, group.meanQuality!!, 0.0)
        assertEquals(1, group.ratings.size)
    }

    @Test fun oneClickSchedulesAllThirtyWordsOnBothMethods() = runBlocking {
        val repo = MemoryRepository(plan())
        val called = mutableListOf<String>()
        val runner = ImageBenchmarkRunner(repo, { _, sample -> called += sample.id; result(sample) to 100L }, {}, betweenSamples = {})
        runner.resetStop()
        runner.run(repo.run.id) {}
        assertEquals(60, called.size)
        assertEquals(60, called.distinct().size)
        assertEquals(60, repo.run.successful)
        assertEquals(ImageRunStatus.COMPLETED, repo.run.status)
        assertEquals(0, repo.run.pending)
        assertTrue(repo.checkpoints >= 122)
        assertTrue(repo.run.groups().all { it.speed!!.count == 30 })
    }

    @Test fun gpuFailureIsRecordedAndDoesNotBecomeACpuTiming() = runBlocking {
        val repo = MemoryRepository(plan())
        var gpuCalls = 0
        val runner = ImageBenchmarkRunner(repo, { _, sample ->
            if (sample.method == ImageMethod.GPU) { gpuCalls++; throw ImageWorkerException("driver exited", true) }
            result(sample) to 100L
        }, {}, betweenSamples = {})
        runner.resetStop(); runner.run(repo.run.id) {}
        assertEquals(1, gpuCalls)
        assertEquals(30, repo.run.successful)
        assertEquals(1, repo.run.failed)
        assertEquals(29, repo.run.samples.count { it.status == ImageSampleStatus.SKIPPED })
        assertNull(repo.run.groups().first { it.method == ImageMethod.GPU }.speed)
        assertEquals(ImageRunStatus.COMPLETED, repo.run.status)
    }

    @Test fun tpuFailureSkipsOnlyThatModelAndNeverCreatesCpuResults() = runBlocking {
        val models = ImageModels.all.take(2)
        val methods = listOf(ImageMethodSupport(ImageMethod.CPU, true, "CPU"), ImageMethodSupport(ImageMethod.TPU, true, "AOT"))
        val repo = MemoryRepository(newImageBenchmarkRun("tpu-failure", 10, "test", models, methods, 20))
        var failures = 0
        val runner = ImageBenchmarkRunner(repo, { _, sample ->
            if (sample.method == ImageMethod.TPU && sample.modelId == "SD_TURBO") {
                failures++; throw IllegalStateException("NPU invocation failed")
            }
            result(sample) to 100L
        }, {}, betweenSamples = {})
        runner.resetStop(); runner.run(repo.run.id) {}
        assertEquals(1, failures)
        assertEquals(90, repo.run.successful)
        assertEquals(1, repo.run.failed)
        assertEquals(29, repo.run.samples.count { it.status == ImageSampleStatus.SKIPPED })
        assertNull(repo.run.groups().first { it.method == ImageMethod.TPU && it.modelId == "SD_TURBO" }.speed)
        assertEquals(30, repo.run.groups().first { it.method == ImageMethod.TPU && it.modelId == "SDXL_TURBO" }.speed!!.count)
    }

    @Test fun cancelAndResumeKeepCompletedSamplesAndDoNotRegenerateThem() = runBlocking {
        val repo = MemoryRepository(newImageBenchmarkRun("resume-run", 10, "test", listOf(ImageModels.get("SD_TURBO")), methods.take(1), 20))
        val called = mutableListOf<String>()
        var canceled = 0
        lateinit var runner: ImageBenchmarkRunner
        runner = ImageBenchmarkRunner(repo, { _, sample ->
            called += sample.id
            if (called.size == 3) runner.stop()
            result(sample) to 100L
        }, { canceled++ }, betweenSamples = {})
        runner.resetStop(); runner.run(repo.run.id) {}
        assertEquals(ImageRunStatus.CANCELED, repo.run.status)
        assertEquals(3, repo.run.successful)
        assertEquals(27, repo.run.pending)
        assertTrue(repo.run.resumable)
        assertEquals(1, canceled)
        runner.resetStop(); runner.run(repo.run.id) {}
        assertEquals(30, repo.run.successful)
        assertEquals(30, called.distinct().size)
        assertEquals(30, called.size)
        assertEquals(ImageRunStatus.COMPLETED, repo.run.status)
    }
}
