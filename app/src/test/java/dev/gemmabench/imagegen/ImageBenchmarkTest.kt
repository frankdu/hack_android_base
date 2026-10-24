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
        assertTrue(run.samples.take(30).all { it.method == ImageMethod.CPU })
        assertTrue(run.samples.drop(30).all { it.method == ImageMethod.GPU })
        assertTrue(run.retainsEngine)
    }

    @Test fun liteRtGpuFailureDoesNotSkipOtherModelPipelines() = runBlocking {
        val models = listOf(ImageModels.get("FLUX_KLEIN_LITERT"), ImageModels.get("IRIS_LITERT"))
        val repo = MemoryRepository(newImageBenchmarkRun("litert-failure", 10, "test", models,
            listOf(ImageMethodSupport(ImageMethod.GPU, true, "test")), 20))
        val runner = ImageBenchmarkRunner(repo, { _, sample ->
            if (sample.modelId == "FLUX_KLEIN_LITERT") error("GPU graph incompatible")
            result(sample) to 100L
        }, {}, betweenSamples = {})
        runner.run(repo.run.id) {}
        assertEquals(1, repo.run.failed)
        assertEquals(29, repo.run.samples.count { it.status == ImageSampleStatus.SKIPPED })
        assertEquals(30, repo.run.successful)
        assertTrue(repo.run.samples.filter { it.modelId == "IRIS_LITERT" }.all { it.status == ImageSampleStatus.SUCCESS })
    }

    @Test fun historicalProtocolKeepsPerImageLoadingBehavior() {
        val historical = plan().copy(protocol = "fresh worker/engine per image; end-to-end latency includes startup/load/encode/denoise/decode/save")
        assertFalse(historical.retainsEngine)
        assertEquals(historical.protocol, historical.copy(status = ImageRunStatus.CANCELED).protocol)
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

    @Test fun allThreeModelsAndBackendsFinishThirtyWordsBeforeUnloading() = runBlocking {
        val models = ImageModels.all.filter { it.pipeline == ImagePipeline.NATIVE }
        assertEquals(3, models.size)
        val allMethods = ImageMethod.entries.map { ImageMethodSupport(it, true, "test") }
        val repo = MemoryRepository(newImageBenchmarkRun("all-combinations", 10, "test", models, allMethods, 20))
        val expected = models.flatMap { model -> ImageMethod.entries.map { model.id to it } }
        val completed = mutableListOf<Pair<String, ImageMethod>>()
        var resident: Pair<String, ImageMethod>? = null
        val words = mutableListOf<String>()
        val runner = ImageBenchmarkRunner(repo, { _, sample ->
            val key = sample.modelId to sample.method
            if (resident == null) resident = key
            assertEquals("Previous combination must unload before a new one loads", resident, key)
            words += sample.input.word
            result(sample) to 100L
        }, {}, betweenSamples = {}, releaseCombination = {
            assertNotNull(resident)
            assertEquals(ImageBenchmarkSuite.prompts.map { it.word }, words)
            completed += requireNotNull(resident)
            resident = null
            words.clear()
        })
        runner.run(repo.run.id) {}
        assertEquals(expected, completed)
        assertNull(resident)
        assertEquals(270, repo.run.successful)
        assertEquals(ImageRunStatus.COMPLETED, repo.run.status)
    }

    @Test fun stopReleasesSessionAndResumeFinishesCombinationBeforeLoadingNext() = runBlocking {
        val repo = MemoryRepository(plan())
        val called = mutableListOf<String>()
        var releases = 0
        var wordsThisSession = 0
        lateinit var runner: ImageBenchmarkRunner
        runner = ImageBenchmarkRunner(repo, { _, sample ->
            called += sample.id
            wordsThisSession++
            if (called.size == 3) runner.stop()
            result(sample) to 100L
        }, {}, betweenSamples = {}, releaseCombination = {
            releases++
            assertEquals(when (releases) { 1 -> 3; 2 -> 27; else -> 30 }, wordsThisSession)
            wordsThisSession = 0
        })
        runner.run(repo.run.id) {}
        assertEquals(1, releases)
        runner.resetStop()
        runner.run(repo.run.id) {}
        assertEquals(3, releases)
        assertEquals(60, called.distinct().size)
        assertEquals(60, called.size)
    }

    @Test fun canceledGenerationAlwaysReleasesWorker() = runBlocking {
        val repo = MemoryRepository(plan())
        var releases = 0
        val runner = ImageBenchmarkRunner(repo, { _, _ -> throw kotlinx.coroutines.CancellationException("stop") },
            {}, betweenSamples = {}, releaseCombination = { releases++ })
        runner.run(repo.run.id) {}
        assertEquals(1, releases)
        assertEquals(ImageRunStatus.CANCELED, repo.run.status)
        assertEquals(60, repo.run.pending)
    }

    @Test fun checkpointFailureStillReleasesWorker() = runBlocking {
        val memory = MemoryRepository(plan())
        val repo = object : ImageBenchmarkRepository {
            override fun load(id: String) = memory.load(id)
            override fun save(run: ImageBenchmarkRun) {
                if (memory.checkpoints >= 2) error("storage full")
                memory.save(run)
            }
        }
        var releases = 0
        val runner = ImageBenchmarkRunner(repo, { _, sample -> result(sample) to 100L }, {},
            betweenSamples = {}, releaseCombination = { releases++ })
        // A generation/checkpoint error is handled per sample, but a second storage failure escapes.
        assertTrue(runCatching { runner.run(memory.run.id) {} }.isFailure)
        assertEquals(1, releases)
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
