package dev.gemmabench

import android.content.ContextWrapper
import android.graphics.BitmapFactory
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.imagegen.*
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class ImageBenchmarkInstrumentedTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun controllerRunsFirstWordCheckpointsAndStopsTheThirtyWordPlan() = runBlocking {
        assumeTrue("Opt in with imageBenchmarkController true",
            InstrumentationRegistry.getArguments().getString("imageBenchmarkController") == "true")
        val app = context.applicationContext as GemmaBenchApplication
        withTimeout(60_000) { while (app.lab.state.value.busy) delay(50) }
        assumeTrue("Install SD-Turbo", "SD_TURBO" in app.lab.state.value.images.ready)
        val previous = app.lab.state.value.imageBenchmark.runs.map { it.id }.toSet()
        app.lab.startImageBenchmark(setOf("SD_TURBO"), setOf(ImageMethod.CPU), 5)
        try {
            withTimeout(180_000) {
                while (app.lab.state.value.imageBenchmark.runs.none { it.id !in previous && (it.successful > 0 || it.failed > 0) }) delay(50)
            }
        } finally {
            app.lab.stop()
            withTimeout(10_000) { while (app.lab.state.value.imageBenchmarkRunning) delay(50) }
        }
        val run = app.lab.state.value.imageBenchmark.runs.first { it.id !in previous }
        assertEquals(30, run.samples.size)
        assertEquals(1, run.successful)
        assertEquals(29, run.pending)
        assertEquals(ImageRunStatus.CANCELED, run.status)
        assertEquals(1, run.groups().single().speed!!.count)
        val store = ImageBenchmarkStore(context)
        store.save(run.copy(device = "VALIDATION SAMPLE · stopped after one image · ${run.device}"))
        assertEquals(1, ImageBenchmarkStore(context).load(run.id).successful)
        Log.i("ImageBenchmarkSmoke", "CONTROLLER PARTIAL RUN ${run.id}: 1 real image, 29 pending, saved and resumable")
        Unit
    }

    @Test fun stopBeforeAndDuringBindingDoesNotLeaveANativeWorkerRunning() = runBlocking {
        assumeTrue("Opt in with imageBenchmarkCancel true",
            InstrumentationRegistry.getArguments().getString("imageBenchmarkCancel") == "true")
        val client = ImageBenchmarkClient(context)
        val model = ImageModels.get("SD_TURBO")
        val sample = newImageBenchmarkRun("cancel-test", 1, "test", listOf(model),
            listOf(ImageMethodSupport(ImageMethod.CPU, true, "CPU")), 3).samples.first()
        val dir = ImageBenchmarkStore(context).images("cancel-smoke-${UUID.randomUUID()}")
        client.stop()
        assertTrue(runCatching { client.generate(sample, dir, 3) { _, _ -> } }.exceptionOrNull() is CancellationException)
        client.resetStop()
        val pending = async { runCatching { client.generate(sample, dir, 3) { _, _ -> } } }
        delay(300)
        client.stop()
        val outcome = withTimeout(10_000) { pending.await() }
        assertTrue(outcome.exceptionOrNull() is CancellationException)
        Log.i("ImageBenchmarkSmoke", "STOP before/during bind passed; parent pid ${Process.myPid()}")
        Unit
    }

    @Test fun checkpointsRecoverAfterProcessExitAndQualityRatingsSurviveReload() {
        val directory = File(context.cacheDir, "image-bench-test-${UUID.randomUUID()}").apply { mkdirs() }
        val isolated = object : ContextWrapper(context) { override fun getFilesDir() = directory }
        try {
            val store = ImageBenchmarkStore(isolated)
            var run = newImageBenchmarkRun("history-test", 100, "test fixture", listOf(ImageModels.get("SD_TURBO")),
                listOf(ImageMethodSupport(ImageMethod.CPU, true, "CPU")), 5)
            val first = run.samples.first()
            val result = ImageResult(first.modelId, first.modelName, first.input.word, first.input.category,
                first.input.prompt, first.input.seed, first.steps, 100, 90, 0, "/test/fixture.png", 100, emptyMap(), "TEST FIXTURE")
            run = run.copy(samples = run.samples.mapIndexed { index, sample -> when (index) {
                0 -> sample.copy(status = ImageSampleStatus.SUCCESS, result = result, elapsedMs = 110)
                1 -> sample.copy(status = ImageSampleStatus.RUNNING)
                else -> sample
            } })
            store.save(run)
            val restored = ImageBenchmarkStore(isolated).runs(recover = true).single()
            assertEquals(ImageRunStatus.INTERRUPTED, restored.status)
            assertEquals(ImageSampleStatus.PENDING, restored.samples[1].status)
            assertEquals(1, restored.successful)
            assertEquals(29, restored.pending)
            store.rate(run.id, first.id, 4)
            assertEquals(4, ImageBenchmarkStore(isolated).load(run.id).samples.first().quality)
            assertTrue(store.json(run.id).contains("\"quality\": 4"))
            assertEquals(restored.samples.first().input.prompt, store.load(run.id).samples.first().input.prompt)
        } finally { directory.deleteRecursively() }
    }

    /** Bounded hardware smoke, not a fabricated 30-image benchmark. -e imageBenchmarkMethod GPU|CPU */
    @Test fun isolatedNativeWorkerReturnsOrContainsADriverFailure() = runBlocking {
        val methodArg = InstrumentationRegistry.getArguments().getString("imageBenchmarkMethod")
        assumeTrue("Opt in with imageBenchmarkMethod", methodArg != null)
        val method = ImageMethod.valueOf(requireNotNull(methodArg))
        val model = ImageModels.get("SD_TURBO")
        assumeTrue("Install SD-Turbo first", model.id in ImageModelStore(context).ready())
        val support = imageMethodSupport(context).first { it.method == method }
        assumeTrue(support.reason, support.available)
        val id = "smoke-${UUID.randomUUID()}"
        val sample = newImageBenchmarkRun(id, System.currentTimeMillis(), "smoke", listOf(model), listOf(support), 3).samples.first()
        val pid = Process.myPid()
        val started = SystemClock.elapsedRealtime()
        val timeout = InstrumentationRegistry.getArguments().getString("imageBenchmarkTimeoutMinutes")?.toInt() ?: 3
        try {
            val result = ImageBenchmarkClient(context).generate(sample, ImageBenchmarkStore(context).images(id), timeout) { step, total ->
                Log.i("ImageBenchmarkSmoke", "$method step $step/$total")
            }
            val bitmap = requireNotNull(BitmapFactory.decodeFile(result.imagePath))
            assertEquals(512, bitmap.width)
            val pixels = IntArray(512 * 512)
            bitmap.getPixels(pixels, 0, 512, 0, 0, 512, 512)
            bitmap.recycle()
            assertTrue("Blank generated image", pixels.toSet().size > 1000)
            assertTrue(result.backend.contains(if (method == ImageMethod.GPU) "Vulkan GPU" else "ARM CPU"))
            Log.i("ImageBenchmarkSmoke", "SUCCESS ${com.google.gson.Gson().toJson(result)} wallMs=${SystemClock.elapsedRealtime() - started}")
        } catch (e: Exception) {
            Log.e("ImageBenchmarkSmoke", "CONTAINED $method failure after ${SystemClock.elapsedRealtime() - started} ms: ${e.message}")
            if (method == ImageMethod.CPU) throw e
            assertTrue(e is ImageWorkerException || e is TimeoutCancellationException)
        }
        assertEquals("UI/test process survived worker termination", pid, Process.myPid())
    }
}
