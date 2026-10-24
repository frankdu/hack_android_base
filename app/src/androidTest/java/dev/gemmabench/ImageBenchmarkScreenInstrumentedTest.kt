package dev.gemmabench

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.imagegen.*
import dev.gemmabench.ui.ImageBenchmarkSection
import dev.gemmabench.ui.ImageBenchmarkPreviewActivity
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.Rule
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onLast
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.assertIsDisplayed
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ImageBenchmarkScreenInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext
    @get:Rule val compose = createAndroidComposeRule<ImageBenchmarkPreviewActivity>()
    private fun scrollTo(text: String) {
        compose.onNodeWithText(text).performScrollTo().assertIsDisplayed()
    }
    private fun capture(name: String) {
        val bitmap = requireNotNull(instrumentation.uiAutomation.takeScreenshot())
        File(context.getExternalFilesDir("benchmark-ui"), "$name.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
    }

    @Test fun benchmarkControlsStatisticsGalleryAndHistoryRender() {
        assumeTrue("Enable -e imageBenchmarkUi true on an emulator",
            InstrumentationRegistry.getArguments().getString("imageBenchmarkUi") == "true" && android.os.Build.HARDWARE in listOf("ranchu", "goldfish"))
        val image = File(context.cacheDir, "benchmark-ui-fixture.png")
        val bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(android.graphics.Color.WHITE)
        canvas.drawCircle(64f, 64f, 45f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.RED })
        image.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        val model = ImageModels.get("SD_TURBO")
        val methods = listOf(ImageMethodSupport(ImageMethod.CPU, true, "UI TEST fixture CPU"),
            ImageMethodSupport(ImageMethod.GPU, true, "UI TEST fixture GPU"), ImageMethodSupport(ImageMethod.TPU, false, "No compiled diffusion model"))
        var run = newImageBenchmarkRun("ui-only-new", 1_792_800_000_000, "UI TEST FIXTURE: synthetic statistics, not phone measurements", listOf(model), methods, 20)
        run = run.copy(status = ImageRunStatus.COMPLETED, samples = run.samples.mapIndexed { index, sample ->
            if (index < 6) {
                val result = ImageResult(model.id, model.name, sample.input.word, sample.input.category, sample.input.prompt,
                    sample.input.seed, 1, 1000, 900, 1024, image.path, 1, emptyMap(), "UI TEST FIXTURE", backend = sample.method.detail,
                    loadAndEncodeMs = 100, denoiseMs = 200, decodeAndSaveMs = 700)
                sample.copy(status = ImageSampleStatus.SUCCESS, result = result, elapsedMs = 1100L + index * 100, quality = 4)
            } else sample.copy(status = ImageSampleStatus.SKIPPED, error = "UI fixture does not run inference")
        })
        val old = run.copy(id = "ui-only-old", startedAtEpochMs = run.startedAtEpochMs - 86_400_000,
            device = "OLDER UI TEST FIXTURE: never persisted as benchmark history")
        val state = LabState(images = ImageLabState(ready = setOf(model.id)),
            imageBenchmark = ImageBenchmarkState(runs = listOf(run, old), methods = methods))
        compose.setContent {
            MaterialTheme {
                Column(Modifier.fillMaxSize().systemBarsPadding().verticalScroll(rememberScrollState()).padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("UI TEST FIXTURE · no benchmark images generated")
                    ImageBenchmarkSection(state, (compose.activity.application as GemmaBenchApplication).lab, model.id)
                }
            }
        }
        compose.waitForIdle()
        capture("benchmark-setup")
        scrollTo("Most recent run")
        capture("benchmark-statistics")
        compose.onAllNodesWithText("Quality rating").onFirst().performScrollTo().assertIsDisplayed()
        capture("benchmark-gallery")
        scrollTo("Historical runs")
        capture("benchmark-history")
        compose.onAllNodesWithText("View images and statistics").onLast().performScrollTo().performClick()
        scrollTo("Historical run")
        capture("benchmark-historical-detail")
        scrollTo("Max")
        capture("benchmark-statistics-details")
        image.delete()
    }
}
