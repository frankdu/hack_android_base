package dev.gemmabench

import android.graphics.Bitmap
import android.os.Build
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.imagegen.ImageModels
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** UI workflow only: never starts a model download or generation. */
@RunWith(AndroidJUnit4::class)
class ImagesLiteRtScreenInstrumentedTest {
    @get:Rule val compose = createAndroidComposeRule<ResearchLabActivity>()

    @Test fun fourModelsBackendControlsAndComparisonAreAccessible() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("imageLiteRtUi") == "true" &&
            Build.HARDWARE in listOf("ranchu", "goldfish"))
        val app = compose.activity.application as GemmaBenchApplication
        compose.waitUntil(20_000) { app.lab.state.value.device != null && !app.lab.state.value.busy }
        compose.onNodeWithText("Images", useUnmergedTree = true).performClick()
        compose.onNodeWithText("LiteRT image generation").performScrollTo().assertIsDisplayed()
        ImageModels.liteRt.forEach { compose.onNodeWithTag("image-model-${it.id}").assertExists() }
        compose.onNodeWithTag("image-model-IRIS_LITERT").assertDoesNotExist()
        compose.onNodeWithTag("image-model-BONSAI_LITERT").performClick()
        compose.onNodeWithText("CPU", useUnmergedTree = true).assertExists()
        compose.onNodeWithText("GPU", useUnmergedTree = true).assertExists()
        compose.onNodeWithText("TPU", useUnmergedTree = true).assertExists()
        compose.onNodeWithText("Download Bonsai Image 4B · LiteRT").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("1 · Model").performScrollTo()
        ImageModels.liteRt.forEach { compose.onNodeWithTag("image-model-${it.id}").assertIsDisplayed() }
        capture(app, "generate-bonsai")
        compose.onNodeWithText("Compare models").performScrollTo().performClick()
        compose.onNodeWithText("4-model · 30-word benchmark").performScrollTo().assertIsDisplayed()
        ImageModels.liteRt.forEach { compose.onNodeWithTag("benchmark-model-${it.id}").assertExists() }
        compose.onNodeWithTag("benchmark-model-IRIS_LITERT").assertDoesNotExist()
        compose.onNodeWithText("Select all 4").performScrollTo().performClick()
        ImageModels.liteRt.forEach { compose.onNodeWithTag("benchmark-model-${it.id}").assertIsDisplayed() }
        capture(app, "compare-four-models")
    }

    private fun capture(app: GemmaBenchApplication, name: String) {
        compose.waitForIdle()
        val image = requireNotNull(InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot())
        val file = File(app.getExternalFilesDir("litert-ui"), "$name.png")
        file.outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
        image.recycle()
    }
}
