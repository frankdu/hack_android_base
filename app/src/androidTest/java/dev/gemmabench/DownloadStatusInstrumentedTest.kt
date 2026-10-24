package dev.gemmabench

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.ModelSlot
import dev.gemmabench.download.ModelCatalog
import dev.gemmabench.download.ModelDownload
import dev.gemmabench.download.TransferPhase
import dev.gemmabench.imagegen.ImageLabState
import dev.gemmabench.imagegen.ImageTransfer
import dev.gemmabench.ui.ImageBenchmarkPreviewActivity
import dev.gemmabench.ui.StatusCard
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CopyOnWriteArrayList

@RunWith(AndroidJUnit4::class)
class DownloadStatusInstrumentedTest {
    @get:Rule val compose = createAndroidComposeRule<ImageBenchmarkPreviewActivity>()
    private val banner = "Model downloads in progress · you can leave the app."
    private val progressBar = SemanticsMatcher.keyIsDefined(SemanticsProperties.ProgressBarRangeInfo)

    @Test fun downloadAndVerificationTransitionsKeepHeaderHeightAndTextStable() {
        val image = ImageTransfer("SD_TURBO_LITERT", "graphs/clip.tflite", 10, 100, "Downloading")
        val llm = ModelDownload(1, ModelCatalog.forSlot(ModelSlot.E2B), TransferPhase.DOWNLOADING)
        val state = mutableStateOf(LabState(images = ImageLabState(transfers = listOf(image))))
        compose.setContent {
            MaterialTheme { Column { StatusCard(state.value) {} } }
        }
        val height = compose.onNodeWithText(banner).onParent().getUnclippedBoundsInRoot().let { it.bottom - it.top }
        // Include the old polling busy pulse, a small file check, the final bundle check,
        // and LLM verification. None should insert a global spinner or resize the header.
        val transitions = listOf(
            state.value.copy(busy = true, progress = "Verifying previous-file…"),
            state.value.copy(imageDownloadVerification = image.filename,
                images = ImageLabState(transfers = listOf(image.copy(phase = "Verifying checksum")))),
            LabState(imageDownloadVerification = "manifest.json"),
            LabState(downloads = listOf(llm)),
            LabState(busy = true, downloads = listOf(llm.copy(phase = TransferPhase.VERIFYING)),
                progress = "Verifying E2B…"),
        )
        transitions.forEach { next ->
            compose.runOnIdle { state.value = next }
            compose.onNodeWithText(banner).assertIsDisplayed()
            compose.onAllNodes(progressBar).assertCountEquals(0)
            assertEquals(height, compose.onNodeWithText(banner).onParent().getUnclippedBoundsInRoot().let { it.bottom - it.top })
        }
        compose.runOnIdle { state.value = LabState(busy = true, progress = "Loading model…") }
        compose.onNodeWithText("Loading model…").assertIsDisplayed()
        compose.onAllNodes(progressBar).assertCountEquals(1)
    }

    /** Reads existing downloads only; never starts, cancels or retries a transfer. */
    @Test fun ordinaryImageDownloadPollingDoesNotPublishForegroundBusy(): Unit = runBlocking {
        val app = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as GemmaBenchApplication
        ActivityScenario.launch(ResearchLabActivity::class.java).use {
            withTimeout(30_000) { while (app.lab.state.value.device == null || app.lab.state.value.busy) delay(50) }
            val pulses = CopyOnWriteArrayList<LabState>()
            val observer = launch(Dispatchers.Unconfined, start = CoroutineStart.UNDISPATCHED) {
                app.lab.state.collect { state ->
                    // LLM installation is a real serialized native action; image polling is not.
                    if (state.busy && state.downloads.none { it.phase == TransferPhase.VERIFYING }) pulses += state
                }
            }
            try { delay(4_500) } finally { observer.cancelAndJoin() }
            assertTrue("Background polls published busy states: ${pulses.map { it.progress }}", pulses.isEmpty())
        }
    }
}
