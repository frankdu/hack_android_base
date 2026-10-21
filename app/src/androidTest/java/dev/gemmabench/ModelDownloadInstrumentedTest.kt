package dev.gemmabench

import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.download.ModelCatalog
import dev.gemmabench.download.TransferPhase
import java.io.File
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Explicitly opt in: this provisions ~6.25 GB of real models over Wi-Fi. */
@RunWith(AndroidJUnit4::class)
class ModelDownloadInstrumentedTest {
    @Test fun publicModelsDownloadVerifyAndInstall(): Unit = runBlocking {
        assumeTrue("Pass -e downloadModels true to provision models on a physical phone.",
            InstrumentationRegistry.getArguments().getString("downloadModels") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val app = instrumentation.targetContext.applicationContext as GemmaBenchApplication
        ActivityScenario.launch(ResearchLabActivity::class.java).use {
            val lab = app.lab
            withTimeout(15_000) { while (lab.state.value.busy || lab.state.value.device == null) delay(100) }
            // TPU provisioning is separately opt-in and only targets this device's supported artifact.
            val specs = ModelCatalog.models.filter { !it.slot.tpu ||
                (InstrumentationRegistry.getArguments().getString("downloadTpu") == "true" &&
                    lab.state.value.tpuSupport.available && it.slot == lab.state.value.tpuSupport.slot) }
            specs.forEach { spec ->
                val installed = lab.state.value.models.any { it.slot == spec.slot && it.sha256 == spec.sha256 }
                val running = lab.state.value.downloads.any { it.spec.slot == spec.slot && it.active }
                if (!installed && !running) {
                    instrumentation.runOnMainSync { lab.downloadModel(spec.slot, wifiOnly = true) }
                    withTimeout(15_000) { while (lab.state.value.busy) delay(100) }
                    assertNull(lab.state.value.error)
                }
            }
            if (InstrumentationRegistry.getArguments().getString("downloadSmokeOnly") == "true") {
                // Exercise system enqueue/recovery and actual received bytes without waiting for all 6.25 GB.
                withTimeout(90_000) {
                    while (lab.state.value.downloads.none { it.receivedBytes > 0 } && lab.state.value.models.isEmpty()) {
                        lab.state.value.downloads.forEach { d -> assertNotEquals(d.detail, TransferPhase.FAILED, d.phase) }
                        delay(500)
                    }
                }
                specs.forEach { spec ->
                    assertTrue("${spec.slot} should be installed or tracked by Android",
                        lab.state.value.models.any { it.slot == spec.slot } ||
                            lab.state.value.downloads.any { it.spec.slot == spec.slot && it.active })
                    val transfer = lab.state.value.downloads.firstOrNull { it.spec.slot == spec.slot }
                    Log.i("GemmaDownloadTest", "${spec.slot}: smoke verified; transfer=$transfer")
                }
                captureScreen(app)
                return@use
            }
            withTimeout(20 * 60 * 1000L) {
                while (specs.any { spec -> lab.state.value.models.none { it.slot == spec.slot && it.sha256 == spec.sha256 } }) {
                    lab.state.value.downloads.forEach { download ->
                        assertNotEquals(download.detail, TransferPhase.FAILED, download.phase)
                        Log.i("GemmaDownloadTest", "${download.spec.slot}: ${download.phase} ${download.receivedBytes}/${download.spec.bytes}")
                    }
                    assertNull(lab.state.value.error)
                    delay(5000)
                }
            }
            specs.forEach { spec ->
                val model = lab.state.value.models.single { it.slot == spec.slot }
                assertEquals(spec.bytes, model.bytes)
                assertEquals(spec.sha256, model.sha256)
                assertNotNull(model.metadata)
                Log.i("GemmaDownloadTest", "${spec.slot}: VERIFIED ${model.bytes} bytes, ${model.metadata}")
            }
            captureScreen(app)
        }
    }

    private fun captureScreen(app: GemmaBenchApplication) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.waitForIdleSync()
        instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
            File(app.getExternalFilesDir(null), "download-test-screen.png").outputStream().use { output ->
                bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, output)
            }
            bitmap.recycle()
        }
    }
}
