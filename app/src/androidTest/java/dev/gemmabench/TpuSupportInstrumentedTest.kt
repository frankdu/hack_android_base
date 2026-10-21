package dev.gemmabench

import android.os.Build
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.ModelSlot
import dev.gemmabench.benchmark.TpuSupport
import dev.gemmabench.runtime.LiteRtRuntime
import dev.gemmabench.data.LabStorage
import java.io.File
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TpuSupportInstrumentedTest {
    /** Opt-in system enqueue/cancel smoke check, including a network-waiting TPU transfer. */
    @Test fun tpuDownloadQueuesAndCancels(): Unit = runBlocking {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(arguments.getString("tpuDownloadSmoke") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val app = instrumentation.targetContext.applicationContext as GemmaBenchApplication
        ActivityScenario.launch(ResearchLabActivity::class.java).use {
            val lab = app.lab
            withTimeout(15_000) { while (lab.state.value.busy || lab.state.value.device == null) delay(100) }
            assumeTrue(lab.state.value.tpuSupport.available)
            val slot = requireNotNull(lab.state.value.tpuSupport.slot)
            val existing = lab.state.value.downloads.firstOrNull { it.spec.slot == slot }
            // An existing transfer is only adopted with its explicitly supplied ID.
            assumeTrue(existing == null || existing.id.toString() == arguments.getString("testDownloadId"))
            if (existing == null) {
                instrumentation.runOnMainSync { lab.downloadModel(slot, wifiOnly = true) }
                withTimeout(15_000) { while (lab.state.value.busy) delay(100) }
            }
            val transfer = requireNotNull(lab.state.value.downloads.firstOrNull { it.spec.slot == slot })
            try {
                assertTrue(transfer.active)
                assertEquals(slot, transfer.spec.slot)
                Log.i("GemmaTpuTest", "TPU enqueue/cancel smoke: $transfer")
            } finally {
                instrumentation.runOnMainSync { lab.cancelDownload(transfer.id) }
                withTimeout(15_000) { while (lab.state.value.downloads.any { it.id == transfer.id }) delay(100) }
            }
        }
    }

    @Test fun detectsRuntimeMigratesMetadataAndRejectsUnsupportedDownloads(): Unit = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val app = instrumentation.targetContext.applicationContext as GemmaBenchApplication
        ActivityScenario.launch(ResearchLabActivity::class.java).use {
            val lab = app.lab
            withTimeout(15_000) { while (lab.state.value.busy || lab.state.value.device == null) delay(100) }
            val support = lab.state.value.tpuSupport
            val library = File(app.applicationInfo.nativeLibraryDir, "libLiteRtDispatch_GoogleTensor.so")
            assertEquals(TpuSupport.deviceSlot(Build.SOC_MODEL), support.slot)
            if (!library.isFile) {
                assertFalse(support.available)
                if (support.slot != null) assertTrue(support.reason.contains("missing"))
            }
            assertEquals(LiteRtRuntime(app, LabStorage(app)::file).tpuSupport, support)
            lab.state.value.models.forEach { model ->
                assertNotNull("Existing model metadata should include vision capability", model.metadata?.visionSupported)
                Log.i("GemmaTpuTest", "${model.slot}: ${model.metadata}")
            }
            Log.i("GemmaTpuTest", "Detected $support")
            val rejected = if (support.available && support.slot == ModelSlot.E2B_G5) ModelSlot.E2B_G6 else ModelSlot.E2B_G5
            val previous = lab.state.value.downloads
            instrumentation.runOnMainSync { lab.downloadModel(rejected, wifiOnly = true) }
            withTimeout(15_000) { while (lab.state.value.busy) delay(100) }
            assertNotNull("Unsupported download should report why it is disabled", lab.state.value.error)
            assertEquals("Unsupported download must not enqueue a transfer", previous, lab.state.value.downloads)
        }
    }
}
