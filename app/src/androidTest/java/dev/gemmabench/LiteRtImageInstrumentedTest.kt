package dev.gemmabench

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.Environment
import dev.gemmabench.imagegen.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Read-only dispatch smoke. It is deliberately not evidence of diffusion TPU execution. */
@RunWith(AndroidJUnit4::class)
class LiteRtImageInstrumentedTest {
    @Test fun liteRtDispatchAndMissingBundleAreReportedHonestly() {
        assumeTrue("Opt in with imageLiteRtSmoke true", InstrumentationRegistry.getArguments().getString("imageLiteRtSmoke") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        Environment.create(context, mapOf(Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir)).use { environment ->
            val available = environment.getAvailableAccelerators()
            assertTrue(Accelerator.CPU in available)
            println("LiteRT 2.3.0 · ${Build.SOC_MODEL} · available accelerators: $available")
        }
        val installed = LiteRtImageBundles(context).installed()
        val tpu = imageMethodSupport(context).single { it.method == ImageMethod.TPU }
        assertEquals(installed.isNotEmpty(), tpu.available)
        assertEquals(installed, tpu.modelIds)
        if (installed.isEmpty()) {
            assertThrows(Exception::class.java) { LiteRtImageBundles(context).load("SD_TURBO") }
        }
    }
}
