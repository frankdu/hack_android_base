package dev.gemmabench

import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class OnnxTelemetryInstrumentedTest {
    @Test fun installedAppDoesNotAutoInitializeOnnxTelemetry() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val providers = context.packageManager.getPackageInfo(
            context.packageName, PackageManager.PackageInfoFlags.of(PackageManager.GET_PROVIDERS.toLong())
        ).providers.orEmpty()
        assertFalse("ONNX telemetry must not construct its broken HTTP client at process startup",
            providers.any { it.name == "ai.onnxruntime.TelemetryInitializer" })
    }
}
