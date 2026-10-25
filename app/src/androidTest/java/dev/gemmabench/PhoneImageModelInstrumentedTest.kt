package dev.gemmabench

import android.graphics.BitmapFactory
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.Gson
import dev.gemmabench.imagegen.*
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Explicit opt-in, one model/backend in a disposable worker; never starts downloads. */
@RunWith(AndroidJUnit4::class)
class PhoneImageModelInstrumentedTest {
    @Test fun stagedPortableFilesMatchPublishedCatalog() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("phoneImageVerify") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = ImageModelStore(context)
        ImageModels.liteRt.filter { it.pipeline == ImagePipeline.LITERT_PLAN }.forEach { model ->
            store.verifyStaged(model)
            assertTrue(model.id in store.ready())
            println("VERIFIED_PUBLISHED_PHONE_MODEL ${model.id}: ${model.bytes} bytes")
        }
    }

    @Test fun stagedPipelineGeneratesOnSelectedHardware() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("phoneImageInference") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = ImageModels.liteRt.single { it.id == args.getString("imageModel") }
        val method = ImageMethod.valueOf(args.getString("imageMethod") ?: "CPU")
        require(method in listOf(ImageMethod.CPU, ImageMethod.GPU))
        val directory = File(context.filesDir, "image-benchmarks/phone-quantization").apply { mkdirs() }
        val public = requireNotNull(context.getExternalFilesDir("phone-image-validation"))
        val report = File(public, "${model.id}-${method.name}.json")
        val client = ImageBenchmarkClient(context)
        try {
            if (model.pipeline == ImagePipeline.LITERT_PLAN) LiteRtImageBundles(context, portable = true).verifyModel(model.id)
            else ImageModelStore(context).verifyStaged(model)
            val input = ImageBenchmarkPrompt("teapot", "Objects", "A blue ceramic teapot on a wooden table, morning light.", 7)
            val sample = ImageBenchmarkSample("phone-quantization", model.id, model.name, method, input, model.steps)
            val result = client.generate(sample, directory, timeoutMinutes = 5) { step, total ->
                println("${model.id}/${method.name}: $step/$total")
            }
            val bitmap = requireNotNull(BitmapFactory.decodeFile(result.imagePath))
            assertEquals(model.width, bitmap.width); assertEquals(model.height, bitmap.height)
            val pixels = IntArray(model.width * model.height)
            bitmap.getPixels(pixels, 0, model.width, 0, 0, model.width, model.height)
            bitmap.recycle()
            assertTrue("Uniform or invalid image", pixels.toSet().size > 1000)
            File(result.imagePath).copyTo(File(public, "${model.id}-${method.name}.png"), overwrite = true)
            report.writeText(Gson().toJson(mapOf("status" to "PASS", "phone" to Build.MODEL,
                "soc" to Build.SOC_MODEL, "fingerprint" to Build.FINGERPRINT, "result" to result)))
            println("PHONE_IMAGE_RESULT ${report.readText()}")
        } catch (e: Throwable) {
            report.writeText(Gson().toJson(mapOf("status" to "FAIL", "phone" to Build.MODEL,
                "soc" to Build.SOC_MODEL, "modelId" to model.id, "method" to method.name,
                "error" to (e.message ?: e.javaClass.simpleName))))
            throw e
        } finally { client.release() }
    }
}
