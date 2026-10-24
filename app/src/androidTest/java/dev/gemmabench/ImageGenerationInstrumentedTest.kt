package dev.gemmabench

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.imagegen.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Real on-device smoke test: -e imageInference true -e imageModel SD_TURBO. */
@RunWith(AndroidJUnit4::class)
class ImageGenerationInstrumentedTest {
    @Test fun systemDownloadsRecoverAndCancelPartialFiles() {
        assumeTrue("Opt in with -e imageDownloads true",
            InstrumentationRegistry.getArguments().getString("imageDownloads") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = ImageModels.get("SD_TURBO")
        val store = ImageModelStore(context)
        assumeTrue("Run on a device without SD-Turbo installed", model.id !in store.ready())
        try {
            store.enqueue(model, wifiOnly = false)
            var transfers = store.refresh { }
            val deadline = android.os.SystemClock.elapsedRealtime() + 30_000
            while (transfers.none { it.bytes > 0 } && transfers.none { it.error != null } &&
                android.os.SystemClock.elapsedRealtime() < deadline) {
                Thread.sleep(500)
                transfers = store.refresh { }
            }
            assertTrue("No model bytes downloaded: $transfers", transfers.any { it.bytes > 0 && it.active })
            val restored = ImageModelStore(context).refresh { }
            assertEquals(model.id, restored.single().modelId)
            assertTrue(restored.single().bytes > 0)
        } finally { store.cancel(model.id) }
        assertTrue(store.refresh { }.isEmpty())
        assertFalse(java.io.File(store.directory(), model.files.first().filename + ".part").exists())
        assertFalse(model.id in store.ready())
    }

    @Test fun stagedModelGeneratesAnActualConceptImage() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Opt in with -e imageInference true", args.getString("imageInference") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = ImageModels.get(args.getString("imageModel") ?: "SD_TURBO")
        val store = ImageModelStore(context)
        assertTrue("Stage all model files in ${store.directory()}", model.files.all { store.file(it).length() == it.bytes })
        store.verifyStaged(model)
        assertTrue(model.id in store.ready())
        DiffusionNative.resetStop()
        val generator = ImageGenerator(context, store)
        val word = args.getString("imageWord") ?: "apple"
        val category = args.getString("imageCategory") ?: "Fruit"
        val result = generator.generate(model, word, category, WordImagePrompt.build(word, category), 42, model.steps) { step, total ->
            Log.i("ImageInference", "${model.name}: $step/$total")
        }
        val image = requireNotNull(BitmapFactory.decodeFile(result.imagePath))
        assertEquals(512, image.width)
        assertEquals(512, image.height)
        val colors = IntArray(512 * 512)
        image.getPixels(colors, 0, 512, 0, 0, 512, 512)
        image.recycle()
        assertTrue("Output is blank", colors.toSet().size > 1000)
        assertTrue(result.totalMs > 0)
        assertTrue(generator.results().any { it.imagePath == result.imagePath })
        Log.i("ImageInference", "RESULT ${com.google.gson.Gson().toJson(result)}")
    }

    @Test fun stopBeforeLoadDoesNotProduceAnImage() {
        DiffusionNative.resetStop()
        DiffusionNative.requestStop()
        assertNull(DiffusionNative.generate("", "", "", "apple", false, 1, 42, 1) { _, _, _ -> })
        DiffusionNative.resetStop()
    }
}
