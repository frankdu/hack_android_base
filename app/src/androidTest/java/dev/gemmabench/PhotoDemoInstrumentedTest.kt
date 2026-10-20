package dev.gemmabench

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.ExifInterface
import android.provider.MediaStore
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.*
import dev.gemmabench.data.LabStorage
import dev.gemmabench.photos.*
import dev.gemmabench.runtime.LiteRtRuntime
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PhotoDemoInstrumentedTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext

    @Test fun pagesWithoutDuplicatesAndNormalizesExif() {
        val resolver = context.contentResolver
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val created = mutableListOf<android.net.Uri>()
        try {
            repeat(3) { index ->
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "gemma-demo-test-$index.jpg")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/GemmaDemoTests")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = requireNotNull(resolver.insert(collection, values))
                created += uri
                val bitmap = Bitmap.createBitmap(120, 80, Bitmap.Config.ARGB_8888)
                bitmap.eraseColor(Color.BLUE)
                resolver.openOutputStream(uri)!!.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
                bitmap.recycle()
                resolver.openFileDescriptor(uri, "rw")!!.use { descriptor ->
                    ExifInterface(descriptor.fileDescriptor).apply {
                        setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
                        saveAttributes()
                    }
                }
                resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            }
            val library = PhotoLibrary(context)
            val lastId = android.content.ContentUris.parseId(created.last())
            val first = library.nextBatch(lastId + 1, 2)
            assertEquals(created.takeLast(2).reversed().map { android.content.ContentUris.parseId(it) }, first.photos.map { it.id })
            assertTrue(first.hasMore)
            val second = library.nextBatch(first.photos.last().id, 1)
            assertEquals(android.content.ContentUris.parseId(created.first()), second.photos.single().id)
            assertTrue(first.photos.none { it.id == second.photos.single().id })
            val prepared = library.prepare(first.photos.first())
            try {
                assertEquals(80, prepared.width)
                assertEquals(120, prepared.height)
            } finally { prepared.file.delete() }
            assertTrue(library.nextBatch(0, 10).photos.isEmpty())
        } finally {
            created.forEach { resolver.delete(it, null, null) }
        }
    }

    /** Opt-in native inference on a generated image only; never reads personal photos. */
    @Test fun gemmaVisionProducesObjectBoxes(): Unit = runBlocking {
        assumeTrue("Pass -e photoInference true for the real Gemma vision smoke test.",
            InstrumentationRegistry.getArguments().getString("photoInference") == "true")
        ActivityScenario.launch(MainActivity::class.java).use {
            val app = context.applicationContext as GemmaBenchApplication
            withTimeout(15_000) { while (app.lab.state.value.busy) kotlinx.coroutines.delay(100) }
            val model = app.lab.state.value.models.firstOrNull { it.slot == ModelSlot.E2B }
            assumeTrue("Install E2B before the photo inference test.", model != null)
            val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLUE }
            canvas.drawRoundRect(130f, 140f, 340f, 400f, 24f, 24f, paint)
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 24f
            canvas.drawOval(290f, 180f, 420f, 340f, paint)
            paint.style = Paint.Style.FILL
            paint.color = Color.DKGRAY
            canvas.drawOval(130f, 125f, 340f, 165f, paint)
            val jpeg = java.io.ByteArrayOutputStream().use { output ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, output)
                output.toByteArray()
            }
            bitmap.recycle()
            withContext(Dispatchers.IO) {
                val artifact = requireNotNull(model)
                val runtime = LiteRtRuntime(context, LabStorage(context)::file)
                try {
                    com.google.ai.edge.litertlm.ModelInfo.from(LabStorage(context).file(artifact).absolutePath).use { info ->
                        Log.i("GemmaPhotoTest", "Vision metadata: ${info.supportedBackends(com.google.ai.edge.litertlm.Modality.VISION)}")
                    }
                    runtime.resetStop()
                    repeat(2) { index ->
                        val measurement = runtime.analyzePhoto(artifact, RunSettings(
                            workload = Workload.PROMPT, contextTokens = 4096, outputTokens = 768,
                            prompt = PhotoObjects.PROMPT,
                        ), jpeg) { }
                        Log.i("GemmaPhotoTest", "VISION ${index + 1}: ${measurement.elapsedMs} ms, reused=${measurement.engineReused}: ${measurement.output}")
                        val objects = PhotoObjects.parse(measurement.output.orEmpty())
                        assertTrue("Expected a major object for the generated blue mug", objects.isNotEmpty())
                        assertEquals(index > 0, measurement.engineReused)
                    }
                } finally { runtime.unload() }
            }
        }
    }
}
