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

    @Test fun screenshotsAreExcludedFromPagesAndEligibleCount() {
        val resolver = context.contentResolver
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val library = PhotoLibrary(context)
        val baseline = library.countEligiblePhotos()
        val created = mutableListOf<android.net.Uri>()
        val photos = mutableListOf<Long>()
        fun insert(name: String, folder: String, eligible: Boolean) {
            val uri = requireNotNull(resolver.insert(collection, ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, folder)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }))
            created += uri
            val bitmap = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
            try {
                resolver.openOutputStream(uri)!!.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
            } finally { bitmap.recycle() }
            resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            if (eligible) photos += android.content.ContentUris.parseId(uri)
        }
        try {
            repeat(2) { index ->
                insert("gemma-photo-filter-$index.jpg", "Pictures/GemmaDemoTests", true)
                // More excluded rows than a query page, including renamed screenshots.
                repeat(4) { screenshot ->
                    insert("Screenshot_gemma-$index-$screenshot.jpg", "Pictures/GemmaDemoTests", false)
                    insert("gemma-renamed-$index-$screenshot.jpg", "Pictures/Screenshots", false)
                }
            }
            assertEquals(baseline + 2, library.countEligiblePhotos())
            val first = library.nextBatch(android.content.ContentUris.parseId(created.last()) + 1, 1)
            assertEquals(photos.last(), first.photos.single().id)
            assertTrue(first.hasMore)
            val second = library.nextBatch(first.photos.single().id, 1)
            assertEquals(photos.first(), second.photos.single().id)
            assertEquals(0, first.unsupportedSkipped)
            assertEquals(8, first.skippedPhotos.size)
            assertTrue(first.skippedPhotos.all { it.reason == "screenshot" })
        } finally { created.forEach { resolver.delete(it, null, null) } }
        assertEquals(baseline, library.countEligiblePhotos())
    }

    @Test fun unsupportedSvgDoesNotShortenOrEndPhotoPages() {
        val resolver = context.contentResolver
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val created = mutableListOf<android.net.Uri>()
        val photos = mutableListOf<Long>()
        val baseline = PhotoLibrary(context).countEligiblePhotos()
        try {
            repeat(3) { index ->
                listOf("image/jpeg" to "jpg", "image/svg+xml" to "svg").forEach { (mime, extension) ->
                    val uri = requireNotNull(resolver.insert(collection, ContentValues().apply {
                        put(MediaStore.Images.Media.DISPLAY_NAME, "gemma-format-test-$index.$extension")
                        put(MediaStore.Images.Media.MIME_TYPE, mime)
                        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/GemmaDemoTests")
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }))
                    created += uri
                    resolver.openOutputStream(uri)!!.use { output ->
                        if (extension == "svg") output.write("""<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>""".toByteArray())
                        else {
                            val bitmap = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
                            bitmap.eraseColor(Color.BLUE)
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, output)
                            bitmap.recycle()
                            photos += android.content.ContentUris.parseId(uri)
                        }
                    }
                    resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
                }
            }
            val library = PhotoLibrary(context)
            assertEquals(baseline + 3, library.countEligiblePhotos())
            val first = library.nextBatch(android.content.ContentUris.parseId(created.last()) + 1, 2)
            assertEquals(photos.takeLast(2).reversed(), first.photos.map { it.id })
            assertTrue(first.hasMore)
            assertEquals(2, first.unsupportedSkipped)
            assertEquals(2, first.skippedPhotos.size)
            assertTrue(first.skippedPhotos.all { it.reason == "unsupported_mime_type" })
            val second = library.nextBatch(first.photos.last().id, 1)
            assertEquals(photos.first(), second.photos.single().id)
            assertEquals(1, second.unsupportedSkipped)
        } finally { created.forEach { resolver.delete(it, null, null) } }
    }

    /** Explicit opt-in: replay a bounded range from the user's reported batch, locally. */
    @Test fun replayReportedPhotoRange(): Unit = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("photoReplay") == "true")
        val before = requireNotNull(args.getString("photoBeforeId")).toLong()
        val minimum = requireNotNull(args.getString("photoMinId")).toLong()
        val slot = ModelSlot.valueOf(args.getString("photoModel") ?: "E2B")
        val accelerator = Accelerator.valueOf(args.getString("photoAccelerator") ?: "CPU")
        val storage = LabStorage(context)
        val artifact = storage.models().single { it.slot == slot }
        val library = PhotoLibrary(context)
        val page = library.nextBatch(before, 50)
        val photos = page.photos.filter { it.id >= minimum }
        assertTrue(photos.isNotEmpty())
        val runtime = LiteRtRuntime(context, storage::file)
        val settings = RunSettings(workload = Workload.PROMPT, accelerator = accelerator,
            contextTokens = artifact.metadata?.takeIf { !it.dynamicContext }?.maxContextTokens?.takeIf { it > 0 } ?: 4096,
            outputTokens = 1024, prompt = PhotoObjects.PROMPT)
        val results = mutableListOf<Map<String, Any?>>()
        try {
            withContext(Dispatchers.IO) {
                runtime.resetStop()
                for ((index, photo) in photos.withIndex()) {
                    var raw = ""
                    var attempts = 0
                    var tokens = 0
                    try {
                        val prepared = library.prepare(photo)
                        val jpeg = prepared.file.readBytes()
                        val objects = analyzePhotoResponse(settings,
                            { runtime.analyzePhoto(artifact, it, jpeg) {} },
                            { attempt, _, measurement -> raw = measurement.output.orEmpty(); attempts = attempt; tokens = measurement.metrics.decodeTokens }, {})
                        results += mapOf("id" to photo.id, "objects" to objects, "rawOutput" to raw, "attempts" to attempts, "decodeTokens" to tokens, "status" to "completed")
                        Log.i("GemmaPhotoReplay", "${index + 1}/${photos.size} id=${photo.id} objects=${objects.size} attempts=$attempts decoded=$tokens OK")
                    } catch (e: Exception) {
                        results += mapOf("id" to photo.id, "rawOutput" to raw, "attempts" to attempts, "error" to e.message, "status" to "failed")
                        Log.e("GemmaPhotoReplay", "${index + 1}/${photos.size} id=${photo.id} ${e.javaClass.simpleName}: ${e.message}")
                    }
                    java.io.File(context.filesDir, "photo-replay-report.json").writeText(storage.gson.toJson(results))
                }
            }
        } finally { runtime.unload() }
        assertEquals("Replay report contains failures", 0, results.count { it["status"] == "failed" })
    }

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
            val slot = ModelSlot.valueOf(InstrumentationRegistry.getArguments().getString("photoModel") ?: "E2B")
            val accelerator = Accelerator.valueOf(InstrumentationRegistry.getArguments().getString("photoAccelerator") ?: "CPU")
            val model = app.lab.state.value.models.firstOrNull { it.slot == slot }
            assumeTrue("Install E2B before the photo inference test.", model != null)
            // Portrait image, mug at upper right: swapping X/Y moves it to lower left.
            val bitmap = Bitmap.createBitmap(512, 768, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLUE }
            canvas.drawRoundRect(260f, 100f, 410f, 340f, 24f, 24f, paint)
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 24f
            canvas.drawOval(370f, 140f, 458f, 280f, paint)
            paint.style = Paint.Style.FILL
            paint.color = Color.DKGRAY
            canvas.drawOval(260f, 85f, 410f, 115f, paint)
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
                            workload = Workload.PROMPT, accelerator = accelerator,
                            contextTokens = artifact.metadata?.takeIf { !it.dynamicContext }?.maxContextTokens?.takeIf { it > 0 } ?: 4096,
                            outputTokens = 1024,
                            prompt = PhotoObjects.PROMPT,
                        ), jpeg) { }
                        Log.i("GemmaPhotoTest", "VISION ${index + 1}: ${measurement.elapsedMs} ms, reused=${measurement.engineReused}: ${measurement.output}")
                        val objects = PhotoObjects.parse(measurement.output.orEmpty())
                        assertTrue("Expected a major object for the generated blue mug", objects.isNotEmpty())
                        val expected = ObjectBox("mug", 260f / 512 * 1000, 85f / 768 * 1000,
                            470f / 512 * 1000, 340f / 768 * 1000)
                        assertTrue("Expected a box overlapping the upper-right mug: $objects", objects.any { box ->
                            val intersection = (minOf(box.right, expected.right) - maxOf(box.left, expected.left)).coerceAtLeast(0f) *
                                (minOf(box.bottom, expected.bottom) - maxOf(box.top, expected.top)).coerceAtLeast(0f)
                            val union = (box.right - box.left) * (box.bottom - box.top) +
                                (expected.right - expected.left) * (expected.bottom - expected.top) - intersection
                            intersection / union > 0.35f
                        })
                        assertEquals(index > 0, measurement.engineReused)
                    }
                } finally { runtime.unload() }
            }
        }
    }
}
