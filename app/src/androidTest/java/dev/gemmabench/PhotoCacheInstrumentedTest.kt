package dev.gemmabench

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.*
import dev.gemmabench.photos.*
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class PhotoCacheInstrumentedTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val photo = DevicePhoto(42, "content://test/photos/42", "test.jpg", "image/jpeg", "1:2:3")
    private val request = PhotoCacheRequest.create(
        ModelArtifact(ModelSlot.E2B, "e2b.litertlm", 100, "a".repeat(64)),
        RunSettings(prompt = PhotoObjects.PROMPT), "GPU", "test-runtime",
    )
    private val input = PhotoCacheRequest.sha256(byteArrayOf(1, 2, 3))

    @Test fun completedAnalysisSurvivesDatabaseReopenWithAllMeasurements() = runBlocking {
        val name = "photo-cache-test-${UUID.randomUUID()}.db"
        fun open() = Room.databaseBuilder(context, PhotoCacheDatabase::class.java, name).build()
        var db = open()
        try {
            val first = Measurement(NativeMetrics(0.1, 0.2, 30, 50, 100.0, 20.0), 300.0,
                engineLoadMs = 10.0, conversationSetupMs = 20.0, firstVisibleTextMs = 30.0,
                output = "{\"objects\":[")
            val raw = """{"objects":[{"name":"cat","box":[10,20,400,500]}]}"""
            val second = first.copy(elapsedMs = 200.0, engineLoadMs = null, engineReused = true, output = raw)
            val result = PhotoResult(photo, PhotoStatus.COMPLETED, previewPath = "/disposable-preview.jpg",
                width = 640, height = 480, objects = PhotoObjects.parse(raw), rawOutput = raw,
                elapsedMs = 500.0, attempts = 2, decodeTokens = 50, outputTokenLimit = 1024,
                measurements = listOf(PhotoAttempt(1, 1024, first), PhotoAttempt(2, 1024, second)), visionBackend = "CPU")
            val saved = PhotoAnalysisCache(db.photoAnalysis()).save(request, input, result)
            db.close()
            db = open()
            val restored = requireNotNull(PhotoAnalysisCache(db.photoAnalysis()).completed(saved.cacheKey!!, photo))
            assertEquals(saved.copy(previewPath = null, fromCache = true), restored)
            assertNotNull(restored.computedAtEpochMs)
            val row = requireNotNull(db.photoAnalysis().find(saved.cacheKey!!))
            assertEquals(request.modelIdentifier, row.modelIdentifier)
            assertEquals("E2B", row.modelVariant)
            assertEquals("CPU", row.visionBackend)
            assertEquals(input, row.inputSha256)
            assertEquals(request.requestJson, row.requestJson)
            assertNull(PhotoAnalysisCache(db.photoAnalysis()).completed(request.key(photo, "changed"), photo))
        } finally {
            db.close()
            context.deleteDatabase(name)
        }
    }

    @Test fun failuresAndSkipsAreRetainedButDoNotSuppressRetryAndUpsertReplacesFailure() = runBlocking {
        val db = Room.inMemoryDatabaseBuilder(context, PhotoCacheDatabase::class.java).build()
        try {
            val cache = PhotoAnalysisCache(db.photoAnalysis())
            val failed = cache.save(request, input, PhotoResult(photo, PhotoStatus.FAILED,
                rawOutput = "truncated", attempts = 2, elapsedMs = 300.0, error = "bad JSON", errorKind = "model_json"))
            assertNull(cache.completed(failed.cacheKey!!, photo))
            assertEquals("bad JSON", db.photoAnalysis().find(failed.cacheKey!!)?.error)
            val skipped = cache.save(request, null, PhotoResult(photo, PhotoStatus.SKIPPED, skippedReason = "screenshot"))
            assertEquals("screenshot", db.photoAnalysis().find(skipped.cacheKey!!)?.skippedReason)
            assertNull(cache.completed(skipped.cacheKey!!, photo))
            assertEquals(setOf("FAILED", "SKIPPED"), db.photoAnalysis().forPhoto(photo.uri).map { it.status }.toSet())
            val success = cache.save(request, input, PhotoResult(photo, PhotoStatus.COMPLETED, rawOutput = "{\"objects\":[]}"))
            assertEquals(failed.cacheKey, success.cacheKey)
            assertNotNull(cache.completed(success.cacheKey!!, photo))
            assertEquals("COMPLETED", db.photoAnalysis().find(success.cacheKey!!)?.status)
            assertEquals(2, db.photoAnalysis().forPhoto(photo.uri).size)
        } finally { db.close() }
    }

    @Test fun pendingAndCancelledWorkCannotBeSaved() = runBlocking {
        val db = Room.inMemoryDatabaseBuilder(context, PhotoCacheDatabase::class.java).build()
        try {
            for (status in listOf(PhotoStatus.PENDING, PhotoStatus.ANALYZING)) {
                try {
                    PhotoAnalysisCache(db.photoAnalysis()).save(request, input, PhotoResult(photo, status))
                    fail("Must reject $status")
                } catch (_: IllegalArgumentException) { }
            }
            assertNull(db.photoAnalysis().find(request.key(photo, input)))
        } finally { db.close() }
    }
}
