package dev.gemmabench

import android.Manifest
import android.content.ContentValues
import android.graphics.Bitmap
import android.provider.MediaStore
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.photos.PhotoCacheDatabase
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PeopleBatchInstrumentedTest {
    @Test fun stopsResumesAndReusesPhotoCacheWithoutGemma(): Unit = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resolver = context.contentResolver
        val created = mutableListOf<android.net.Uri>()
        instrumentation.uiAutomation.adoptShellPermissionIdentity(Manifest.permission.READ_MEDIA_IMAGES)
        try {
            repeat(2) { index ->
                val uri = requireNotNull(resolver.insert(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "people-batch-test-$index.jpg")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/GemmaPeopleTests")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }))
                created += uri
                val bitmap = Bitmap.createBitmap(160, 120, Bitmap.Config.ARGB_8888)
                try { resolver.openOutputStream(uri)!!.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) } }
                finally { bitmap.recycle() }
                resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            }
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                val lab = (context.applicationContext as GemmaBenchApplication).lab
                suspend fun idle() = withTimeout(30_000) { while (lab.state.value.unavailable || lab.state.value.people.refreshing) delay(20) }
                idle()
                lab.refreshPeople(); idle()
                scenario.onActivity { lab.findPeople(2); lab.stop() }
                idle()
                assertEquals(2, lab.state.value.people.results.size)
                assertTrue(lab.state.value.people.unfinished)
                assertTrue(lab.state.value.people.results.all { result -> created.any { android.content.ContentUris.parseId(it) == result.photo.id } })
                scenario.onActivity { lab.findPeople(2, resume = true) }
                idle()
                assertTrue(lab.state.value.people.results.all { it.status == "COMPLETED" && it.faces == 0 && !it.cached })
                lab.refreshPeople(); idle()
                scenario.onActivity { lab.findPeople(2) }
                idle()
                assertTrue(lab.state.value.people.results.all { it.status == "COMPLETED" && it.cached })
            }
        } finally {
            val db = PhotoCacheDatabase.getInstance(context)
            for (uri in created) {
                resolver.delete(uri, null, null)
                // The production inventory URI uses external rather than external_primary.
                val id = android.content.ContentUris.parseId(uri)
                val inventoryUri = android.content.ContentUris.withAppendedId(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL), id).toString()
                db.openHelper.writableDatabase.execSQL("DELETE FROM people_photos WHERE uri = ?", arrayOf(inventoryUri))
            }
            instrumentation.uiAutomation.dropShellPermissionIdentity()
        }
    }
}
