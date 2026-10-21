package dev.gemmabench

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonParser
import dev.gemmabench.people.*
import dev.gemmabench.photos.*
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class PeopleRepositoryInstrumentedTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private fun photo(id: Int, revision: String = "1") = DevicePhoto(id.toLong(), "content://test/$id", "$id.jpg", "image/jpeg", revision)
    private fun face(a: Float = 1f, b: Float = 0f) = DetectedFace(100f, 100f, 500f, 500f, FaceMatching.normalize(floatArrayOf(a, b)), 0.99f)
    private val policy = MatchPolicy(0.8f, 0.05f, true)

    @Test fun identityCorrectionsAndPrivacySurviveReopen() = runBlocking {
        val name = "people-test-${UUID.randomUUID()}.db"
        fun open() = Room.databaseBuilder(context, PhotoCacheDatabase::class.java, name).addMigrations(PEOPLE_MIGRATION_1_2).build()
        var db = open()
        try {
            var repo = PeopleRepository(db)
            val photos = (1..3).map { photo(it) }
            val access = photos.associate { it.uri to it.sourceRevision }
            repo.save(photos[0], listOf(face()), access, policy)
            repo.save(photos[1], listOf(face(0.99f, 0.01f)), access, policy)
            assertTrue(repo.cached(photos[0]))
            var groups = repo.gallery(access).first
            assertEquals(1, groups.size)
            val firstId = groups.single().id
            assertEquals(2, groups.single().photoCount)
            repo.rename(firstId, " Alice ")
            val secondFace = groups.single().faces.first { it.photoUri == photos[1].uri }
            repo.move(secondFace.id, null)
            groups = repo.gallery(access).first
            assertEquals(2, groups.size)
            val otherId = groups.first { it.id != firstId }.id
            repo.save(photos[1], listOf(face()), access, policy)
            assertEquals(otherId, db.people().facesForPhoto(photos[1].uri).single().personId)
            assertTrue(db.people().blocks(secondFace.id).any { it.personId == firstId })
            db.close(); db = open(); repo = PeopleRepository(db)
            assertEquals("Alice", repo.gallery(access).first.first { it.id == firstId }.name)
            repo.merge(otherId, firstId)
            assertEquals(1, repo.gallery(access).first.size)
            assertEquals(2, repo.gallery(access).first.single().photoCount)
            assertEquals(1, repo.gallery(mapOf(photos[0].uri to "1")).first.single().photoCount)
            assertTrue(repo.gallery(emptyMap()).first.isEmpty())
            assertTrue(repo.gallery(mapOf(photos[0].uri to "edited")).first.isEmpty())
            assertFalse(repo.cached(photo(1, "edited")))
            repo.clear()
            assertTrue(db.people().faces().isEmpty())
            assertTrue(db.people().people().isEmpty())
        } finally { db.close(); context.deleteDatabase(name) }
    }

    @Test fun failedAndUnavailableFacesCannotAnchorNewMatches() = runBlocking {
        val db = Room.inMemoryDatabaseBuilder(context, PhotoCacheDatabase::class.java).build()
        try {
            val repo = PeopleRepository(db)
            val a = photo(1); val b = photo(2)
            val access = mapOf(a.uri to "1", b.uri to "1")
            repo.save(a, listOf(face()), access, policy)
            repo.failed(a, "decode failed")
            assertFalse(repo.cached(a))
            repo.save(b, listOf(face()), access, policy)
            assertNotEquals(db.people().facesForPhoto(a.uri).single().personId, db.people().facesForPhoto(b.uri).single().personId)
            repo.save(photo(3), listOf(face().copy(embedding = null)), access + (photo(3).uri to "1"), policy)
            assertEquals(1, repo.gallery(access + (photo(3).uri to "1")).second.size)
        } finally { db.close() }
    }

    @Test fun migrationPreservesVersionOneObjectCache() = runBlocking {
        val name = "people-migration-${UUID.randomUUID()}.db"
        val schema = InstrumentationRegistry.getInstrumentation().context.assets
            .open("dev.gemmabench.photos.PhotoCacheDatabase/1.json").bufferedReader().use { JsonParser.parseReader(it).asJsonObject["database"].asJsonObject }
        val legacy = context.openOrCreateDatabase(name, 0, null)
        try {
            for (entity in schema["entities"].asJsonArray) {
                val obj = entity.asJsonObject
                legacy.execSQL(obj["createSql"].asString.replace("\${TABLE_NAME}", obj["tableName"].asString))
                for (index in obj["indices"].asJsonArray) legacy.execSQL(index.asJsonObject["createSql"].asString.replace("\${TABLE_NAME}", obj["tableName"].asString))
            }
            legacy.execSQL("INSERT INTO photo_analysis (cacheKey, photoUri, photoId, photoName, sourceRevision, modelIdentifier, modelVariant, modelLabel, textBackend, runtimeVersion, requestJson, status, objectsJson, rawOutput, width, height, measurementsJson, attempts, computedAtEpochMs) VALUES ('legacy','uri',1,'photo','revision','model','E2B','Gemma','CPU','runtime','{}','COMPLETED','[]','{}',100,100,'[]',1,1)")
            legacy.version = 1
        } finally { legacy.close() }
        val db = Room.databaseBuilder(context, PhotoCacheDatabase::class.java, name).addMigrations(PEOPLE_MIGRATION_1_2).build()
        try {
            assertEquals("photo", db.photoAnalysis().find("legacy")?.photoName)
            assertTrue(db.people().people().isEmpty())
            assertEquals(2, db.openHelper.readableDatabase.version)
        } finally { db.close(); context.deleteDatabase(name) }
    }
}
