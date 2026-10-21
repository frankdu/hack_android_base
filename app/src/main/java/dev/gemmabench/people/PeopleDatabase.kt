package dev.gemmabench.people

import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Entity(tableName = "people")
data class Person(@PrimaryKey val id: String, val name: String? = null, val mergedInto: String? = null, val anchorFaceId: String? = null)

@Entity(tableName = "people_photos")
data class PeoplePhoto(
    @PrimaryKey val uri: String, val revision: String, val cacheKey: String,
    val name: String, val status: String, val error: String? = null,
)

@Entity(tableName = "people_faces", indices = [Index("photoUri"), Index("personId")])
data class PersonFace(
    @PrimaryKey val id: String, val photoUri: String, val personId: String?,
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val embedding: ByteArray?, val quality: Float, val modelVersion: String,
    val manual: Boolean = false,
)

@Entity(tableName = "people_blocks", primaryKeys = ["faceId", "personId"])
data class PersonBlock(val faceId: String, val personId: String)

data class FaceAnchor(val personId: String, val photoUri: String, val revision: String, val modelVersion: String, val embedding: ByteArray)
data class StoredVisibleFace(@Embedded val face: VisibleFace, val revision: String)

@Dao
interface PeopleDao {
    @Query("SELECT * FROM people WHERE mergedInto IS NULL ORDER BY id")
    suspend fun people(): List<Person>
    @Query("SELECT * FROM people_faces ORDER BY id")
    suspend fun faces(): List<PersonFace>
    @Query("SELECT * FROM people_faces WHERE id = :id") suspend fun face(id: String): PersonFace?
    @Query("SELECT p.id AS personId, f.photoUri, ph.revision, f.modelVersion, f.embedding FROM people p JOIN people_faces f ON p.anchorFaceId = f.id AND f.personId = p.id JOIN people_photos ph ON ph.uri = f.photoUri WHERE p.mergedInto IS NULL AND f.embedding IS NOT NULL AND ph.status = 'COMPLETED'")
    suspend fun anchors(): List<FaceAnchor>
    @Query("SELECT f.id, f.photoUri, f.personId, f.`left`, f.top, f.`right`, f.bottom, ph.revision FROM people_faces f JOIN people_photos ph ON ph.uri = f.photoUri WHERE ph.status = 'COMPLETED' ORDER BY f.id")
    suspend fun visibleFaces(): List<StoredVisibleFace>
    @Query("SELECT id FROM people_faces WHERE personId = :personId AND embedding IS NOT NULL ORDER BY id LIMIT 1")
    suspend fun replacementAnchor(personId: String): String?
    @Query("SELECT * FROM people_photos WHERE uri = :uri")
    suspend fun photo(uri: String): PeoplePhoto?
    @Query("SELECT * FROM people_faces WHERE photoUri = :uri ORDER BY id")
    suspend fun facesForPhoto(uri: String): List<PersonFace>
    @Query("SELECT * FROM people_blocks WHERE faceId = :faceId")
    suspend fun blocks(faceId: String): List<PersonBlock>
    @Upsert suspend fun save(person: Person)
    @Upsert suspend fun save(photo: PeoplePhoto)
    @Upsert suspend fun save(face: PersonFace)
    @Insert(onConflict = OnConflictStrategy.IGNORE) suspend fun block(block: PersonBlock)
    @Query("DELETE FROM people_faces WHERE photoUri = :uri") suspend fun deletePhotoFaces(uri: String)
    @Query("UPDATE people SET name = :name WHERE id = :id AND mergedInto IS NULL") suspend fun rename(id: String, name: String?)
    @Query("UPDATE people_faces SET personId = :target, manual = 1 WHERE personId = :source") suspend fun moveGroup(source: String, target: String)
    @Query("UPDATE people SET mergedInto = :target WHERE id = :source OR mergedInto = :source") suspend fun alias(source: String, target: String)
    @Query("INSERT OR IGNORE INTO people_blocks (faceId, personId) SELECT faceId, :target FROM people_blocks WHERE personId = :source")
    suspend fun transferBlocks(source: String, target: String)
    @Query("DELETE FROM people_blocks WHERE personId = :id") suspend fun deleteBlocksForPerson(id: String)
    @Query("DELETE FROM people_blocks WHERE faceId = :faceId AND personId = :personId") suspend fun unblock(faceId: String, personId: String)
    @Query("DELETE FROM people_blocks WHERE personId = :target AND faceId IN (SELECT id FROM people_faces WHERE personId = :target)")
    suspend fun unblockAssigned(target: String)
    @Query("DELETE FROM people_blocks WHERE faceId NOT IN (SELECT id FROM people_faces)") suspend fun deleteOrphanBlocks()
    @Query("DELETE FROM people_blocks") suspend fun clearBlocks()
    @Query("DELETE FROM people_faces") suspend fun clearFaces()
    @Query("DELETE FROM people_photos") suspend fun clearPhotos()
    @Query("DELETE FROM people") suspend fun clearPeople()
}

val PEOPLE_MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("CREATE TABLE IF NOT EXISTS people (id TEXT NOT NULL PRIMARY KEY, name TEXT, mergedInto TEXT, anchorFaceId TEXT)")
        db.execSQL("CREATE TABLE IF NOT EXISTS people_photos (uri TEXT NOT NULL PRIMARY KEY, revision TEXT NOT NULL, cacheKey TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, error TEXT)")
        db.execSQL("CREATE TABLE IF NOT EXISTS people_faces (id TEXT NOT NULL PRIMARY KEY, photoUri TEXT NOT NULL, personId TEXT, `left` REAL NOT NULL, `top` REAL NOT NULL, `right` REAL NOT NULL, bottom REAL NOT NULL, embedding BLOB, quality REAL NOT NULL, modelVersion TEXT NOT NULL, manual INTEGER NOT NULL)")
        db.execSQL("CREATE INDEX IF NOT EXISTS index_people_faces_photoUri ON people_faces (photoUri)")
        db.execSQL("CREATE INDEX IF NOT EXISTS index_people_faces_personId ON people_faces (personId)")
        db.execSQL("CREATE TABLE IF NOT EXISTS people_blocks (faceId TEXT NOT NULL, personId TEXT NOT NULL, PRIMARY KEY(faceId, personId))")
    }
}
