package dev.gemmabench.photos

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Upsert
import dev.gemmabench.people.*

/** Durable analysis data only: preview files are disposable and are never stored here. */
@Entity(tableName = "photo_analysis", indices = [Index("photoUri")])
data class CachedPhotoAnalysis(
    @PrimaryKey val cacheKey: String,
    val photoUri: String,
    val photoId: Long,
    val photoName: String,
    val mimeType: String?,
    val sourceRevision: String,
    val inputSha256: String?,
    val modelIdentifier: String,
    val modelVariant: String,
    val modelLabel: String,
    val textBackend: String,
    val visionBackend: String?,
    val runtimeVersion: String,
    val requestJson: String,
    val status: String,
    val objectsJson: String,
    val rawOutput: String,
    val width: Int,
    val height: Int,
    val elapsedMs: Double?,
    val measurementsJson: String,
    val attempts: Int,
    val decodeTokens: Int?,
    val outputTokenLimit: Int?,
    val error: String?,
    val errorKind: String?,
    val skippedReason: String?,
    val computedAtEpochMs: Long,
)

@Dao
interface PhotoAnalysisDao {
    @Query("SELECT * FROM photo_analysis WHERE cacheKey = :key")
    suspend fun find(key: String): CachedPhotoAnalysis?

    @Query("SELECT * FROM photo_analysis WHERE photoUri = :uri ORDER BY computedAtEpochMs DESC")
    suspend fun forPhoto(uri: String): List<CachedPhotoAnalysis>

    @Upsert
    suspend fun upsert(result: CachedPhotoAnalysis)
}

@Database(entities = [CachedPhotoAnalysis::class, Person::class, PeoplePhoto::class, PersonFace::class, PersonBlock::class], version = 2, exportSchema = true)
abstract class PhotoCacheDatabase : RoomDatabase() {
    abstract fun photoAnalysis(): PhotoAnalysisDao
    abstract fun people(): PeopleDao

    companion object {
        @Volatile private var instance: PhotoCacheDatabase? = null

        fun getInstance(context: Context): PhotoCacheDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext, PhotoCacheDatabase::class.java, "photo-analysis.db",
            ).addMigrations(PEOPLE_MIGRATION_1_2).build().also { instance = it }
        }
    }
}
