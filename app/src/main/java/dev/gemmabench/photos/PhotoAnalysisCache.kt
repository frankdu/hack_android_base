package dev.gemmabench.photos

import com.google.gson.Gson
import dev.gemmabench.benchmark.ModelArtifact
import dev.gemmabench.benchmark.RunSettings
import java.security.MessageDigest

/** Identity includes everything that can affect the output or the meaning of its latency. */
data class PhotoCacheRequest(
    val modelIdentifier: String,
    val modelVariant: String,
    val modelLabel: String,
    val textBackend: String,
    val visionBackend: String?,
    val runtimeVersion: String,
    val requestJson: String,
) {
    fun key(photo: DevicePhoto, inputSha256: String?): String = sha256(Gson().toJson(listOf(
        photo.uri, inputSha256 ?: "source:${photo.sourceRevision}:${photo.mimeType}:${photo.name}",
        modelIdentifier, modelVariant, textBackend, visionBackend, runtimeVersion, requestJson,
    )).toByteArray(Charsets.UTF_8))

    companion object {
        // Bump when preprocessing, response validation, or runtime sampling behavior changes.
        const val PIPELINE_VERSION = 2

        fun create(model: ModelArtifact, settings: RunSettings, visionBackend: String?, runtimeVersion: String) = PhotoCacheRequest(
            modelIdentifier = model.sha256.lowercase(), modelVariant = model.slot.name,
            modelLabel = model.slot.label, textBackend = settings.accelerator.name,
            visionBackend = visionBackend, runtimeVersion = runtimeVersion,
            requestJson = Gson().toJson(linkedMapOf(
                "pipelineVersion" to PIPELINE_VERSION, "settings" to settings,
                "retryPrompt" to PhotoObjects.RETRY_PROMPT, "schema" to PhotoObjects.SCHEMA,
                "preprocessing" to "upright-max1024-jpeg90",
            )),
        )

        fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
            .digest(bytes).joinToString("") { "%02x".format(it) }
    }
}

class PhotoAnalysisCache(private val dao: PhotoAnalysisDao) {
    private val gson = Gson()

    /** Failures remain available for diagnostics, but are retried on a later scan. */
    suspend fun completed(key: String, photo: DevicePhoto): PhotoResult? {
        val row = dao.find(key)?.takeIf { it.status == PhotoStatus.COMPLETED.name } ?: return null
        return PhotoResult(
            photo = photo, status = PhotoStatus.COMPLETED, width = row.width, height = row.height,
            objects = gson.fromJson(row.objectsJson, Array<ObjectBox>::class.java).toList(),
            rawOutput = row.rawOutput, elapsedMs = row.elapsedMs, error = row.error,
            attempts = row.attempts, decodeTokens = row.decodeTokens, outputTokenLimit = row.outputTokenLimit,
            errorKind = row.errorKind, skippedReason = row.skippedReason,
            measurements = gson.fromJson(row.measurementsJson, Array<PhotoAttempt>::class.java).toList(),
            cacheKey = key, fromCache = true, computedAtEpochMs = row.computedAtEpochMs,
            visionBackend = row.visionBackend,
        )
    }

    suspend fun save(request: PhotoCacheRequest, inputSha256: String?, result: PhotoResult): PhotoResult {
        require(result.status in setOf(PhotoStatus.COMPLETED, PhotoStatus.FAILED, PhotoStatus.SKIPPED)) {
            "Only terminal photo results can be cached."
        }
        val key = request.key(result.photo, inputSha256)
        val computedAt = System.currentTimeMillis()
        dao.upsert(CachedPhotoAnalysis(
            cacheKey = key, photoUri = result.photo.uri, photoId = result.photo.id, photoName = result.photo.name,
            mimeType = result.photo.mimeType, sourceRevision = result.photo.sourceRevision, inputSha256 = inputSha256,
            modelIdentifier = request.modelIdentifier, modelVariant = request.modelVariant, modelLabel = request.modelLabel,
            textBackend = request.textBackend, visionBackend = result.visionBackend ?: request.visionBackend,
            runtimeVersion = request.runtimeVersion, requestJson = request.requestJson, status = result.status.name,
            objectsJson = gson.toJson(result.objects), rawOutput = result.rawOutput, width = result.width, height = result.height,
            elapsedMs = result.elapsedMs, measurementsJson = gson.toJson(result.measurements), attempts = result.attempts,
            decodeTokens = result.decodeTokens, outputTokenLimit = result.outputTokenLimit,
            error = result.error, errorKind = result.errorKind, skippedReason = result.skippedReason,
            computedAtEpochMs = computedAt,
        ))
        return result.copy(cacheKey = key, fromCache = false, computedAtEpochMs = computedAt)
    }
}
