package dev.gemmabench.photos

import android.Manifest
import android.content.ContentResolver
import android.content.ContentUris
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import java.io.File
import java.io.IOException
import kotlin.math.roundToInt

enum class PhotoAccess { FULL, SELECTED, DENIED }

object PhotoPermissions {
    fun requested(): Array<String> = when {
        Build.VERSION.SDK_INT >= 34 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
        Build.VERSION.SDK_INT >= 33 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
        else -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
    }

    fun access(context: Context): PhotoAccess {
        fun granted(permission: String) = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED
        return when {
            Build.VERSION.SDK_INT >= 33 && granted(Manifest.permission.READ_MEDIA_IMAGES) -> PhotoAccess.FULL
            Build.VERSION.SDK_INT >= 34 && granted(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) -> PhotoAccess.SELECTED
            Build.VERSION.SDK_INT <= 32 && granted(Manifest.permission.READ_EXTERNAL_STORAGE) -> PhotoAccess.FULL
            else -> PhotoAccess.DENIED
        }
    }
}

data class SkippedPhoto(val photo: DevicePhoto, val reason: String)
data class PhotoPage(
    val photos: List<DevicePhoto>, val hasMore: Boolean, val unsupportedSkipped: Int = 0,
    val skippedPhotos: List<SkippedPhoto> = emptyList(),
)
data class PreparedPhoto(val file: File, val width: Int, val height: Int)

class PhotoLibrary(private val context: Context) {
    private val previews = File(context.cacheDir, "photo-demo")
    private val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
    private val projection = arrayOf(
        MediaStore.Images.Media._ID, MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.MIME_TYPE,
        MediaStore.Images.Media.RELATIVE_PATH, MediaStore.Images.Media.BUCKET_DISPLAY_NAME,
        MediaStore.Images.Media.GENERATION_MODIFIED, MediaStore.Images.Media.DATE_MODIFIED, MediaStore.Images.Media.SIZE,
    )

    private fun Cursor.isScreenshot() = PhotoEligibility.isScreenshot(getString(1), getString(3), getString(4))
    private fun Cursor.isSupported() = getString(2)?.let(ImageDecoder::isMimeTypeSupported) ?: true

    /** Current permission-scoped inventory, also used to hide revoked/deleted People photos. */
    fun accessiblePhotos(): List<DevicePhoto> {
        val photos = mutableListOf<DevicePhoto>()
        requireNotNull(context.contentResolver.query(collection, projection, null, null, null)).use { cursor ->
            while (cursor.moveToNext()) {
                if (cursor.isScreenshot() || !cursor.isSupported()) continue
                val id = cursor.getLong(0)
                photos += DevicePhoto(id, ContentUris.withAppendedId(collection, id).toString(), cursor.getString(1) ?: "Photo $id",
                    cursor.getString(2), "${cursor.getLong(5)}:${cursor.getLong(6)}:${cursor.getLong(7)}")
            }
        }
        return photos
    }

    /** Counts accessible, supported non-screenshot images without decoding image data. Call off the UI thread. */
    fun countEligiblePhotos(): Long {
        var count = 0L
        requireNotNull(context.contentResolver.query(collection, projection, null, null, null)) {
            "Android could not read the photo library. Try again."
        }.use { cursor ->
            while (cursor.moveToNext()) {
                if (!cursor.isScreenshot() && cursor.isSupported()) count++
            }
        }
        return count
    }

    /** Keyset pagination avoids skipping/duplicating items when newer photos arrive between batches. */
    fun nextBatch(beforeId: Long?, size: Int): PhotoPage {
        require(size in 1..PhotoDemoState.MAX_BATCH_SIZE) { "Batch size must be 1–${PhotoDemoState.MAX_BATCH_SIZE}." }
        val id = MediaStore.Images.Media._ID
        fun queryArgs(cursorBefore: Long?) = Bundle().apply {
            putStringArray(ContentResolver.QUERY_ARG_SORT_COLUMNS, arrayOf(id))
            putInt(ContentResolver.QUERY_ARG_SORT_DIRECTION, ContentResolver.QUERY_SORT_DIRECTION_DESCENDING)
            putInt(ContentResolver.QUERY_ARG_LIMIT, size + 1)
            if (cursorBefore != null) {
                putString(ContentResolver.QUERY_ARG_SQL_SELECTION, "$id < ?")
                putStringArray(ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf(cursorBefore.toString()))
            }
        }
        val found = mutableListOf<DevicePhoto>()
        val skippedPhotos = mutableListOf<SkippedPhoto>()
        var scanBefore = beforeId
        // Scan past excluded rows before deciding whether a page is full/exhausted.
        while (found.size <= size) {
            var scanned = 0
            requireNotNull(context.contentResolver.query(
                collection, projection,
                queryArgs(scanBefore), null,
            )) { "Android could not read the photo library. Try again." }.use { cursor ->
                while (found.size <= size && cursor.moveToNext()) {
                    scanned++
                    val photoId = cursor.getLong(0)
                    scanBefore = photoId
                    val mime = cursor.getString(2)
                    val photo = DevicePhoto(photoId, ContentUris.withAppendedId(collection, photoId).toString(),
                        cursor.getString(1) ?: "Photo $photoId", mime,
                        "${cursor.getLong(5)}:${cursor.getLong(6)}:${cursor.getLong(7)}")
                    if (cursor.isScreenshot()) {
                        skippedPhotos += SkippedPhoto(photo, "screenshot")
                        continue
                    }
                    if (!cursor.isSupported()) {
                        skippedPhotos += SkippedPhoto(photo, "unsupported_mime_type")
                        continue
                    }
                    found += photo
                }
            }
            if (scanned < size + 1) break
        }
        val hasMore = found.size > size
        // Defer lookahead exclusions to the next page, but retain trailing exclusions at end of library.
        val pageSkipped = if (hasMore) skippedPhotos.filter { it.photo.id > found[size - 1].id } else skippedPhotos
        return PhotoPage(found.take(size), hasMore, pageSkipped.count { it.reason == "unsupported_mime_type" }, pageSkipped)
    }

    /** ImageDecoder applies EXIF orientation before resizing. Only one full decode is held at a time. */
    fun prepare(photo: DevicePhoto): PreparedPhoto {
        require(photo.mimeType == null || ImageDecoder.isMimeTypeSupported(photo.mimeType)) {
            "Android cannot decode ${photo.mimeType} images. Choose a supported photo format."
        }
        try {
            return decode(photo)
        } catch (e: IOException) {
            throw IOException("Cannot decode this photo. Its image data may be damaged or unavailable.", e)
        }
    }

    private fun decode(photo: DevicePhoto): PreparedPhoto {
        val source = ImageDecoder.createSource(context.contentResolver, Uri.parse(photo.uri))
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val scale = minOf(1.0, 1024.0 / maxOf(info.size.width, info.size.height))
            decoder.setTargetSize(maxOf(1, (info.size.width * scale).roundToInt()), maxOf(1, (info.size.height * scale).roundToInt()))
        }
        try {
            previews.mkdirs()
            val file = File(previews, "${photo.id}.jpg")
            file.outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it)) { "Could not prepare this photo." } }
            return PreparedPhoto(file, bitmap.width, bitmap.height)
        } finally { bitmap.recycle() }
    }

    fun clearPreviews() { previews.listFiles()?.forEach { it.delete() } }
}
