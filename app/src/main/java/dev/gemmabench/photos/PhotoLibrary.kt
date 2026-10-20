package dev.gemmabench.photos

import android.Manifest
import android.content.ContentResolver
import android.content.ContentUris
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import java.io.File
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

data class PhotoPage(val photos: List<DevicePhoto>, val hasMore: Boolean)
data class PreparedPhoto(val file: File, val width: Int, val height: Int)

class PhotoLibrary(private val context: Context) {
    private val previews = File(context.cacheDir, "photo-demo")

    /** Keyset pagination avoids skipping/duplicating items when newer photos arrive between batches. */
    fun nextBatch(beforeId: Long?, size: Int): PhotoPage {
        require(size in 1..PhotoDemoState.MAX_BATCH_SIZE) { "Batch size must be 1–${PhotoDemoState.MAX_BATCH_SIZE}." }
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
        val id = MediaStore.Images.Media._ID
        val args = Bundle().apply {
            putStringArray(ContentResolver.QUERY_ARG_SORT_COLUMNS, arrayOf(id))
            putInt(ContentResolver.QUERY_ARG_SORT_DIRECTION, ContentResolver.QUERY_SORT_DIRECTION_DESCENDING)
            putInt(ContentResolver.QUERY_ARG_LIMIT, size + 1)
            if (beforeId != null) {
                putString(ContentResolver.QUERY_ARG_SQL_SELECTION, "$id < ?")
                putStringArray(ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf(beforeId.toString()))
            }
        }
        val found = mutableListOf<DevicePhoto>()
        requireNotNull(context.contentResolver.query(
            collection, arrayOf(id, MediaStore.Images.Media.DISPLAY_NAME), args, null,
        )) { "Android could not read the photo library. Try again." }.use { cursor ->
            while (found.size <= size && cursor.moveToNext()) {
                val photoId = cursor.getLong(0)
                found += DevicePhoto(photoId, ContentUris.withAppendedId(collection, photoId).toString(), cursor.getString(1) ?: "Photo $photoId")
            }
        }
        return PhotoPage(found.take(size), found.size > size)
    }

    /** ImageDecoder applies EXIF orientation before resizing. Only one full decode is held at a time. */
    fun prepare(photo: DevicePhoto): PreparedPhoto {
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
