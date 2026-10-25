package dev.gemmabench.data

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.IOException

/** Call on an IO dispatcher. Missing illustrations leave the normal word experience available. */
class BundledWordImages(private val assets: AssetManager) {
    fun jpeg(word: String, category: String): ByteArray? = try {
        assets.open(WordImageAsset.path(word, category)).use { it.readBytes() }
    } catch (_: IOException) { null }

    fun bitmap(word: String, category: String, maxSize: Int): Bitmap? = try {
        val path = WordImageAsset.path(word, category)
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        assets.open(path).use { BitmapFactory.decodeStream(it, null, options) }
        options.inSampleSize = 1
        while (maxOf(options.outWidth, options.outHeight) / (options.inSampleSize * 2) >= maxSize.coerceAtLeast(1)) {
            options.inSampleSize *= 2
        }
        options.inJustDecodeBounds = false
        assets.open(path).use { BitmapFactory.decodeStream(it, null, options) }
    } catch (_: IOException) { null }
}
