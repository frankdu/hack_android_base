package dev.gemmabench.voice

import java.io.File
import java.io.IOException
import java.security.MessageDigest

/** Only validated, completed descriptions of bundled artwork are cached, never conversation/audio. */
data class WordImageCaption(val caption: String, val summary: String) {
    val context: String get() = "Caption: $caption\nSummary: $summary"

    companion object {
        const val MAX_LENGTH = 1600
        private val fields = Regex("^Caption:\\s*(.+?)\\s*Summary:\\s*(.+)$",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))

        fun parse(raw: String): WordImageCaption? {
            val text = raw.trim()
            if (text.length > MAX_LENGTH) return null
            val match = fields.matchEntire(text) ?: return null
            val caption = match.groupValues[1].trim()
            val summary = match.groupValues[2].trim()
            return if (caption.isBlank() || summary.isBlank()) null else WordImageCaption(caption, summary)
        }
    }
}

/** Content + prompt identity invalidates descriptions when the artwork or caption instructions change. */
class WordImageCaptionCache(private val directory: File) {
    private fun file(jpeg: ByteArray, prompt: String): File {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("word-image-caption-v1\u0000$prompt\u0000".toByteArray(Charsets.UTF_8))
        val key = digest.digest(jpeg).joinToString("") { "%02x".format(it) }
        return File(directory, "$key.txt")
    }

    /** Cache IO is optional: unreadable entries simply get regenerated. Call off the UI thread. */
    fun read(jpeg: ByteArray, prompt: String): WordImageCaption? = try {
        val target = file(jpeg, prompt)
        if (target.isFile && target.length() <= WordImageCaption.MAX_LENGTH * 4L)
            WordImageCaption.parse(target.readText(Charsets.UTF_8)) else null
    } catch (_: IOException) { null }

    fun write(jpeg: ByteArray, prompt: String, caption: WordImageCaption) {
        var temporary: File? = null
        try {
            if (!directory.isDirectory && !directory.mkdirs()) return
            temporary = File.createTempFile("caption-", ".tmp", directory)
            temporary.writeText(caption.context, Charsets.UTF_8)
            temporary.renameTo(file(jpeg, prompt))
        } catch (_: IOException) {
            // A cache failure must not interrupt a conversation or discard the in-memory description.
        } finally { temporary?.delete() }
    }
}
