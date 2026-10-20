package dev.gemmabench.download

import java.io.File
import java.io.IOException
import java.security.MessageDigest

/** Separate from Android/native APIs so byte integrity and interrupted files are testable. */
object ModelFileVerifier {
    fun verify(file: File, expectedBytes: Long, expectedSha256: String, onProgress: (Long) -> Unit = {}): String {
        if (!file.isFile || file.length() != expectedBytes) {
            throw IOException("Download is incomplete: expected $expectedBytes bytes, found ${file.length()}. Tap Retry.")
        }
        val digest = MessageDigest.getInstance("SHA-256")
        var read = 0L
        var lastReport = 0L
        file.inputStream().buffered(1024 * 1024).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
                read += count
                if (read - lastReport >= 32L * 1024 * 1024) {
                    onProgress(read)
                    lastReport = read
                }
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        if (read != expectedBytes || !actual.equals(expectedSha256, ignoreCase = true)) {
            throw IOException("Model checksum did not match the official artifact. Tap Retry to download a fresh copy.")
        }
        onProgress(read)
        return actual
    }
}
