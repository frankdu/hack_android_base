package dev.gemmabench.download

import dev.gemmabench.benchmark.ModelSlot
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelDownloadTest {
    @get:Rule val directory = TemporaryFolder()
    private val bytes = "downloaded model bytes".toByteArray()
    private val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun file(contents: ByteArray = bytes): File = directory.newFile().apply { writeBytes(contents) }

    @Test fun catalogUsesPinnedOfficialFilesForBothModels() {
        assertEquals(setOf(ModelSlot.E2B, ModelSlot.E4B), ModelCatalog.models.map { it.slot }.toSet())
        ModelCatalog.models.forEach { model ->
            assertTrue(model.url.startsWith("https://huggingface.co/litert-community/"))
            assertTrue(model.url.contains("/resolve/${model.revision}/"))
            assertTrue(model.revision.matches(Regex("[a-f0-9]{40}")))
            assertTrue(model.sha256.matches(Regex("[a-f0-9]{64}")))
            assertTrue(model.bytes > 2_000_000_000L)
            assertTrue(model.remoteFilename.endsWith(".litertlm"))
            assertFalse(model.url.contains("/main/"))
        }
    }

    @Test fun validFileIsVerifiedAndReportsCompletion() {
        val completed = mutableListOf<Long>()
        val source = file()
        assertEquals(sha, ModelFileVerifier.verify(source, bytes.size.toLong(), sha, completed::add))
        assertEquals(listOf(bytes.size.toLong()), completed)
        assertTrue(source.readBytes().contentEquals(bytes))
    }

    @Test fun interruptedDownloadIsRejectedBeforeBeingMarkedReady() {
        val source = file(bytes.copyOf(bytes.size / 2))
        assertThrows(IOException::class.java) { ModelFileVerifier.verify(source, bytes.size.toLong(), sha) }
    }

    @Test fun sameSizeCorruptionIsRejected() {
        val corrupted = bytes.copyOf().also { it[0] = (it[0] + 1).toByte() }
        val source = file(corrupted)
        assertThrows(IOException::class.java) { ModelFileVerifier.verify(source, bytes.size.toLong(), sha) }
        assertTrue(source.exists()) // Caller controls retry/cleanup, never installs a corrupt file.
    }

    @Test fun missingFileIsRejected() {
        val source = File(directory.root, "missing.download")
        assertThrows(IOException::class.java) { ModelFileVerifier.verify(source, bytes.size.toLong(), sha) }
    }

    @Test fun largeTransferProgressUsesLongsAndWaitsForVerification() {
        val model = ModelCatalog.forSlot(ModelSlot.E4B)
        val transfer = ModelDownload(1, model, TransferPhase.DOWNLOADING, receivedBytes = 3_000_000_000L)
        assertTrue(transfer.active)
        assertEquals(3_000_000_000.0 / model.bytes, transfer.fraction.toDouble(), 0.000001)
        assertTrue(transfer.copy(phase = TransferPhase.VERIFYING).active)
        assertFalse(transfer.copy(phase = TransferPhase.FAILED).active)
        assertEquals(1f, transfer.copy(receivedBytes = Long.MAX_VALUE).fraction)
        assertEquals(0f, transfer.copy(receivedBytes = -10).fraction)
    }
}
