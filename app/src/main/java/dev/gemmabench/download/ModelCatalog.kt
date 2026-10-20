package dev.gemmabench.download

import dev.gemmabench.benchmark.ModelSlot

/** Public LiteRT Community artifacts, pinned for reproducible benchmark inputs. */
data class DownloadSpec(
    val slot: ModelSlot,
    val repository: String,
    val revision: String,
    val remoteFilename: String,
    val bytes: Long,
    val sha256: String,
) {
    val url: String get() = "https://huggingface.co/$repository/resolve/$revision/$remoteFilename?download=true"
    val temporaryFilename: String get() = "${slot.name.lowercase()}-$revision.download"
}

object ModelCatalog {
    val models = listOf(
        DownloadSpec(
            ModelSlot.E2B, "litert-community/gemma-4-E2B-it-litert-lm",
            "41c5569623a7c189d6eec3e5225260c58cedbce7", "gemma-4-E2B-it.litertlm",
            2_588_147_712L, "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c",
        ),
        DownloadSpec(
            ModelSlot.E4B, "litert-community/gemma-4-E4B-it-litert-lm",
            "e6a3892a8c22cf87ff99583996149cf3cb32452b", "gemma-4-E4B-it.litertlm",
            3_659_530_240L, "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0",
        ),
    )

    fun forSlot(slot: ModelSlot): DownloadSpec = models.first { it.slot == slot }
}

enum class TransferPhase { QUEUED, DOWNLOADING, WAITING, VERIFYING, FAILED, CANCELING }

data class ModelDownload(
    val id: Long,
    val spec: DownloadSpec,
    val phase: TransferPhase,
    val receivedBytes: Long = 0,
    val detail: String? = null,
) {
    val active: Boolean get() = phase != TransferPhase.FAILED
    val fraction: Float get() = (receivedBytes.toDouble() / spec.bytes).coerceIn(0.0, 1.0).toFloat()
}
