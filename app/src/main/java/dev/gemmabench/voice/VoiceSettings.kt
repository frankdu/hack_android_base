package dev.gemmabench.voice

import dev.gemmabench.benchmark.ArtifactMetadata

/** A nonpositive metadata limit is unknown, including E4B's fixed-context report of zero. */
fun ArtifactMetadata?.voiceContextTokens(): Int {
    val limit = this?.maxContextTokens?.takeIf { it > 0 } ?: return 8192
    return if (dynamicContext) minOf(8192, limit) else limit
}
