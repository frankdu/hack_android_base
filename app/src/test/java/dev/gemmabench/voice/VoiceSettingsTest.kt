package dev.gemmabench.voice

import dev.gemmabench.benchmark.ArtifactMetadata
import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceSettingsTest {
    private fun metadata(limit: Int, dynamic: Boolean) = ArtifactMetadata(
        textBackends = listOf("CPU", "GPU"), maxContextTokens = limit,
        dynamicContext = dynamic, minRuntimeVersion = null, supportsSpeculativeDecoding = false,
    )

    @Test fun unknownContextUsesPositiveAssistantDefault() {
        assertEquals(8192, null.voiceContextTokens())
        for (dynamic in listOf(false, true)) {
            assertEquals(8192, metadata(0, dynamic).voiceContextTokens())
            assertEquals(8192, metadata(-1, dynamic).voiceContextTokens())
        }
    }

    @Test fun dynamicContextRespectsModelLimitAndAssistantCap() {
        assertEquals(2048, metadata(2048, true).voiceContextTokens())
        assertEquals(8192, metadata(32003, true).voiceContextTokens())
    }

    @Test fun fixedContextRetainsRequiredSize() {
        assertEquals(4224, metadata(4224, false).voiceContextTokens())
        assertEquals(16384, metadata(16384, false).voiceContextTokens())
    }
}
