package dev.gemmabench.settings

import dev.gemmabench.benchmark.*
import dev.gemmabench.tts.SpeechOption.*
import org.junit.Assert.*
import org.junit.Test

class DefaultSelectionTest {
    private val g5 = TpuSupport.check("Tensor G5", listOf("arm64-v8a"), true)
    private val noTpu = TpuSupport(reason = "Not supported")
    private fun model(slot: ModelSlot, backends: List<String> = listOf("CPU", "GPU"), vision: Boolean = true) =
        ModelArtifact(slot, slot.filename, 1, "hash", ArtifactMetadata(backends, 4096, true, null, false, vision, listOf("CPU")))

    @Test fun speechDefaultsFollowSystemKittenKokoroOrder() {
        assertEquals(listOf(SYSTEM, KITTEN, KOKORO), speechCandidates(null))
        assertEquals(listOf(KITTEN, KOKORO), speechCandidates(null, setOf(SYSTEM)))
        assertEquals(listOf(KOKORO), speechCandidates(null, setOf(SYSTEM, KITTEN)))
        assertTrue(speechCandidates(null, setOf(SYSTEM, KITTEN, KOKORO)).isEmpty())
    }

    @Test fun explicitSpeechChoiceWinsAndUnavailableChoicesAreSkipped() {
        assertEquals(listOf(KOKORO, SYSTEM, KITTEN), speechCandidates(KOKORO))
        assertEquals(listOf(KITTEN, SYSTEM, KOKORO), speechCandidates(KITTEN))
        assertEquals(listOf(SYSTEM, KOKORO), speechCandidates(KITTEN, setOf(KITTEN)))
    }

    @Test fun automaticLlmCandidatesUseBackendThenFamilyPriority() {
        assertEquals(listOf(
            LocalLlmChoice(ModelSlot.E2B, ModelSlot.E2B_G5, Accelerator.TPU),
            LocalLlmChoice(ModelSlot.E2B, ModelSlot.E2B, Accelerator.GPU),
            LocalLlmChoice(ModelSlot.E4B, ModelSlot.E4B, Accelerator.GPU),
            LocalLlmChoice(ModelSlot.E2B, ModelSlot.E2B, Accelerator.CPU),
            LocalLlmChoice(ModelSlot.E4B, ModelSlot.E4B, Accelerator.CPU),
        ), localLlmCandidates(null, g5))
        // E4B TPU has no published slot, so it cannot be selected accidentally.
        assertFalse(localLlmCandidates(null, g5).any { it.family == ModelSlot.E4B && it.accelerator == Accelerator.TPU })
    }

    @Test fun installedCompatibleTpuWinsOverBothStandardModels() {
        val choice = selectLocalLlm(listOf(model(ModelSlot.E2B), model(ModelSlot.E4B), model(ModelSlot.E2B_G5, listOf("NPU"))), g5)
        assertEquals(Accelerator.TPU, choice?.accelerator)
        assertEquals(ModelSlot.E2B_G5, choice?.slot)
    }

    @Test fun fallbackSkipsMissingWrongGenerationAndUnsupportedBackends() {
        val models = listOf(model(ModelSlot.E2B_G6, listOf("NPU")), model(ModelSlot.E2B, listOf("CPU")), model(ModelSlot.E4B))
        assertEquals(LocalLlmChoice(ModelSlot.E4B, ModelSlot.E4B, Accelerator.GPU), selectLocalLlm(models, g5))
        assertEquals(Accelerator.GPU, selectLocalLlm(models, noTpu)?.accelerator)
        assertEquals(LocalLlmChoice(ModelSlot.E2B, ModelSlot.E2B, Accelerator.CPU),
            selectLocalLlm(listOf(model(ModelSlot.E2B, listOf("CPU")), model(ModelSlot.E4B, listOf("CPU"))), noTpu))
        assertEquals(ModelSlot.E4B, selectLocalLlm(listOf(model(ModelSlot.E4B, listOf("CPU"))), noTpu)?.family)
    }

    @Test fun explicitModelFamilyWinsEvenWhenAnotherFamilyHasTpu() {
        val models = listOf(model(ModelSlot.E2B_G5, listOf("NPU")), model(ModelSlot.E4B, listOf("CPU")))
        assertEquals(LocalLlmChoice(ModelSlot.E4B, ModelSlot.E4B, Accelerator.CPU), selectLocalLlm(models, g5, ModelSlot.E4B))
        assertEquals(ModelSlot.E2B_G5, selectLocalLlm(models.take(1), g5, ModelSlot.E4B)?.slot)
    }

    @Test fun noInstalledModelLeavesDefaultUnavailable() {
        assertNull(selectLocalLlm(emptyList(), g5))
        assertNull(selectLocalLlm(listOf(model(ModelSlot.E2B_G6, listOf("NPU"))), g5))
    }

    @Test fun photoDefaultsRequireVisionAndPreserveBackendPriority() {
        val models = listOf(model(ModelSlot.E2B, vision = false), model(ModelSlot.E4B))
        assertEquals(ModelSlot.E2B, selectLocalLlm(models, noTpu)?.family)
        assertEquals(ModelSlot.E4B, selectLocalLlm(models, noTpu, photos = true)?.family)
    }
}
