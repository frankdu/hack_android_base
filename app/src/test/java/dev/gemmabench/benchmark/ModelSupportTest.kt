package dev.gemmabench.benchmark

import org.junit.Assert.*
import org.junit.Test

class ModelSupportTest {
    private val g5 = TpuSupport.check("Tensor G5", listOf("arm64-v8a"), true)
    private fun model(slot: ModelSlot, text: List<String> = listOf("CPU", "GPU"), vision: Boolean = true,
                      visionBackends: List<String> = listOf("CPU")) = ModelArtifact(
        slot, slot.filename, 1, "hash", ArtifactMetadata(text, 4096, true, null, false, vision, visionBackends),
    )

    @Test fun requiresKnownTensorArm64AndLoadableRuntime() {
        assertTrue(g5.available)
        assertEquals(ModelSlot.E2B_G5, g5.slot)
        assertEquals(ModelSlot.E2B_G6, TpuSupport.check("Google_Tensor_G6", listOf("arm64-v8a"), true).slot)
        assertFalse(TpuSupport.check("Tensor G4", listOf("arm64-v8a"), true).available)
        assertFalse(TpuSupport.check("Snapdragon G5", listOf("arm64-v8a"), true).available)
        assertFalse(TpuSupport.check("Tensor G50", listOf("arm64-v8a"), true).available)
        assertFalse(TpuSupport.check("Tensor G5", listOf("x86_64"), true).available)
        assertFalse(TpuSupport.check("Tensor G5", listOf("arm64-v8a"), false).available)
        val broken = TpuSupport.check("Tensor G5", listOf("arm64-v8a"), true, "missing dependency")
        assertFalse(broken.available)
        assertTrue(broken.reason.contains("missing dependency"))
    }

    @Test fun rejectsWrongGenerationStandardTpuAndUnsupportedTextBackend() {
        assertNull(model(ModelSlot.E2B_G5, listOf("NPU")).unavailableReason(Accelerator.TPU, g5))
        assertNotNull(model(ModelSlot.E2B_G6, listOf("NPU")).unavailableReason(Accelerator.TPU, g5))
        assertNotNull(model(ModelSlot.E2B).unavailableReason(Accelerator.TPU, g5))
        assertNotNull(model(ModelSlot.E2B_G5, listOf("NPU")).unavailableReason(Accelerator.CPU, g5))
        assertNotNull(model(ModelSlot.E2B, listOf("CPU")).unavailableReason(Accelerator.GPU, g5))
        assertNull(model(ModelSlot.E2B).unavailableReason(Accelerator.GPU, g5))
    }

    @Test fun photosRequireVisionAndUseDeclaredEncoderBackend() {
        assertNotNull(model(ModelSlot.E2B, vision = false).unavailableReason(Accelerator.CPU, g5, photos = true))
        assertNull(model(ModelSlot.E2B, vision = false).unavailableReason(Accelerator.CPU, g5))
        assertEquals(Accelerator.CPU, visionAccelerator(listOf("CPU"), Accelerator.TPU, true))
        assertEquals(Accelerator.TPU, visionAccelerator(listOf("NPU"), Accelerator.TPU, true))
        assertEquals(Accelerator.TPU, visionAccelerator(listOf("GPU", "NPU"), Accelerator.TPU, true))
        assertNull(visionAccelerator(listOf("NPU"), Accelerator.CPU, false))
        assertNull(visionAccelerator(listOf("UNKNOWN"), Accelerator.GPU, true))
        assertEquals(Accelerator.GPU, visionAccelerator(emptyList(), Accelerator.CPU, false))
        assertEquals(Accelerator.TPU, visionAccelerator(emptyList(), Accelerator.TPU, true))
    }

    @Test fun familyPickerMapsOnlyE2bToMatchingTpuArtifact() {
        assertEquals(ModelSlot.E2B_G5, modelSlot(ModelSlot.E2B, Accelerator.TPU, g5))
        assertNull(modelSlot(ModelSlot.E4B, Accelerator.TPU, g5))
        assertNull(modelSlot(ModelSlot.E2B, Accelerator.TPU, TpuSupport()))
        assertEquals(ModelSlot.E4B, modelSlot(ModelSlot.E4B, Accelerator.GPU, g5))
    }
}
