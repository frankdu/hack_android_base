package dev.gemmabench.benchmark

/** Prerequisites for the published Google Tensor artifacts; engine initialization is the final check. */
data class TpuSupport(val slot: ModelSlot? = null, val reason: String = "Checking TPU support…") {
    val available: Boolean get() = slot != null && reason.isEmpty()

    companion object {
        fun deviceSlot(soc: String): ModelSlot? = when (soc.uppercase().replace(Regex("[ _-]"), "")) {
            "TENSORG5", "GOOGLETENSORG5" -> ModelSlot.E2B_G5
            "TENSORG6", "GOOGLETENSORG6" -> ModelSlot.E2B_G6
            else -> null
        }

        fun check(soc: String, abis: List<String>, libraryPresent: Boolean, loadError: String? = null): TpuSupport {
            val slot = deviceSlot(soc)
                ?: return TpuSupport(reason = "TPU unavailable: published Gemma TPU models require Tensor G5 or G6.")
            if ("arm64-v8a" !in abis) return TpuSupport(slot, "TPU unavailable: an ARM64 runtime is required.")
            if (!libraryPresent) return TpuSupport(slot, "TPU unavailable: this app build is missing the Tensor runtime libraries.")
            if (loadError != null) return TpuSupport(slot, "TPU unavailable: Tensor runtime could not load ($loadError).")
            return TpuSupport(slot, "")
        }
    }
}

fun ModelArtifact.unavailableReason(accelerator: Accelerator, tpu: TpuSupport, photos: Boolean = false): String? {
    if (accelerator == Accelerator.TPU) {
        if (!tpu.available) return tpu.reason
        if (slot != tpu.slot) return "Use the E2B TPU model matching this device's Tensor generation."
    } else if (slot.tpu) return "Choose the standard model for CPU/GPU."
    val backend = if (accelerator == Accelerator.TPU) "NPU" else accelerator.name
    metadata?.let { meta ->
        if (meta.textBackends.isNotEmpty() && backend !in meta.textBackends) {
            return "This model does not support ${accelerator.name} text inference."
        }
        if (photos) {
            if (meta.visionSupported == false) return "This model has no vision encoder."
            if (visionAccelerator(meta.visionBackends.orEmpty(), accelerator, tpu.available) == null) {
                return "This model's vision encoder requires an unavailable backend."
            }
        }
    }
    return null
}

/** Vision and text may use different backends, as declared by the artifact. No retry/fallback on failure. */
fun visionAccelerator(backends: List<String>, text: Accelerator, tpuAvailable: Boolean): Accelerator? {
    val candidates = if (text == Accelerator.TPU) listOf(Accelerator.TPU, Accelerator.GPU, Accelerator.CPU)
        else listOf(Accelerator.GPU, Accelerator.CPU, Accelerator.TPU)
    return candidates.firstOrNull {
        (it != Accelerator.TPU || tpuAvailable) &&
            (backends.isEmpty() || (if (it == Accelerator.TPU) "NPU" else it.name) in backends)
    }
}

fun modelSlot(family: ModelSlot, accelerator: Accelerator, tpu: TpuSupport): ModelSlot? = when {
    accelerator != Accelerator.TPU -> family.takeIf { !it.tpu }
    family == ModelSlot.E2B && tpu.available -> tpu.slot
    else -> null
}
