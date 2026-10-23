package dev.gemmabench.settings

import dev.gemmabench.benchmark.*
import dev.gemmabench.tts.SpeechOption

val SpeechOption.settingsLabel: String get() = when (this) {
    SpeechOption.SYSTEM -> "System"
    SpeechOption.KITTEN -> "Kitten TTS"
    SpeechOption.KOKORO -> "Kokoro TTS"
}

fun speechCandidates(preferred: SpeechOption?, unavailable: Set<SpeechOption> = emptySet()): List<SpeechOption> =
    (listOfNotNull(preferred) + listOf(SpeechOption.SYSTEM, SpeechOption.KITTEN, SpeechOption.KOKORO))
        .distinct().filterNot { it in unavailable }

data class LocalLlmChoice(val family: ModelSlot, val slot: ModelSlot, val accelerator: Accelerator) {
    val label: String get() = "${family.label} · ${accelerator.name}"
}

/** Automatic defaults prioritize backend, then family. A saved family takes precedence. */
fun localLlmCandidates(preferred: ModelSlot?, tpu: TpuSupport): List<LocalLlmChoice> {
    val families = listOf(ModelSlot.E2B, ModelSlot.E4B)
    val backends = listOf(Accelerator.TPU, Accelerator.GPU, Accelerator.CPU)
    val pairs = if (preferred in families) {
        (listOf(preferred!!) + families.filterNot { it == preferred }).flatMap { family -> backends.map { family to it } }
    } else backends.flatMap { backend -> families.map { it to backend } }
    return pairs.mapNotNull { (family, backend) ->
        modelSlot(family, backend, tpu)?.let { LocalLlmChoice(family, it, backend) }
    }
}

/** Availability means installed artifact + declared backend support + TPU device/runtime checks. */
fun selectLocalLlm(models: List<ModelArtifact>, tpu: TpuSupport, preferred: ModelSlot? = null,
                   photos: Boolean = false, audio: Boolean = false): LocalLlmChoice? = localLlmCandidates(preferred, tpu).firstOrNull { choice ->
    models.any { it.slot == choice.slot && it.unavailableReason(choice.accelerator, tpu, photos) == null &&
        (!audio || (it.metadata?.audioSupported != false &&
            (it.metadata?.audioBackends.isNullOrEmpty() ||
                visionAccelerator(it.metadata.audioBackends, choice.accelerator, tpu.available) != null))) }
}
