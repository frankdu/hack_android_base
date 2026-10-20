package dev.gemmabench.photos

import dev.gemmabench.benchmark.Measurement
import dev.gemmabench.benchmark.RunSettings

class PhotoResponseException(message: String, cause: Throwable? = null) : IllegalArgumentException(message, cause)

/** Retry malformed output once with a fresh conversation and a shorter requested object list. */
suspend fun analyzePhotoResponse(
    settings: RunSettings,
    infer: suspend (RunSettings) -> Measurement,
    onMeasurement: (attempt: Int, settings: RunSettings, measurement: Measurement) -> Unit,
    onRetry: () -> Unit,
): List<ObjectBox> {
    for (attempt in 1..2) {
        val request = if (attempt == 1) settings else settings.copy(prompt = PhotoObjects.RETRY_PROMPT)
        val measurement = infer(request)
        onMeasurement(attempt, request, measurement)
        try {
            return PhotoObjects.parse(measurement.output.orEmpty())
        } catch (e: PhotoResponseException) {
            if (attempt == 2) throw PhotoResponseException("Gemma returned incomplete or malformed JSON after 2 attempts. See raw response.", e)
            onRetry()
        }
    }
    error("Unreachable photo analysis state")
}
