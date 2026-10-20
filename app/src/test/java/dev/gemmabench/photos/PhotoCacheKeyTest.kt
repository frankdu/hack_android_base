package dev.gemmabench.photos

import dev.gemmabench.benchmark.*
import org.junit.Assert.*
import org.junit.Test

class PhotoCacheKeyTest {
    private val photo = DevicePhoto(1, "content://media/external/images/media/1", "photo.jpg", "image/jpeg", "1:2:3")
    private val model = ModelArtifact(ModelSlot.E2B, "e2b.litertlm", 100, "a".repeat(64))
    private val settings = RunSettings(prompt = PhotoObjects.PROMPT)
    private val request = PhotoCacheRequest.create(model, settings, "GPU", "0.18.0")
    private val input = PhotoCacheRequest.sha256(byteArrayOf(1, 2, 3))

    @Test fun stableAcrossRequestRecreationAndPhotoRename() {
        assertEquals(request.key(photo, input), PhotoCacheRequest.create(model, settings, "GPU", "0.18.0").key(photo, input))
        assertEquals(request.key(photo, input), request.key(photo.copy(name = "renamed.jpg", sourceRevision = "changed"), input))
        assertEquals(64, request.key(photo, input).length)
    }

    @Test fun imageModelVariantBackendsRuntimeAndAnalysisSettingsInvalidate() {
        val original = request.key(photo, input)
        val alternatives = listOf(
            request.copy(modelIdentifier = "b".repeat(64)), request.copy(modelVariant = ModelSlot.E4B.name),
            request.copy(textBackend = "GPU"), request.copy(visionBackend = "CPU"),
            request.copy(runtimeVersion = "next"),
            PhotoCacheRequest.create(model, settings.copy(prompt = "new prompt"), "GPU", "0.18.0"),
            PhotoCacheRequest.create(model, settings.copy(outputTokens = 512), "GPU", "0.18.0"),
            PhotoCacheRequest.create(model, settings.copy(contextTokens = 4096), "GPU", "0.18.0"),
            request.copy(requestJson = request.requestJson + "pipeline changed"),
        )
        alternatives.forEach { assertNotEquals(original, it.key(photo, input)) }
        assertNotEquals(original, request.key(photo, PhotoCacheRequest.sha256(byteArrayOf(1, 2, 4))))
        assertNotEquals(original, request.key(photo.copy(uri = "content://another-photo"), input))
    }

    @Test fun skippedAndUndecodablePhotosUseSourceRevisionInsteadOfImageHash() {
        assertNotEquals(request.key(photo, input), request.key(photo, null))
        assertNotEquals(request.key(photo, null), request.key(photo.copy(sourceRevision = "2:2:3"), null))
        assertNotEquals(request.key(photo, null), request.key(photo.copy(name = "Screenshot.jpg"), null))
    }
}
