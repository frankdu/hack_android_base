package dev.gemmabench.imagegen

import org.junit.Assert.*
import org.junit.Test

class ImageSourceAndBackendTest {
    @Test fun pausedDownloadsDistinguishRetryFromNetworkAndWifiWaits() {
        assertEquals("Connection interrupted · Android will retry", imageDownloadWaitingReason(1))
        assertEquals("Waiting for a network connection", imageDownloadWaitingReason(2))
        assertEquals("Waiting for Wi-Fi", imageDownloadWaitingReason(3))
        assertEquals("Waiting for Android to resume the transfer", imageDownloadWaitingReason(4))
    }

    @Test fun activeCatalogHasFourLiteRtModelsAndRetainsIrisOnlyForHistoricalLookup() {
        assertEquals(setOf("FLUX_KLEIN_LITERT", "BONSAI_LITERT", "SD_TURBO_LITERT", "SDXL_TURBO_LITERT"), ImageModels.liteRt.map { it.id }.toSet())
        assertEquals(ImagePipeline.IRIS_LITERT, ImageModels.get("IRIS_LITERT").pipeline)
        assertTrue(ImageModels.liteRt.all { it.pipeline != ImagePipeline.NATIVE && it.files.isNotEmpty() })
    }

    @Test fun communityFilesArePreferredAndMirrorsKeepTheSameIntegrityContract() {
        ImageModels.liteRt.filter { it.pipeline != ImagePipeline.LITERT_PLAN }.forEach { model ->
            model.files.forEach { file ->
                assertTrue(file.locations.first().repository.startsWith("litert-community/"))
                assertEquals(2, file.locations.size)
                assertTrue(file.locations.last().repository.startsWith("frankdu/"))
                assertTrue(file.locations.all { it.revision.matches(Regex("[a-f0-9]{40}")) })
                assertTrue(file.sha256.matches(Regex("[a-f0-9]{64}")))
                if (file.remotePath == "README.md") assertEquals("SOURCE_MODEL_CARD.md", file.locations.last().remotePath)
                else assertEquals(file.remotePath, file.locations.last().remotePath)
            }
        }
        assertTrue(imageDownloadCanFallback(404)); assertTrue(imageDownloadCanFallback(403)); assertTrue(imageDownloadCanFallback(503))
        assertFalse(imageDownloadCanFallback(1006)) // insufficient space
        assertFalse(imageDownloadCanFallback(1007)) // missing storage device
        assertFalse(imageDownloadCanFallback(0)) // Wi-Fi/network wait is not a failed artifact
    }

    @Test fun sdDownloadsIncludeTheExactPortableManifestAndRequiredBundleFiles() {
        ImageModels.liteRt.filter { it.pipeline == ImagePipeline.LITERT_PLAN }.forEach { model ->
            assertEquals("frankdu/${model.name.removeSuffix(" · LiteRT")}-LiteRT", model.files.first().repository)
            assertTrue(model.files.any { it.remotePath == "manifest.json" })
            assertTrue(model.files.any { it.remotePath == "NOTICE" })
            assertTrue(model.files.any { it.remotePath.startsWith("tokenizer0/") })
            assertTrue(model.files.any { it.remotePath.startsWith("graphs/") })
            assertTrue(model.files.all { it.filename == "${model.id}/${it.remotePath}" && it.bytes > 0 })
        }
    }

    @Test fun hardwarePresenceDoesNotMakeAnUncompiledModelAvailableOnTpu() {
        val ids = ImageModels.liteRt.map { it.id }.toSet()
        val methods = imageBackendSupport(ImageModels.liteRt, ids, emptySet(), true, true, true, false, "Tensor_G5")
        assertTrue(methods.first { it.method == ImageMethod.CPU }.supports("SD_TURBO_LITERT"))
        assertTrue(methods.first { it.method == ImageMethod.GPU }.supports("BONSAI_LITERT"))
        val tpu = methods.first { it.method == ImageMethod.TPU }
        assertFalse(tpu.available)
        assertTrue(ids.all { !tpu.supports(it) && tpu.reasonFor(it).contains("No verified") })
        val noGpu = imageBackendSupport(ImageModels.liteRt, ids, ids, true, false, false, false, "Tensor_G5")
        assertFalse(noGpu.first { it.method == ImageMethod.GPU }.supports("FLUX_KLEIN_LITERT"))
        assertFalse(noGpu.first { it.method == ImageMethod.TPU }.available)
    }

    @Test fun benchmarkSchedulesFourModelsAcrossThreeMethodsAndExplainsUnavailablePairs() {
        val ids = ImageModels.liteRt.map { it.id }.toSet()
        val methods = imageBackendSupport(ImageModels.liteRt, ids, setOf("SD_TURBO_LITERT"), true, true, true, false, "Tensor_G5")
        val run = newImageBenchmarkRun("four", 1, "device", ImageModels.liteRt, methods, 20, includeUnavailable = true)
        assertEquals(360, run.samples.size)
        assertEquals(270, run.samples.count { it.status == ImageSampleStatus.PENDING })
        assertEquals(90, run.samples.count { it.status == ImageSampleStatus.SKIPPED })
        assertTrue(run.samples.filter { it.status == ImageSampleStatus.SKIPPED }.all { it.method == ImageMethod.TPU && !it.error.isNullOrBlank() })
        assertFalse(run.samples.any { it.modelId == "IRIS_LITERT" })
        assertEquals(4, run.samples.map { it.modelId }.distinct().size)
    }

    @Test fun aMissingModelDoesNotPretendToHaveAnInstalledBackend() {
        val methods = imageBackendSupport(ImageModels.liteRt, setOf("SD_TURBO_LITERT"), emptySet(), true, true, false, false, null)
        val cpu = methods.first { it.method == ImageMethod.CPU }
        assertTrue(cpu.compatible("BONSAI_LITERT"))
        assertFalse(cpu.supports("BONSAI_LITERT"))
        assertTrue(cpu.reasonFor("BONSAI_LITERT").contains("Download"))
        val run = newImageBenchmarkRun("partial", 1, "device", ImageModels.liteRt, methods, 20, includeUnavailable = true)
        assertEquals(60, run.pending)
        assertEquals(300, run.samples.count { it.status == ImageSampleStatus.SKIPPED })
    }
}
