package dev.gemmabench.imagegen

data class ImageFile(val filename: String, val repository: String, val revision: String,
    val remotePath: String, val bytes: Long, val sha256: String) {
    val url get() = "https://huggingface.co/$repository/resolve/$revision/$remotePath"
}
data class ImageModel(val id: String, val name: String, val role: String, val steps: Int,
    val flux: Boolean, val description: String, val source: String, val license: String,
    val files: List<ImageFile>, val pipeline: ImagePipeline = ImagePipeline.NATIVE,
    val width: Int = 512, val height: Int = width) {
    val bytes get() = files.sumOf { it.bytes }
}
enum class ImagePipeline { NATIVE, FLUX_LITERT, BONSAI_LITERT, IRIS_LITERT, LITERT_PLAN }

/** Pinned repository revisions and LFS SHA-256 metadata, researched 2026-10-23. */
object ImageModels {
    val all = listOf(
    ImageModel("SD_TURBO", "SD-Turbo", "Fast baseline", 1, false,
        "Smaller SD 2.1 model; simple objects work best. Less prompt fidelity and detail than FLUX.",
        "https://huggingface.co/stabilityai/sd-turbo",
        "Stability AI research / non-commercial terms; see source license for commercial use.",
        listOf(ImageFile("sd_turbo-f16-q8_0.gguf", "Green-Sky/SD-Turbo-GGUF", "19a31586d02d64a73b4419bc193b3ecfaf38e1f0", "sd_turbo-f16-q8_0.gguf", 2023745376L, "d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580"))),
    ImageModel("SDXL_TURBO", "SDXL-Turbo", "Quality baseline", 4, false,
        "Better detail and prompt understanding than SD-Turbo; heavier memory use. Not FLUX quality parity.",
        "https://huggingface.co/stabilityai/sdxl-turbo",
        "Stability AI research / non-commercial terms; see source license for commercial use.",
        listOf(ImageFile("sd_xl_turbo_1.0.q8_0.gguf", "OlegSkutte/sdxl-turbo-GGUF", "62a28b8b2c3cb4bf76d9aad45a5b798047dc786c", "sd_xl_turbo_1.0.q8_0.gguf", 4098988672L, "f5659e6cdad699b1432870cc9fe811359e45c8a331687be505d0ba05c8364720"))),
    ImageModel("FLUX_KLEIN", "FLUX.2 Klein 4B", "Experimental · FLUX quality target", 4, true,
        "Actual FLUX family model, quantized to Q4. Requires a Qwen3 encoder and VAE. CPU generation can take several minutes; measure speed and memory on your phone.",
        "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        "Apache 2.0 model and encoder; decoder licensing follows its source repository.",
        listOf(ImageFile("flux-2-klein-4b-Q4_0.gguf", "leejet/FLUX.2-klein-4B-GGUF", "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd", "flux-2-klein-4b-Q4_0.gguf", 2460378560L, "d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4"),
            ImageFile("Qwen3-4B-Q4_K_M.gguf", "unsloth/Qwen3-4B-GGUF", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120", "Qwen3-4B-Q4_K_M.gguf", 2497281312L, "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a"),
            ImageFile("flux2-vae.safetensors", "Comfy-Org/flux2-klein-4B", "533c4cbc3d3b89af1743d9466c434e89810391c0", "split_files/vae/flux2-vae.safetensors", 336211292L, "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3")))
    ) + CommunityImageModels.all
    fun get(id: String) = all.first { it.id == id }
}

data class ImageTransfer(val modelId: String, val filename: String, val bytes: Long,
    val total: Long, val phase: String, val error: String? = null) {
    val active get() = error == null
}
data class ImageResult(val modelId: String, val modelName: String, val word: String, val category: String,
    val prompt: String, val seed: Long, val steps: Int, val totalMs: Long, val nativeMs: Long,
    val peakRssKb: Long, val imagePath: String, val createdAtEpochMs: Long,
    val modelHashes: Map<String, String>, val device: String,
    val backend: String = "ARM CPU", val width: Int = 512, val height: Int = 512,
    val runtime: String = "stable-diffusion.cpp 95ff76230889898e1bf40058ba272a3699b50dc6",
    val loadAndEncodeMs: Long = 0, val denoiseMs: Long = 0, val decodeAndSaveMs: Long = 0)
data class ImageLabState(val ready: Set<String> = emptySet(), val transfers: List<ImageTransfer> = emptyList(),
    val results: List<ImageResult> = emptyList(), val step: Int = 0, val totalSteps: Int = 0) {
    val downloading get() = transfers.any { it.active }
}
