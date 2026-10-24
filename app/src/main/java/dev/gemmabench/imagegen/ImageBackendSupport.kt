package dev.gemmabench.imagegen

/** Hardware support and installation are separate: a user can choose a backend before downloading. */
fun imageBackendSupport(models: List<ImageModel>, installed: Set<String>, tpuBundles: Set<String>,
    cpu: Boolean, gpu: Boolean, npu: Boolean, nativeGpu: Boolean, soc: String?): List<ImageMethodSupport> {
    val ids = models.map { it.id }.toSet()
    val gpuCompatible = models.filter { if (it.pipeline == ImagePipeline.NATIVE) nativeGpu else gpu }.map { it.id }.toSet()
    val tpuReady = if (npu && soc != null) tpuBundles.intersect(ids) else emptySet()
    return listOf(
        ImageMethodSupport(ImageMethod.CPU, cpu, "LiteRT CPU · 4 threads", if (cpu) installed.intersect(ids) else emptySet(), ids,
            models.associate { it.id to when {
                !cpu -> "CPU runtime unavailable on this device."
                it.id !in installed -> "Download and verify the model first."
                else -> if (it.pipeline == ImagePipeline.NATIVE) "Legacy native CPU" else "LiteRT CPU · 4 threads"
            } }),
        ImageMethodSupport(ImageMethod.GPU, gpu || nativeGpu, "LiteRT FP32 GPU · compatibility checked during generation",
            installed.intersect(gpuCompatible), gpuCompatible,
            models.associate { it.id to when {
                it.id !in gpuCompatible -> "GPU runtime unavailable on this device/build."
                it.id !in installed -> "Download and verify the model first."
                it.pipeline == ImagePipeline.BONSAI_LITERT -> "FP32 GPU denoiser · text encoder and VAE on CPU"
                it.pipeline == ImagePipeline.NATIVE -> "Legacy Vulkan denoiser · text/VAE on CPU"
                else -> "LiteRT FP32 GPU · graph compatibility checked at generation"
            } }),
        ImageMethodSupport(ImageMethod.TPU, tpuReady.isNotEmpty(), "Compiled Tensor TPU graphs + CPU host/partitions", tpuReady, tpuReady,
            models.associate { it.id to when {
                soc == null -> "TPU needs a supported Tensor G5/G6 device."
                !npu -> "Tensor NPU runtime unavailable on this device/build."
                it.id !in tpuReady -> "No verified ${it.width} × ${it.height} $soc TPU bundle installed."
                else -> "Verified $soc TPU bundle · CPU host/unsupported partitions"
            } }),
    )
}
