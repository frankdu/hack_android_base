package dev.gemmabench.imagegen

/** Versioned inputs are saved into every run; later edits cannot change historical comparisons. */
data class ImageBenchmarkPrompt(val word: String, val category: String, val prompt: String, val seed: Long)
object ImageBenchmarkSuite {
    const val VERSION = "word-concepts-30-v1"
    const val MIN_WORD_COUNT = 2
    const val MAX_WORD_COUNT = 30
    private val style = "Clear educational illustration, bright natural colors, soft light, simple background. No writing, labels, watermark or collage."
    val prompts: List<ImageBenchmarkPrompt> = listOf(
        Triple("apple", "Fruit", "One ripe red apple with a short stem and one green leaf, centered on a plain cream background."),
        Triple("banana", "Fruit", "One curved yellow banana, unpeeled, clearly recognizable, centered on a plain background."),
        Triple("orange", "Fruit", "One round orange fruit with textured orange peel and a small green leaf, centered."),
        Triple("strawberry", "Fruit", "One ripe red strawberry with tiny seeds and a green leafy top, centered."),
        Triple("watermelon", "Fruit", "One triangular watermelon slice showing red flesh, black seeds and green rind."),
        Triple("dog", "Animals", "One friendly brown dog sitting, full body visible, with floppy ears, four legs and a wagging tail."),
        Triple("cat", "Animals", "One friendly orange cat sitting, full body visible, with pointed ears, whiskers and a curled tail."),
        Triple("elephant", "Animals", "One gray elephant in side view, full body visible, with a long trunk, large ears and four legs."),
        Triple("giraffe", "Animals", "One giraffe standing, full body visible, with a very long neck and brown patches."),
        Triple("butterfly", "Animals", "One colorful butterfly with two pairs of wings spread open symmetrically, clearly visible antennae."),
        Triple("ball", "Everyday objects", "One round red toy ball resting on a flat surface, centered, with a soft shadow."),
        Triple("bicycle", "Everyday objects", "One simple blue bicycle in side view, two wheels, pedals, a seat and handlebars clearly visible."),
        Triple("umbrella", "Everyday objects", "One open yellow umbrella with a curved handle, standing upright against a simple background."),
        Triple("toothbrush", "Everyday objects", "One blue toothbrush with a long handle and a small head of white bristles, centered."),
        Triple("chair", "Everyday objects", "One wooden chair with a seat, backrest and four legs, clear three-quarter view."),
        Triple("jump", "Actions", "One child jumping into the air, both feet above the ground, bent knees and raised arms, clear ground shadow."),
        Triple("run", "Actions", "One child running, full body visible, one leg reaching forward and arms swinging, simple park setting."),
        Triple("swim", "Actions", "One person swimming in a pool, arms reaching through blue water, head above the surface."),
        Triple("clap", "Actions", "One child clapping with both open hands meeting in front of the chest, happy face, hands clearly visible."),
        Triple("read", "Actions", "One child reading an open picture book, eyes looking down at the pages, simple quiet room, no visible letters."),
        Triple("happy", "Feelings", "One child feeling happy, with a big warm smile, bright eyes and relaxed open arms."),
        Triple("sad", "Feelings", "One child feeling sad, downturned mouth, lowered head and one tear, gentle everyday scene."),
        Triple("angry", "Feelings", "One child feeling angry, eyebrows lowered and arms folded, safe gentle scene without aggression."),
        Triple("surprised", "Feelings", "One child feeling surprised, wide eyes, raised eyebrows, a small open mouth and hands beside the cheeks."),
        Triple("sleepy", "Feelings", "One sleepy child yawning with drooping eyelids, one hand near the mouth, a pillow nearby."),
        Triple("red", "Colors", "One bright red toy cube on a plain white background, the red color clear and dominant."),
        Triple("blue", "Colors", "One bright blue toy cube on a plain white background, the blue color clear and dominant."),
        Triple("circle", "Shapes", "One large solid green circle, perfectly round with a smooth outline, flat front view on white."),
        Triple("triangle", "Shapes", "One large solid orange triangle with exactly three straight sides and three corners, flat front view on white."),
        Triple("square", "Shapes", "One large solid purple square with four equal straight sides and four right-angle corners, flat front view on white."),
    ).mapIndexed { index, (word, category, scene) -> ImageBenchmarkPrompt(word, category, "$scene $style", 42L + index) }
}

enum class ImageMethod(val label: String, val detail: String) {
    CPU("CPU", "LiteRT CPU · 4 threads"),
    GPU("GPU", "LiteRT FP32 GPU · Bonsai text encoder and VAE on CPU · experimental"),
    TPU("TPU", "LiteRT Tensor TPU + CPU host/unsupported partitions · experimental"),
}
data class ImageMethodSupport(val method: ImageMethod, val available: Boolean, val reason: String,
    val modelIds: Set<String>? = null, val compatibleModelIds: Set<String>? = null,
    val modelReasons: Map<String, String>? = null) {
    fun supports(modelId: String) = available && (modelIds == null || modelId in modelIds)
    fun compatible(modelId: String) = available && (compatibleModelIds == null || modelId in compatibleModelIds)
    fun reasonFor(modelId: String) = modelReasons?.get(modelId) ?: reason
}
data class ImageBenchmarkState(val runs: List<ImageBenchmarkRun> = emptyList(),
    val methods: List<ImageMethodSupport> = emptyList(), val activeRunId: String? = null,
    val step: Int = 0, val steps: Int = 0)
enum class ImageRunStatus { RUNNING, COMPLETED, CANCELED, INTERRUPTED }
enum class ImageSampleStatus { PENDING, RUNNING, SUCCESS, FAILED, SKIPPED }
data class ImageBenchmarkSample(
    val id: String, val modelId: String, val modelName: String, val method: ImageMethod,
    val input: ImageBenchmarkPrompt, val steps: Int, val status: ImageSampleStatus = ImageSampleStatus.PENDING,
    val result: ImageResult? = null, val elapsedMs: Long? = null, val error: String? = null,
    val quality: Int? = null, val bundleManifestSha256: String? = null,
)
data class ImageBenchmarkRun(
    val id: String, val startedAtEpochMs: Long, val updatedAtEpochMs: Long,
    val device: String, val suiteVersion: String, val samples: List<ImageBenchmarkSample>,
    val skippedMethods: List<ImageMethodSupport>, val timeoutMinutes: Int,
    val status: ImageRunStatus = ImageRunStatus.RUNNING, val finishedAtEpochMs: Long? = null,
    val protocol: String = "model resolution (FLUX LiteRT 256x256, Iris 768x768, others 512x512); fixed prompt/seed per word; model default steps; CPU=all CPU, GPU=Vulkan denoiser with CPU text/VAE for GGUF or LiteRT FP32 GPU (Bonsai text/VAE CPU), TPU=LiteRT AOT Tensor graphs with CPU host/unsupported partitions (compiler coverage recorded per result); seeds repeat within each runtime, not identical noise across runtimes; worker/engine retained per model/method; all selected words before unloading; worker termination awaited before next combination; LiteRT streams graphs; Bonsai retains its DiT during sampling; LiteRT model/environment recreated per image; TPU streams one compiled shard at a time; end-to-end latency includes startup/load on first image and encode/denoise/decode/save on every image; no warm-up excluded; 1s between samples; RSS high-water is per combination",
) {
    val retainsEngine get() = protocol.contains("worker/engine retained per model/method")
    val successful get() = samples.count { it.status == ImageSampleStatus.SUCCESS }
    val failed get() = samples.count { it.status == ImageSampleStatus.FAILED }
    val pending get() = samples.count { it.status == ImageSampleStatus.PENDING || it.status == ImageSampleStatus.RUNNING }
    val resumable get() = status != ImageRunStatus.RUNNING && pending > 0
}
data class ImageSpeedStats(val count: Int, val meanMs: Double, val medianMs: Double, val minMs: Long, val maxMs: Long) {
    companion object {
        fun from(values: List<Long>): ImageSpeedStats? {
            if (values.isEmpty()) return null
            require(values.all { it >= 0 })
            val sorted = values.sorted()
            val middle = sorted.size / 2
            val median = if (sorted.size % 2 == 0) sorted[middle - 1] / 2.0 + sorted[middle] / 2.0 else sorted[middle].toDouble()
            return ImageSpeedStats(sorted.size, sorted.sumOf { it.toDouble() } / sorted.size, median, sorted.first(), sorted.last())
        }
    }
}
data class ImageBenchmarkGroup(val modelId: String, val modelName: String, val method: ImageMethod,
    val samples: List<ImageBenchmarkSample>) {
    val speed get() = ImageSpeedStats.from(samples.filter { it.status == ImageSampleStatus.SUCCESS }.mapNotNull { it.elapsedMs })
    val ratings get() = samples.filter { it.status == ImageSampleStatus.SUCCESS }.mapNotNull { it.quality }
    val meanQuality get() = ratings.takeIf { it.isNotEmpty() }?.average()
}
fun ImageBenchmarkRun.groups(): List<ImageBenchmarkGroup> = samples.groupBy { it.modelId to it.method }
    .map { (key, values) -> ImageBenchmarkGroup(key.first, values.first().modelName, key.second, values) }
fun newImageBenchmarkRun(id: String, now: Long, device: String, models: List<ImageModel>,
    methods: List<ImageMethodSupport>, timeoutMinutes: Int, includeUnavailable: Boolean = false,
    wordCount: Int = ImageBenchmarkSuite.MAX_WORD_COUNT): ImageBenchmarkRun {
    require(models.isNotEmpty() && models.distinctBy { it.id }.size == models.size)
    require(methods.any { it.available } && methods.distinctBy { it.method }.size == methods.size)
    require(timeoutMinutes in 1..60)
    require(wordCount in ImageBenchmarkSuite.MIN_WORD_COUNT..ImageBenchmarkSuite.MAX_WORD_COUNT)
    val combinations = models.flatMap { model -> methods.filter { it.supports(model.id) }.map { model to it.method } }
    require(combinations.isNotEmpty()) { "No selected model/method combination is installed." }
    val pairs = if (includeUnavailable) models.flatMap { model -> methods.map { model to it.method } } else combinations
    val samples = pairs.flatMap { (model, method) ->
        ImageBenchmarkSuite.prompts.take(wordCount).mapIndexed { index, input ->
            ImageBenchmarkSample("${model.id}-${method.name}-$index", model.id,
                model.name, method, input, model.steps,
                status = if (methods.first { it.method == method }.supports(model.id)) ImageSampleStatus.PENDING else ImageSampleStatus.SKIPPED,
                error = methods.first { it.method == method }.takeUnless { it.supports(model.id) }?.reasonFor(model.id))
        }
    }
    return ImageBenchmarkRun(id, now, now, device, ImageBenchmarkSuite.VERSION, samples, methods.filterNot { it.available }, timeoutMinutes)
}
