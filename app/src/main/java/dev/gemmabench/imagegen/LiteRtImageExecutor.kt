package dev.gemmabench.imagegen

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.gson.Gson
import com.google.gson.JsonObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random

/** Executes exporter-owned tensor plans, with one shard resident at a time. */
class LiteRtImageExecutor(private val context: Context, private val method: ImageMethod = ImageMethod.TPU) : AutoCloseable {
    private var modelId: String? = null
    private var bundle: Pair<java.io.File, JsonObject>? = null
    private var environment: Environment? = null
    override fun close() {
        environment?.close()
        environment = null
        bundle = null
        modelId = null
    }
    data class Output(val pixels: IntArray, val hashes: Map<String, String>, val backend: String,
        val encodeMs: Long, val denoiseMs: Long, val decodeMs: Long)

    fun generate(model: ImageModel, prompt: String, seed: Long, steps: Int, progress: (Int, Int) -> Unit): Output {
        require(modelId == null || modelId == model.id) { "Close the TPU session before changing models." }
        val portable = method != ImageMethod.TPU
        val (dir, manifest) = bundle ?: LiteRtImageBundles(context, portable).load(model.id).also {
            bundle = it
            modelId = model.id
        }
        require(steps == manifest["steps"].asInt) { "The compiled bundle fixes ${manifest["steps"]} steps. Re-export to change it." }
        val tensors = mutableMapOf<String, Any>()
        val tokenizers = mutableMapOf<String, ImageBpeTokenizer.Encoded>()
        val random = Random(seed)
        manifest.getAsJsonArray("externals").forEach { entry ->
            val e = entry.asJsonObject
            val count = e.getAsJsonArray("shape").fold(1L) { n, v -> Math.multiplyExact(n, v.asLong) }
            require(count in 1..Int.MAX_VALUE)
            val value: Any = when (e["kind"].asString) {
                "tokens", "mask" -> {
                    val encoded = tokenizers.getOrPut(e["tokenizer"].asString) {
                        val tokenizer = ImageBpeTokenizer(Gson().fromJson(LiteRtImageBundles.safeFile(dir, e["tokenizer"].asString).readText(), JsonObject::class.java))
                        tokenizer.validateFixtures(); tokenizer.encode(prompt)
                    }
                    if (e["kind"].asString == "tokens") encoded.ids else encoded.mask
                }
                "normal" -> FloatArray(count.toInt()) { random.nextGaussian().toFloat() }
                "constant" -> {
                    val bytes = LiteRtImageBundles.safeFile(dir, e["path"].asString).readBytes()
                    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    when (e["dtype"].asString) {
                        "float32" -> FloatArray(count.toInt()).also { buffer.asFloatBuffer().get(it) }
                        "int64" -> LongArray(count.toInt()).also { buffer.asLongBuffer().get(it) }
                        "int32" -> IntArray(count.toInt()).also { buffer.asIntBuffer().get(it) }
                        "bool" -> BooleanArray(count.toInt()) { buffer.get().toInt() != 0 }
                        else -> error("Unsupported constant tensor type")
                    }
                }
                else -> error("Unsupported image bundle input")
            }
            tensors[e["tensor"].asString] = value
        }
        val actions = manifest.getAsJsonArray("actions").map { it.asJsonObject }
        val uses = actions.flatMap { it.getAsJsonArray("inputs").map { i -> i.asJsonObject["tensor"].asString } }.groupingBy { it }.eachCount().toMutableMap()
        val final = manifest["output"].asString
        uses[final] = (uses[final] ?: 0) + 1
        val times = mutableMapOf("encode" to 0L, "denoise" to 0L, "decode" to 0L)
        var reportedStep = 0
        val activeEnvironment = environment ?: Environment.create(context,
            mapOf(Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir)).also { environment = it }
        run {
            val environment = activeEnvironment
            if (!portable) check(Accelerator.NPU in environment.getAvailableAccelerators()) { "LiteRT Google Tensor NPU dispatch is unavailable on this device/build." }
            for (action in actions) {
                val phase = action["phase"].asString
                val step = action["step"].asInt
                if (phase == "denoise" && step != reportedStep) { reportedStep = step; progress((step - 1).coerceAtLeast(0), steps) }
                val start = SystemClock.elapsedRealtime()
                val graph = action["model"].asString
                val offloaded = if (portable) 0 else manifest.getAsJsonObject("offload").getAsJsonObject(graph)["offloadedOps"].asLong
                // Conditioning uses int64 token/gather operations; sampler and VAE stay on CPU.
                // GPU is an explicit denoiser pipeline, never a retry after a GPU failure.
                val gpuDenoiser = portable && method == ImageMethod.GPU && phase == "denoise" &&
                    (graph.startsWith("graphs/unet-") || graph.startsWith("graphs/dit-"))
                val accelerator = if (gpuDenoiser) Accelerator.GPU else if (offloaded > 0) Accelerator.NPU else Accelerator.CPU
                val options = CompiledModel.Options(accelerator).apply {
                    if (accelerator == Accelerator.GPU) gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
                    if (accelerator == Accelerator.CPU) cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
                }
                // Never retry a failed NPU graph with CPU options. Partial CPU partitions are
                // declared in compiler coverage and explicitly included in the backend label.
                Log.i("LiteRtImage", "${model.id}/${method.name}: $graph · $accelerator")
                CompiledModel.create(LiteRtImageBundles.safeFile(dir, graph).absolutePath,
                    options, environment).use { compiled ->
                    val inputs = mutableMapOf<String, TensorBuffer>()
                    val outputs = mutableMapOf<String, TensorBuffer>()
                    val signature = action["signature"].asString
                    try {
                        action.getAsJsonArray("inputs").forEach { input ->
                            val i = input.asJsonObject
                            val name = i["name"].asString
                            val buffer = compiled.createInputBuffer(name, signature)
                            inputs[name] = buffer
                            when (val data = requireNotNull(tensors[i["tensor"].asString])) {
                                is FloatArray -> buffer.writeFloat(data)
                                is LongArray -> buffer.writeLong(data)
                                is IntArray -> buffer.writeInt(data)
                                is BooleanArray -> buffer.writeBoolean(data)
                                else -> error("Unsupported tensor input type")
                            }
                        }
                        action.getAsJsonArray("outputs").forEach { output ->
                            val o = output.asJsonObject
                            // A passthrough output can be the *same* TFLite tensor as an input.
                            // Allocating it separately replaces the input allocation with garbage.
                            outputs[o["name"].asString] = o["aliasOfInput"]?.let { inputs.getValue(it.asString) }
                                ?: compiled.createOutputBuffer(o["name"].asString, signature)
                        }
                        compiled.run(inputs, outputs, signature)
                        action.getAsJsonArray("outputs").forEach { output ->
                            val o = output.asJsonObject
                            val buffer = requireNotNull(outputs[o["name"].asString])
                            tensors[o["tensor"].asString] = when (o["dtype"].asString) {
                                "float32" -> buffer.readFloat().also { values -> check(values.all { it.isFinite() }) { "Non-finite LiteRT output; compiler precision may be incompatible." } }
                                "int64" -> buffer.readLong()
                                "int32" -> buffer.readInt()
                                "bool" -> buffer.readBoolean()
                                else -> error("Unsupported tensor output type")
                            }
                        }
                    } finally { (inputs.values + outputs.values).distinct().forEach { it.close() } }
                }
                times[phase] = requireNotNull(times[phase]) + SystemClock.elapsedRealtime() - start
                action.getAsJsonArray("inputs").forEach { input ->
                    val ref = input.asJsonObject["tensor"].asString
                    val left = requireNotNull(uses[ref]) - 1
                    uses[ref] = left
                    if (left == 0) tensors.remove(ref)
                }
            }
        }
        progress(steps, steps)
        val rgb = tensors[final] as? FloatArray ?: error("Expected float32 RGB output")
        val size = model.width * model.height
        check(rgb.size == 3 * size)
        val pixels = IntArray(size) { i ->
            fun channel(offset: Int) = (rgb[offset + i].coerceIn(0f, 1f) * 255 + 0.5f).toInt()
            (0xff shl 24) or (channel(0) shl 16) or (channel(size) shl 8) or channel(size * 2)
        }
        val offload = if (portable) emptyList() else manifest.getAsJsonObject("offload").entrySet().map { it.value.asJsonObject }
        val offloaded = offload.sumOf { it["offloadedOps"].asLong }
        val total = offload.sumOf { it["totalOps"].asLong }
        return Output(pixels, manifest.getAsJsonObject("files").entrySet().associate { it.key to it.value.asJsonObject["sha256"].asString },
            if (portable && method == ImageMethod.GPU) "LiteRT FP32 GPU denoiser + CPU text/VAE/sampler/host · experimental"
            else if (portable) "LiteRT CPU · CPU host · experimental" else "LiteRT ${manifest["targetSoc"].asString} TPU + CPU host/partitions · $offloaded/$total unique graph ops AOT offloaded · experimental",
            times.getValue("encode"), times.getValue("denoise"), times.getValue("decode"))
    }
}
