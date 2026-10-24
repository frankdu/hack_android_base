package dev.gemmabench.imagegen

import android.content.Context
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.gson.JsonParser
import dev.gemmabench.imagegen.bonsai.BonsaiMath
import dev.gemmabench.imagegen.bonsai.QwenTokenizer
import dev.gemmabench.imagegen.iris.*
import java.io.File
import java.util.Random

/** Shared environment, sequential graph residency, explicit accelerator selection. */
class PortableImagePipeline(private val context: Context) : AutoCloseable {
    private val environment = Environment.create(context)
    override fun close() = environment.close()
    data class Output(val pixels: IntArray, val backend: String, val encodeMs: Long,
        val denoiseMs: Long, val decodeMs: Long)

    fun generate(model: ImageModel, dir: File, prompt: String, seed: Long, steps: Int,
        method: ImageMethod, progress: (Int, Int) -> Unit): Output {
        require(method != ImageMethod.TPU)
        require(steps == model.steps) { "This LiteRT pipeline uses ${model.steps} steps." }
        return when (model.pipeline) {
            ImagePipeline.BONSAI_LITERT -> bonsai(dir, prompt, seed, steps, method, progress)
            ImagePipeline.IRIS_LITERT -> iris(dir, prompt, seed, method, progress)
            ImagePipeline.FLUX_LITERT -> flux(dir, prompt, seed, steps, method, progress)
            else -> error("Unsupported portable image pipeline")
        }
    }

    /** Inputs and outputs follow numbered converter arguments, never shapes or map iteration. */
    private inner class Graph(file: File, gpu: Boolean, inputCount: Int, outputCount: Int = 1) : AutoCloseable {
        private val inputs = linkedMapOf<String, TensorBuffer>()
        private val outputs = linkedMapOf<String, TensorBuffer>()
        private val options = CompiledModel.Options(if (gpu) Accelerator.GPU else Accelerator.CPU).apply {
            if (gpu) gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
            else cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }
        private val compiled = CompiledModel.create(file.absolutePath, options, environment)
        init {
            try {
                repeat(inputCount) { inputs["args_$it"] = compiled.createInputBuffer("args_$it", "serving_default") }
                repeat(outputCount) { outputs["output_$it"] = compiled.createOutputBuffer("output_$it", "serving_default") }
            } catch (failure: Throwable) { close(); throw failure }
        }
        fun run(vararg values: Any): List<FloatArray> {
            require(values.size == inputs.size)
            inputs.entries.zip(values.toList()).forEach { (entry, value) ->
                val buffer = entry.value
                val count = requireNotNull(compiled.getInputTensorType(entry.key, "serving_default").layout).dimensions.fold(1L) { n, d -> n * d }
                when (value) {
                    is FloatArray -> { require(count == value.size.toLong()); buffer.writeFloat(value) }
                    is IntArray -> { require(count == value.size.toLong()); buffer.writeInt(value) }
                    else -> error("Unsupported input array")
                }
            }
            compiled.run(inputs, outputs, "serving_default")
            return outputs.values.map { it.readFloat().also { a -> check(a.all(Float::isFinite)) { "Non-finite LiteRT output." } } }
        }
        override fun close() {
            inputs.values.forEach { it.close() }; outputs.values.forEach { it.close() }; compiled.close()
        }
    }

    private fun bonsai(dir: File, prompt: String, seed: Long, steps: Int, method: ImageMethod,
        progress: (Int, Int) -> Unit): Output {
        val start = SystemClock.elapsedRealtime()
        val meta = JsonParser.parseString(File(dir, "pipeline_meta.json").readText()).asJsonObject
        val encoded = File(dir, "tokenizer/vocab.json").inputStream().use { vocab ->
            File(dir, "tokenizer/merges.txt").inputStream().use { merges -> QwenTokenizer(vocab, merges).encodePrompt(prompt) }
        }
        val embeds = Graph(File(dir, "textenc_int4.tflite"), false, 2).use { it.run(encoded.ids, encoded.mask).single() }
        val encodeEnd = SystemClock.elapsedRealtime()
        val latent = BonsaiMath.noise(seed)
        val sigmas = BonsaiMath.sigmas(steps)
        val imageIds = BonsaiMath.imgIds(); val textIds = BonsaiMath.txtIds()
        val gpu = method == ImageMethod.GPU
        Graph(File(dir, if (gpu) "dit_gpu_int4b32.tflite" else "dit_int4b32.tflite"), gpu, 5).use { graph ->
            repeat(steps) { step ->
                progress(step, steps)
                val v = graph.run(latent, embeds, floatArrayOf(sigmas[step]), imageIds, textIds).single()
                require(v.size == latent.size)
                val ds = sigmas[step + 1] - sigmas[step]
                for (i in latent.indices) latent[i] += ds * v[i]
            }
        }
        val denoiseEnd = SystemClock.elapsedRealtime(); progress(steps, steps)
        val scale = meta.getAsJsonArray("latent_bn_scale").map { it.asFloat }.toFloatArray()
        val shift = meta.getAsJsonArray("latent_bn_shift").map { it.asFloat }.toFloatArray()
        val rgb = Graph(File(dir, "vae_dec_fp32.tflite"), false, 1).use { it.run(BonsaiMath.unpatchify(latent, scale, shift)).single() }
        return Output(pixels(rgb, 512), if (gpu) "LiteRT FP32 GPU DiT + CPU text/VAE/host" else "LiteRT CPU (4 threads) + CPU host",
            encodeEnd - start, denoiseEnd - encodeEnd, SystemClock.elapsedRealtime() - denoiseEnd)
    }

    private fun iris(dir: File, prompt: String, seed: Long, method: ImageMethod,
        progress: (Int, Int) -> Unit): Output {
        val start = SystemClock.elapsedRealtime()
        val rows = File(dir, "app_rows.json")
        rows.writeText("""{"kind":"generic","config":"iris3b_config.json","inputs":[],"rows":[]}""")
        val files = RunFiles(dir, rows.name)
        files.materializeTokenizer()
        val original = files.engineConfig()
        val base = LiteRtExec(environment, dir)
        val exec = object : GraphExec {
            override fun load(spec: GraphSpec): LoadedGraph {
                LiteRtImageBundles.safeFile(dir, spec.file)
                return base.load(GraphSpec(spec.role, spec.file, "fp32", spec.signature, spec.inputs,
                    spec.outputs, if (method == ImageMethod.GPU) "gpu" else "cpu", 4, false, spec.bytes))
            }
            override fun run(graph: LoadedGraph, inputs: List<FloatArray>): GraphRun = base.run(graph, inputs).also {
                check(it.outputs.all { a -> a.all(Float::isFinite) }) { "Non-finite Iris output." }
            }
            override fun close(graph: LoadedGraph) = base.close(graph)
        }
        var denoiseStart = 0L; var denoiseEnd = 0L
        val log = StageLog(System::nanoTime, { emptyMap() }) { event ->
            if (event.stage == "load" && event.graph == original.dual.first().role && denoiseStart == 0L)
                denoiseStart = SystemClock.elapsedRealtime() - (event.ms["ms"] ?: 0.0).toLong()
            if (event.stage == "sampler") { denoiseEnd = SystemClock.elapsedRealtime(); progress(event.step + 1, 4) }
        }
        val random = Random(seed)
        val noise = FloatArray(HostOps.IMAGE_FLOATS) { random.nextGaussian().toFloat() }
        val result = PromptTokens(File(dir, "tok"), File(dir, "iris_embed_bf16.f32")).use { tokens ->
            IrisPipeline(exec, original, files.host, tokens).generate(prompt, noise, log)
        }
        return Output(pixels(result.image, 768), "LiteRT ${if (method == ImageMethod.GPU) "FP32 GPU" else "CPU (4 threads)"} + CPU host",
            denoiseStart - start, denoiseEnd - denoiseStart, SystemClock.elapsedRealtime() - denoiseEnd)
    }

    private fun flux(dir: File, prompt: String, seed: Long, steps: Int, method: ImageMethod,
        progress: (Int, Int) -> Unit): Output = CommunityFlux.generate(context.assets.open("image-litert/flux_host.json").bufferedReader().use { it.readText() }, dir, prompt, seed, steps, method, progress) { name, input, outputs ->
            Graph(File(dir, name), method == ImageMethod.GPU, input.size, outputs).use { it.run(*input.toTypedArray()) }
        }

    companion object {
        fun pixels(rgb: FloatArray, size: Int): IntArray {
            val plane = size * size
            require(rgb.size == plane * 3 && rgb.all(Float::isFinite)) { "Invalid LiteRT RGB image." }
            return IntArray(plane) { p ->
                fun channel(c: Int) = ((rgb[c * plane + p] / 2f + 0.5f).coerceIn(0f, 1f) * 255f + 0.5f).toInt()
                (0xff shl 24) or (channel(0) shl 16) or (channel(1) shl 8) or channel(2)
            }
        }
    }
}
