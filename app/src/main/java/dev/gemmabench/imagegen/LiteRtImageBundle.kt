package dev.gemmabench.imagegen

import android.content.Context
import android.os.Build
import com.google.gson.Gson
import com.google.gson.JsonObject
import java.io.File
import java.security.MessageDigest

/** Compiled diffusion graphs have a separate store; GGUF files are never LiteRT inputs. */
class LiteRtImageBundles(private val context: Context, private val portable: Boolean = false) {
    private val gson = Gson()
    private val root get() = File(context.getExternalFilesDir(null), "image-litert")
    val soc get() = tensorSoc(Build.SOC_MODEL)
    private fun directory(modelId: String) = File(root, "${modelId.removeSuffix("_LITERT")}/${if (portable) "portable" else requireNotNull(soc) { "Tensor G5/G6 required." }}")

    fun installed(): Set<String> = if (!portable && soc == null) emptySet() else ImageModels.all.filter {
        if (portable) it.pipeline == ImagePipeline.LITERT_PLAN else it.pipeline == ImagePipeline.NATIVE
    }.mapNotNull { model ->
        runCatching { load(model.id); model.id }.getOrNull()
    }.toSet()

    /** Explicit verification is performed on the Lab worker, never during Compose rendering. */
    fun verify(): String {
        require(portable || soc != null) { "Requires a Google Tensor G5 or G6 device." }
        val messages = ImageModels.all.filter {
            if (portable) it.pipeline == ImagePipeline.LITERT_PLAN else it.pipeline == ImagePipeline.NATIVE
        }.mapNotNull { model ->
            val dir = directory(model.id)
            if (!File(dir, "manifest.json").isFile) return@mapNotNull null
            try {
                val manifest = parse(dir, model.id)
                val stamps = JsonObject()
                manifest.getAsJsonObject("files").entrySet().forEach { (relative, spec) ->
                    val file = safeFile(dir, relative)
                    check(file.isFile && file.length() == spec.asJsonObject["bytes"].asLong) { "Missing or wrong-sized $relative" }
                    check(sha256(file) == spec.asJsonObject["sha256"].asString) { "SHA-256 mismatch: $relative" }
                    stamps.addProperty(relative, "${file.length()}:${file.lastModified()}")
                }
                val tokenizers = manifest.getAsJsonArray("externals").mapNotNull { it.asJsonObject["tokenizer"]?.asString }.distinct()
                tokenizers.forEach { ImageBpeTokenizer(gson.fromJson(safeFile(dir, it).readText(), JsonObject::class.java)).validateFixtures() }
                val marker = JsonObject().apply {
                    addProperty("manifestSha256", sha256(File(dir, "manifest.json"))); add("stamps", stamps)
                }
                File(dir, ".verified.tmp").writeText(gson.toJson(marker))
                check(File(dir, ".verified.tmp").renameTo(File(dir, ".verified")))
                "${model.name}: verified · ${if (portable) "portable LiteRT" else soc} bundle"
            } catch (e: Exception) {
                File(dir, ".verified").delete()
                "${model.name}: ${e.message}"
            }
        }
        return messages.ifEmpty { listOf(if (portable) "No portable SD/SDXL bundles staged. See docs/IMAGE_LITERT.md." else "No compiled TPU bundles staged. See docs/IMAGE_TPU.md.") }.joinToString("\n")
    }

    fun load(modelId: String): Pair<File, JsonObject> {
        val dir = directory(modelId)
        val manifest = parse(dir, modelId)
        val marker = gson.fromJson(File(dir, ".verified").readText(), JsonObject::class.java)
        check(marker["manifestSha256"].asString == sha256(File(dir, "manifest.json"))) { "Bundle changed; verify it again." }
        manifest.getAsJsonObject("files").entrySet().forEach { (relative, spec) ->
            val file = safeFile(dir, relative)
            check(file.isFile && file.length() == spec.asJsonObject["bytes"].asLong &&
                marker.getAsJsonObject("stamps")[relative]?.asString == "${file.length()}:${file.lastModified()}") { "Bundle changed; verify it again." }
        }
        return dir to manifest
    }

    private fun parse(dir: File, modelId: String): JsonObject {
        val file = File(dir, "manifest.json")
        require(file.length() in 1..16_000_000) { "Missing or invalid manifest.json" }
        val m = gson.fromJson(file.readText(), JsonObject::class.java)
        require(m["format"].asString == "word-image-litert-v1" && m["modelId"].asString == modelId.removeSuffix("_LITERT"))
        if (portable) require(!m["compiled"].asBoolean) { "Portable graphs must not contain target-specific AOT compilation." }
        else require(m["compiled"].asBoolean && m["targetSoc"].asString == soc) { "AOT bundle must match this phone's $soc." }
        require(m["width"].asInt == 512 && m["height"].asInt == 512 && m["steps"].asInt == ImageModels.get(modelId).steps)
        val files = m.getAsJsonObject("files")
        files.keySet().forEach { safeFile(dir, it) }
        val tensors = m.getAsJsonArray("externals").map { it.asJsonObject["tensor"].asString }.toMutableSet()
        var neuralOffloaded = 0L
        m.getAsJsonArray("externals").forEach { entry ->
            val e = entry.asJsonObject
            require(e["kind"].asString in setOf("tokens", "mask", "normal", "constant"))
            for (key in listOf("path", "tokenizer")) e[key]?.let { require(files.has(it.asString)) }
        }
        m.getAsJsonArray("actions").forEach { entry ->
            val action = entry.asJsonObject
            val graph = action["model"].asString
            require(files.has(graph) && graph.endsWith(".tflite"))
            require(action["phase"].asString in setOf("encode", "denoise", "decode"))
            if (!portable) {
                val coverage = m.getAsJsonObject("offload").getAsJsonObject(graph)
                require(coverage["totalOps"].asLong > 0 && coverage["offloadedOps"].asLong in 0..coverage["totalOps"].asLong)
                if (action["phase"].asString == "denoise" && (graph.contains("unet-") || graph.contains("dit-"))) neuralOffloaded += coverage["offloadedOps"].asLong
            }
            action.getAsJsonArray("inputs").forEach { require(it.asJsonObject["tensor"].asString in tensors) { "Invalid tensor dependency" } }
            action.getAsJsonArray("outputs").forEach { require(tensors.add(it.asJsonObject["tensor"].asString)) { "Duplicate tensor output" } }
        }
        require(portable || neuralOffloaded > 0) { "No denoiser operations were compiled for TPU; cannot report this as a TPU method." }
        require(m["output"].asString in tensors)
        return m
    }

    companion object {
        fun tensorSoc(value: String): String? = when (value.lowercase().replace(Regex("[^a-z0-9]"), "")) {
            "tensorg5" -> "Tensor_G5"
            "tensorg6" -> "Tensor_G6"
            else -> null
        }
        fun safeFile(dir: File, relative: String): File {
            require(relative.isNotBlank() && !File(relative).isAbsolute)
            val target = File(dir, relative).canonicalFile
            require(target.path.startsWith(dir.canonicalPath + File.separator)) { "Bundle path escapes its directory" }
            return target
        }
        fun sha256(file: File): String {
            val hash = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { stream ->
                val block = ByteArray(1024 * 1024)
                while (true) { val read = stream.read(block); if (read < 0) break; hash.update(block, 0, read) }
            }
            return hash.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
