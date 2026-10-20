package dev.gemmabench.photos

import com.google.gson.JsonParser
import com.google.gson.JsonParseException
import com.google.gson.Strictness
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.StringReader
import dev.gemmabench.benchmark.ModelSlot

data class DevicePhoto(val id: Long, val uri: String, val name: String, val mimeType: String? = null)

/** Coordinates use the upright image, origin top-left, on a 0..1000 scale. */
data class ObjectBox(val name: String, val left: Float, val top: Float, val right: Float, val bottom: Float)

enum class PhotoStatus { PENDING, ANALYZING, COMPLETED, FAILED }

data class PhotoResult(
    val photo: DevicePhoto,
    val status: PhotoStatus = PhotoStatus.PENDING,
    val previewPath: String? = null,
    val width: Int = 0,
    val height: Int = 0,
    val objects: List<ObjectBox> = emptyList(),
    val rawOutput: String = "",
    val elapsedMs: Double? = null,
    val error: String? = null,
    val attempts: Int = 0,
    val decodeTokens: Int? = null,
    val outputTokenLimit: Int? = null,
    val errorKind: String? = null,
)

data class PhotoDemoState(
    val results: List<PhotoResult> = emptyList(),
    val beforeId: Long? = null,
    val hasMore: Boolean = true,
    val batchNumber: Int = 0,
    val modelLabel: String = "",
    val modelSlot: ModelSlot? = null,
    val accessScope: PhotoAccess? = null,
    val modelSha256: String = "",
    val textBackend: String = "",
    val visionBackend: String? = null,
    val batchSize: Int = DEFAULT_BATCH_SIZE,
    val unsupportedSkipped: Int = 0,
) {
    companion object {
        const val DEFAULT_BATCH_SIZE = 10
        const val MAX_BATCH_SIZE = 50
    }
}

object PhotoObjects {
    const val SCHEMA = """{"type":"object","properties":{"objects":{"type":"array","maxItems":8,"items":{"type":"object","properties":{"name":{"type":"string","minLength":1,"maxLength":120},"box":{"type":"array","items":{"type":"integer","minimum":0,"maximum":1000},"minItems":4,"maxItems":4}},"required":["name","box"],"additionalProperties":false}}},"required":["objects"],"additionalProperties":false}"""
    const val PROMPT = """Identify up to 8 major visible objects in this image. Return JSON only:
{"objects":[{"name":"object name","box":[left,top,right,bottom]}]}
Each box tightly encloses that object. Coordinates are integers from 0 to 1000, normalized
to the full image width and height, origin at top-left. left < right and top < bottom.
Use short common object names. List separate instances separately. Do not invent objects.
If there are no identifiable objects, return {"objects":[]}. No explanation or markdown."""
    val RETRY_PROMPT = PROMPT.replace("up to 8", "up to 4") + "\nKeep the response compact and finish the entire JSON object."

    fun parse(text: String): List<ObjectBox> {
        val json = text.trim().let {
            if (it.startsWith("```")) it.substringAfter('\n').substringBeforeLast("```").trim() else it
        }
        if (json.isBlank()) throw PhotoResponseException("Gemma returned an empty response. See raw response.")
        val root = try {
            JsonReader(StringReader(json)).use { reader ->
                reader.strictness = Strictness.STRICT
                val value = JsonParser.parseReader(reader)
                require(reader.peek() == JsonToken.END_DOCUMENT) { "Unexpected content after objects JSON." }
                value
            }
        } catch (e: JsonParseException) {
            throw PhotoResponseException("Gemma returned incomplete or malformed JSON. See raw response.", e)
        } catch (e: java.io.IOException) {
            throw PhotoResponseException("Gemma returned incomplete or malformed JSON. See raw response.", e)
        }
        require(root.isJsonObject && root.asJsonObject.has("objects") && root.asJsonObject["objects"].isJsonArray) {
            "Gemma did not return the expected objects JSON. See raw response."
        }
        val objects = root.asJsonObject.getAsJsonArray("objects")
        require(objects.size() <= 32) { "Gemma returned too many objects. See raw response." }
        return objects.map { item ->
            require(item.isJsonObject) { "Invalid object entry." }
            val entry = item.asJsonObject
            val nameValue = entry["name"]
            require(nameValue?.isJsonPrimitive == true && nameValue.asJsonPrimitive.isString) { "Missing object name." }
            val name = nameValue.asString.trim()
            require(name.isNotEmpty() && name.length <= 120) { "Invalid object name." }
            require(entry["box"]?.isJsonArray == true) { "Missing bounding box for $name." }
            val box = entry.getAsJsonArray("box")
            require(box.size() == 4) { "Expected four coordinates for $name." }
            val values = box.map {
                require(it.isJsonPrimitive && it.asJsonPrimitive.isNumber) { "Non-numeric box for $name." }
                it.asFloat.also { v -> require(v.isFinite() && v in 0f..1000f) { "Box outside image for $name." } }
            }
            require(values[0] < values[2] && values[1] < values[3]) { "Empty or reversed box for $name." }
            ObjectBox(name, values[0], values[1], values[2], values[3])
        }
    }
}
