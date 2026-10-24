package dev.gemmabench.imagegen

import com.google.gson.JsonObject
import java.text.Normalizer
import java.util.Locale
import java.util.regex.Pattern

/** Byte BPE for the exported CLIP/Qwen tokenizer ABI; fixtures guard tokenizer compatibility. */
class ImageBpeTokenizer(private val spec: JsonObject) {
    private val clip = spec["kind"].asString == "clip"
    private val vocab = spec.getAsJsonObject("vocab").entrySet().associate { it.key to it.value.asInt }
    private val ranks = spec.getAsJsonArray("merges").mapIndexed { index, value -> value.asString to index }.toMap()
    private val length = spec["length"].asInt
    private val specials = spec.getAsJsonArray("specialTokens").associate { it.asJsonObject["text"].asString to it.asJsonObject["id"].asInt }
    private val specialPattern = Pattern.compile(specials.keys.sortedByDescending { it.length }.joinToString("|") { Pattern.quote(it) }.ifEmpty { "(?!)" })
    private val pattern = Pattern.compile(spec["pattern"]?.asString ?: if (clip)
        "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        else "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    private val bytes = run {
        val direct = (33..126).toList() + (161..172).toList() + (174..255).toList()
        var extra = 0
        (0..255).associateWith { if (it in direct) it.toChar().toString() else (256 + extra++).toChar().toString() }
    }
    private val cache = mutableMapOf<String, List<Int>>()
    data class Encoded(val ids: LongArray, val mask: LongArray)

    fun encode(prompt: String): Encoded {
        var text = spec["prefix"].asString + prompt + spec["suffix"].asString
        text = Normalizer.normalize(text, Normalizer.Form.NFC)
        if (clip) {
            // CLIP performs double HTML unescape before whitespace normalization.
            repeat(2) { text = htmlUnescape(text) }
            text = text.replace(Regex("\\s+"), " ").trim().lowercase(Locale.ROOT)
        }
        val ids = mutableListOf<Int>()
        spec["bos"].takeUnless { it.isJsonNull }?.let { ids += it.asInt }
        var start = 0
        val specialMatcher = specialPattern.matcher(text)
        while (specialMatcher.find()) {
            tokenize(text.substring(start, specialMatcher.start()), ids)
            ids += requireNotNull(specials[specialMatcher.group()])
            start = specialMatcher.end()
        }
        tokenize(text.substring(start), ids)
        val eos = spec["eos"].takeUnless { it.isJsonNull }?.asInt
        if (eos != null) ids += eos
        if (ids.size > length) {
            ids.subList(length, ids.size).clear()
            if (eos != null) ids[length - 1] = eos
        }
        val count = ids.size
        val pad = spec["pad"].asInt
        val left = spec["paddingSide"].asString == "left"
        return Encoded(LongArray(length) { i ->
            val source = if (left) i - (length - count) else i
            if (source in ids.indices) ids[source].toLong() else pad.toLong()
        }, LongArray(length) { i -> if (if (left) i >= length - count else i < count) 1L else 0L })
    }

    fun validateFixtures() {
        spec.getAsJsonArray("fixtures").forEach { fixture ->
            val f = fixture.asJsonObject
            val result = encode(f["prompt"].asString)
            check(result.ids.contentEquals(f.getAsJsonArray("ids").map { it.asLong }.toLongArray()) &&
                result.mask.contentEquals(f.getAsJsonArray("mask").map { it.asLong }.toLongArray())) {
                "Tokenizer parity failed (${spec["kind"].asString}: ${f["prompt"].asString.take(40)}); this bundle is incompatible. Re-export before generation."
            }
        }
    }

    private fun tokenize(text: String, output: MutableList<Int>) {
        val matcher = pattern.matcher(text)
        while (matcher.find()) {
            val word = matcher.group()
            output += cache.getOrPut(word) {
                val encoded = word.toByteArray(Charsets.UTF_8).joinToString("") { requireNotNull(bytes[it.toInt() and 255]) }
                val parts = encoded.map { it.toString() }.toMutableList()
                if (clip) parts[parts.lastIndex] = parts.last() + "</w>"
                while (parts.size > 1) {
                    var best = -1
                    var rank = Int.MAX_VALUE
                    for (i in 0 until parts.lastIndex) {
                        val candidate = ranks["${parts[i]} ${parts[i + 1]}"] ?: continue
                        if (candidate < rank) { rank = candidate; best = i }
                    }
                    if (best < 0) break
                    val first = parts[best]; val second = parts[best + 1]
                    var index = 0
                    while (index < parts.lastIndex) {
                        if (parts[index] == first && parts[index + 1] == second) {
                            parts[index] = first + second; parts.removeAt(index + 1)
                        }
                        index++
                    }
                }
                parts.map { requireNotNull(vocab[it]) { "Unknown BPE token in bundle." } }
            }
        }
    }

    private fun htmlUnescape(value: String) = Regex("&(#x[0-9a-fA-F]+|#[0-9]+|amp|lt|gt|quot|apos|nbsp);").replace(value) { match ->
        val entity = match.groupValues[1]
        val point = when {
            entity.startsWith("#x") -> entity.substring(2).toIntOrNull(16)
            entity.startsWith("#") -> entity.substring(1).toIntOrNull()
            else -> mapOf("amp" to 38, "lt" to 60, "gt" to 62, "quot" to 34, "apos" to 39, "nbsp" to 32)[entity]
        }
        if (point != null && Character.isValidCodePoint(point)) String(Character.toChars(point)) else match.value
    }
}
