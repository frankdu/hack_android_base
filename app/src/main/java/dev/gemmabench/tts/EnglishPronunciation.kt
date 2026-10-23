package dev.gemmabench.tts

import android.content.Context
import com.google.gson.JsonParser
import java.util.Locale
import java.util.zip.GZIPInputStream

/** General English speech, with spelling for words outside the bundled pronunciation lexicon. */
class EnglishPronunciation(private val context: Context) {
    private val lexicon by lazy {
        GZIPInputStream(context.assets.open("tts/english-lexicon.bin")).bufferedReader().useLines { lines ->
            lines.map { it.split('\t', limit = 2) }.associate { it[0] to it[1] }
        }
    }
    private val kokoro by lazy {
        context.assets.open("tts/kokoro-vocab.json").bufferedReader().use { reader ->
            JsonParser.parseReader(reader).asJsonObject.entrySet().associate { it.key.single() to it.value.asLong }
        }
    }
    private val kitten by lazy {
        ("$" + ";:,.!?¡¿—…\"«»\"\" " + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
            .mapIndexed { index, c -> c to index.toLong() }.toMap()
    }
    private fun phones(word: String): String {
        val key = word.lowercase(Locale.US).replace('’', '\'')
        when (key) { "a" -> return "ə"; "an" -> return "ən"; "the" -> return "ðə"; "to" -> return "tə" }
        lexicon[key]?.let { return it }
        if (key.endsWith("'s")) return phones(key.dropLast(2)) + "z"
        val digits = listOf("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
        return key.mapNotNull { c -> lexicon[if (c.isDigit() && c in '0'..'9') digits[c - '0'] else c.toString()] }
            .joinToString(" ").ifBlank { lexicon.getValue("word") }
    }
    fun tokenize(text: String, option: SpeechOption): SpeechTokens {
        val vocab = if (option == SpeechOption.KOKORO) kokoro else kitten
        val ids = mutableListOf<Long>()
        val spans = mutableListOf<Pair<IntRange, IntRange>>()
        val ranges = wordRanges(text)
        for ((index, range) in ranges.withIndex()) {
            var ps = phones(text.substring(range))
            if (option == SpeechOption.KITTEN) {
                ps = ps.replace("A", "eɪ").replace("I", "aɪ").replace("O", "oʊ")
                    .replace("W", "aʊ").replace("Y", "ɔɪ").replace("T", "ɾ")
                    .replace("ʤ", "dʒ").replace("ʧ", "tʃ").replace("ɜɹ", "ɜːɹ")
            }
            val start = ids.size
            ids += ps.mapNotNull(vocab::get)
            spans += range to (start until ids.size)
            val next = ranges.getOrNull(index + 1)?.first ?: text.length
            val punctuation = text.substring(range.last + 1, next).filter { it in ",.!?;:" }
            ids += punctuation.mapNotNull(vocab::get)
            if (index < ranges.lastIndex) vocab[' ']?.let(ids::add)
        }
        require(ids.size in 1..510) { "Please use a shorter sentence." }
        return SpeechTokens(ids.toLongArray(), spans, text.length)
    }
}
