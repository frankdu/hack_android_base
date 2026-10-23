package dev.gemmabench.tts

data class SpokenWord(val start: Int, val end: Int, val startMs: Long, val endMs: Long)
data class SpeechTokens(val ids: LongArray, val words: List<Pair<IntRange, IntRange>>, val styleIndex: Int)

/** Character ranges always refer to the displayed text, including apostrophes. */
fun wordRanges(text: String): List<IntRange> = Regex("[\\p{L}\\p{N}]+(?:['’][\\p{L}]+)*")
    .findAll(text).map { it.range }.toList()

/** Fallback for engines that do not expose alignment; anchored to actual playback time. */
fun estimatedWordTimes(text: String, durationMs: Long): List<SpokenWord> {
    val ranges = wordRanges(text)
    val total = ranges.sumOf { it.count().coerceAtLeast(2) }.coerceAtLeast(1)
    var weight = 0
    return ranges.map { range ->
        val start = weight * durationMs / total
        weight += range.count().coerceAtLeast(2)
        SpokenWord(range.first, range.last + 1, start, weight * durationMs / total)
    }
}

/** Hold partial sentences until a boundary; cap long model sentences for TTS input limits. */
class SpeechSentences {
    private val pending = StringBuilder()
    fun append(delta: String, finished: Boolean = false): List<String> {
        pending.append(delta)
        val ready = mutableListOf<String>()
        while (pending.isNotEmpty()) {
            val boundary = Regex("[.!?](?=\\s|$)|\\n").find(pending)?.range?.last?.plus(1)
            val cut = boundary?.takeIf { it <= 240 } ?: if (pending.length >= 240)
                pending.lastIndexOf(" ", 240).takeIf { it > 0 } ?: 240
            else if (finished) pending.length else break
            pending.substring(0, cut).trim().takeIf { it.isNotEmpty() }?.let(ready::add)
            pending.delete(0, cut)
        }
        return ready
    }
}
