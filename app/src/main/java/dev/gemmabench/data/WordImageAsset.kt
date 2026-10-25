package dev.gemmabench.data

import java.text.Normalizer

/** Matches scripts/generate_word_images.py, including case and existing dashes. */
object WordImageAsset {
    private val specialCharacters = Regex("[^\\p{L}\\p{N}_-]+")
    private fun component(text: String) = Normalizer.normalize(text, Normalizer.Form.NFC)
        .replace(specialCharacters, "-").trim('-')

    fun path(word: String, category: String): String =
        "word-images/${component(word)}_${component(category)}.jpg"
}
