package dev.gemmabench.data

import java.io.Reader

data class WordCategory(val name: String, val items: List<String>)

object KidsEnglishDictionary {
    const val ASSET_NAME = "kids_english_dictionary.txt"

    /** Parses blank-line-separated sections, preserving item text and source order.
     * The caller owns [reader] and must close it.
     */
    fun parse(reader: Reader): List<WordCategory> {
        val categories = mutableListOf<WordCategory>()
        var name: String? = null
        val items = mutableListOf<String>()

        fun finishCategory() {
            name?.let {
                require(items.isNotEmpty()) { "Category '$it' has no items." }
                categories += WordCategory(it, items.toList())
            }
            name = null
            items.clear()
        }

        reader.buffered().lineSequence().forEachIndexed { index, raw ->
            val line = (if (index == 0) raw.removePrefix("\uFEFF") else raw).trim()
            when {
                line.isEmpty() -> finishCategory()
                line.startsWith("Category:") -> {
                    finishCategory()
                    name = line.removePrefix("Category:").trim()
                    require(!name.isNullOrEmpty()) { "Missing category name at line ${index + 1}." }
                }
                else -> {
                    require(name != null) { "Expected a Category: header at line ${index + 1}." }
                    items += line
                }
            }
        }
        finishCategory()
        require(categories.isNotEmpty()) { "The dictionary has no categories." }
        return categories.toList()
    }
}
