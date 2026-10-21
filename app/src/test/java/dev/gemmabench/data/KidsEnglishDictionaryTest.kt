package dev.gemmabench.data

import org.junit.Assert.*
import org.junit.Test

class KidsEnglishDictionaryTest {
    @Test fun parsesSectionsAndPreservesItemTextAndOrder() {
        val source = "\uFEFF  Category: animal  \r\n dog \r\nguinea pig\r\n\r\n\r\nCategory: past tense verbs\r\ngo - went"
        assertEquals(listOf(
            WordCategory("animal", listOf("dog", "guinea pig")),
            WordCategory("past tense verbs", listOf("go - went")),
        ), KidsEnglishDictionary.parse(source.reader()))
    }

    @Test fun acceptsTrailingBlankLinesAndAdjacentHeaders() {
        assertEquals(listOf(
            WordCategory("animal", listOf("dog")),
            WordCategory("bird", listOf("owl")),
        ), KidsEnglishDictionary.parse("\nCategory: animal\ndog\nCategory: bird\nowl\n\n".reader()))
    }

    @Test fun rejectsMissingHeadersNamesAndItems() {
        listOf("", "\n \n", "dog", "Category: \ndog", "Category: animal\n",
            "Category: animal\ndog\n\ncat", "Category: animal\n\nCategory: bird\nowl").forEach { source ->
            assertThrows(source, IllegalArgumentException::class.java) {
                KidsEnglishDictionary.parse(source.reader())
            }
        }
    }

    @Test fun loadsTheActualSeedIncludingItsLastSection() {
        val categories = requireNotNull(javaClass.classLoader?.getResourceAsStream(KidsEnglishDictionary.ASSET_NAME))
            .bufferedReader(Charsets.UTF_8).use(KidsEnglishDictionary::parse)
        assertEquals(100, categories.size)
        assertTrue(categories.all { it.items.isNotEmpty() })
        assertEquals(10000, categories.sumOf { it.items.size })
        assertEquals("animal", categories.first().name)
        assertEquals("dog", categories.first().items.first())
        assertEquals("jungle and rainforest", categories.last().name)
        assertEquals("flower", categories.last().items.last())
        assertTrue(categories.first { it.name == "past tense verbs" }.items.contains("go - went"))
    }
}
