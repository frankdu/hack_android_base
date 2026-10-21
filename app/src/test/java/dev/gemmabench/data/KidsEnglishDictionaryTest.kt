package dev.gemmabench.data

import com.google.gson.JsonParser
import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Test

class KidsEnglishDictionaryTest {
    @Test fun speechAssetsCoverEveryWordAndMatchTheSeed() {
        val loader = requireNotNull(javaClass.classLoader)
        val source = requireNotNull(loader.getResourceAsStream(KidsEnglishDictionary.ASSET_NAME)).use { it.readBytes() }
        val data = requireNotNull(loader.getResourceAsStream("dictionary-tokens.json")).bufferedReader().use {
            JsonParser.parseReader(it).asJsonObject
        }
        val hash = MessageDigest.getInstance("SHA-256").digest(source).joinToString("") { "%02x".format(it) }
        assertEquals("Run scripts/prepare_dictionary_tts.py after vocabulary edits", hash, data["dictionarySha256"].asString)
        val expected = KidsEnglishDictionary.parse(source.toString(Charsets.UTF_8).reader()).flatMap { it.items }.toSet()
        val words = data.getAsJsonObject("words")
        assertEquals(expected, words.keySet())
        val kokoroStyles = requireNotNull(loader.getResourceAsStream("kokoro-voice.bin")).use { it.readBytes().size / 1024 }
        val kittenStyles = requireNotNull(loader.getResourceAsStream("kitten-voice.bin")).use { it.readBytes().size / 1024 }
        for ((word, value) in words.entrySet()) {
            val entry = value.asJsonObject
            for ((model, vocabSize) in listOf("kokoro" to 178, "kitten" to 180)) {
                val ids = entry.getAsJsonArray(model)
                assertTrue("$model: $word", ids.size() in 1..510 && ids.all { it.asInt in 0 until vocabSize })
            }
            assertTrue("Kokoro style: $word", entry.getAsJsonArray("kokoro").size() < kokoroStyles)
            assertTrue("Kitten style: $word", entry["kittenStyle"].asInt in 0 until kittenStyles)
        }
    }
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
