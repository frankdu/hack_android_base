package dev.gemmabench.imagegen

import org.junit.Assert.*
import org.junit.Test

class WordImagePromptTest {
    @Test fun categoryDisambiguatesAndShowsTheMeaning() {
        val fruit = WordImagePrompt.build("apple", "Fruit")
        assertTrue(fruit.startsWith("apple,"))
        assertTrue(fruit.contains("Fruit"))
        assertTrue(WordImagePrompt.build("jump", "Actions").contains("performing jump"))
        assertTrue(WordImagePrompt.build("happy", "Feelings").contains("facial expression"))
        assertTrue(WordImagePrompt.build("red", "Colors").contains("red color"))
        assertTrue(WordImagePrompt.build("circle", "Shapes").contains("circle shape"))
        assertFalse(fruit.contains("performing"))
    }
    @Test fun normalizesAndRejectsEmptyFields() {
        assertEquals(WordImagePrompt.build("red apple", "Fruit"), WordImagePrompt.build(" red\n apple ", " Fruit "))
        for ((word, category) in listOf("" to "Fruit", "apple" to " ", "a".repeat(81) to "Fruit")) {
            assertThrows(IllegalArgumentException::class.java) { WordImagePrompt.build(word, category) }
        }
    }
    @Test fun catalogIncludesAllFluxComponentsAndPinnedIntegrity() {
        assertEquals(8, ImageModels.all.size)
        val flux = ImageModels.get("FLUX_KLEIN")
        assertEquals(3, flux.files.size)
        assertTrue(flux.flux)
        ImageModels.all.flatMap { it.files }.forEach {
            assertTrue(it.bytes > 0)
            assertTrue(it.sha256.matches(Regex("[0-9a-f]{64}")))
            assertTrue(it.revision.matches(Regex("[0-9a-f]{40}")))
            assertTrue(it.url.contains("/resolve/${it.revision}/"))
        }
    }
}
