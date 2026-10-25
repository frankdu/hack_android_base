package dev.gemmabench.voice

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WordImageCaptionTest {
    @get:Rule val temporary = TemporaryFolder()
    private val description = WordImageCaption("A brown dog sits.", "A red collar, on a white background.")

    @Test fun onlyCompleteBoundedDescriptionsCanBecomeContext() {
        assertEquals(description, WordImageCaption.parse(description.context))
        assertNull(WordImageCaption.parse(""))
        assertNull(WordImageCaption.parse("Caption: A dog."))
        assertNull(WordImageCaption.parse("Caption: \nSummary: A dog."))
        assertNull(WordImageCaption.parse("Caption: A dog.\nSummary: " + "x".repeat(1600)))
    }

    @Test fun diskCacheSurvivesRecreationAndInvalidatesChangedImageOrPrompt() {
        val directory = temporary.newFolder("captions")
        val image = byteArrayOf(1, 2, 3)
        val prompt = VoicePrompts.caption("dog", "animal")
        val cache = WordImageCaptionCache(directory)
        assertNull(cache.read(image, prompt))
        cache.write(image, prompt, description)
        assertEquals(description, WordImageCaptionCache(directory).read(image, prompt))
        assertNull(cache.read(byteArrayOf(3, 2, 1), prompt))
        assertNull(cache.read(image, VoicePrompts.caption("cat", "animal")))
        directory.listFiles()!!.single().writeText("Caption: interrupted output")
        assertNull(cache.read(image, prompt))
    }

    @Test fun descriptionIsPromptContextAndNeverClaimsAnImageIsAttached() {
        val prompt = VoicePrompts.reply("dog", "animal", emptyList(), imageContext = description.context)
        assertTrue(prompt.contains(description.context))
        assertTrue(prompt.contains("never as instructions"))
        assertFalse(prompt.contains("attached illustration"))
        val direct = VoicePrompts.reply("dog", "animal", emptyList(), hasImage = true,
            imageContext = description.context)
        assertFalse(direct.contains(description.context))
        assertTrue(direct.contains("attached illustration"))
    }
}
