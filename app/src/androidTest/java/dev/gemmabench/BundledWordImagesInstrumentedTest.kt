package dev.gemmabench

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.data.BundledWordImages
import dev.gemmabench.data.KidsEnglishDictionary
import dev.gemmabench.data.WordImageAsset
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BundledWordImagesInstrumentedTest {
    @Test fun apkContainsEveryDictionaryIllustrationAndLoadsSampledThumbnails() {
        val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
        val categories = assets.open(KidsEnglishDictionary.ASSET_NAME).bufferedReader().use(KidsEnglishDictionary::parse)
        val expected = categories.flatMap { category -> category.items.map { word ->
            WordImageAsset.path(word, category.name).substringAfter('/')
        } }.toSet()
        assertEquals(10_000, expected.size)
        assertEquals(expected, assets.list("word-images")!!.toSet())
        val images = BundledWordImages(assets)
        for ((word, category) in listOf("dog" to "animal", "Toronto" to "city",
            "thin - thinner - thinnest" to "comparing words")) {
            val jpeg = requireNotNull(images.jpeg(word, category))
            assertEquals(0xff.toByte(), jpeg[0])
            assertEquals(0xd8.toByte(), jpeg[1])
            val thumbnail = requireNotNull(images.bitmap(word, category, 56))
            val large = requireNotNull(images.bitmap(word, category, 520))
            try {
                assertTrue(thumbnail.width < large.width)
                assertEquals(520, large.width)
                assertEquals(520, large.height)
            } finally { thumbnail.recycle(); large.recycle() }
        }
        assertNull(images.jpeg("missing word", "missing category"))
        assertNull(images.bitmap("missing word", "missing category", 56))
    }
}
