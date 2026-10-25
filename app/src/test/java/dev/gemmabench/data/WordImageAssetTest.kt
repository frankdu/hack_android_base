package dev.gemmabench.data

import org.junit.Assert.assertEquals
import org.junit.Test

class WordImageAssetTest {
    @Test fun matchesGeneratedNamesWithoutCollapsingExistingDashesOrChangingCase() {
        assertEquals("word-images/guinea-pig_animal.jpg", WordImageAsset.path("guinea pig", "animal"))
        assertEquals("word-images/thin---thinner---thinnest_comparing-words.jpg",
            WordImageAsset.path("thin - thinner - thinnest", "comparing words"))
        assertEquals("word-images/Toronto_city.jpg", WordImageAsset.path("Toronto", "city"))
        assertEquals("word-images/café_food-and-drink.jpg", WordImageAsset.path(" cafe\u0301! ", "food and drink"))
        assertEquals("word-images/one_two_三.jpg", WordImageAsset.path("one_two", "三"))
    }
}
