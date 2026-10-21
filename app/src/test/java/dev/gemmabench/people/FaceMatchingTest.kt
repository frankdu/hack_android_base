package dev.gemmabench.people

import org.junit.Assert.*
import org.junit.Test

class FaceMatchingTest {
    @Test fun rejectsAmbiguousAndUncalibratedMatches() {
        val scores = listOf("a" to 0.8f, "b" to 0.78f)
        assertNull(FaceMatching.choose(scores, MatchPolicy(0.6f, 0.05f, true)))
        assertNull(FaceMatching.choose(listOf("a" to 0.9f), MatchPolicy(0.6f, 0.05f)))
        assertNull(FaceMatching.choose(listOf("a" to 0.5f), MatchPolicy(0.6f, 0.05f, true)))
        assertEquals("a", FaceMatching.choose(listOf("a" to 0.8f, "b" to 0.6f), MatchPolicy(0.6f, 0.05f, true)))
    }

    @Test fun embeddingsRoundTripAndCompare() {
        val a = FaceMatching.normalize(floatArrayOf(3f, 4f))
        assertArrayEquals(a, FaceMatching.decode(FaceMatching.encode(a)), 0f)
        assertEquals(1f, FaceMatching.cosine(a, a), 0.00001f)
        assertEquals(0f, FaceMatching.cosine(floatArrayOf(1f, 0f), floatArrayOf(0f, 1f)), 0f)
        for (bad in listOf(floatArrayOf(), floatArrayOf(0f), floatArrayOf(Float.NaN))) {
            assertThrows(IllegalArgumentException::class.java) { FaceMatching.normalize(bad) }
        }
    }
}
