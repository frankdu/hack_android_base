package dev.gemmabench.photos

import dev.gemmabench.benchmark.*
import java.util.concurrent.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class PhotoAnalysisTest {
    private val settings = RunSettings(outputTokens = 1024, contextTokens = 4096, prompt = PhotoObjects.PROMPT)
    private fun measurement(text: String) = Measurement(NativeMetrics(null, null, 10, 30, null, null), 100.0, output = text)

    @Test fun retriesTruncatedJsonWithFreshCompactRequestAndKeepsAttemptMetadata() = runTest {
        val prompts = mutableListOf<String>()
        val attempts = mutableListOf<Int>()
        var retries = 0
        val objects = analyzePhotoResponse(settings, infer = {
            prompts += it.prompt
            measurement(if (prompts.size == 1) "{\"objects\":[" else "{\"objects\":[]}")
        }, onMeasurement = { attempt, _, _ -> attempts += attempt }, onRetry = { retries++ })
        assertTrue(objects.isEmpty())
        assertEquals(listOf(1, 2), attempts)
        assertEquals(1, retries)
        assertEquals(PhotoObjects.RETRY_PROMPT, prompts.last())
    }

    @Test fun stopsAfterTwoMalformedResponsesInsteadOfAcceptingPartialBoxes() = runTest {
        var count = 0
        try {
            analyzePhotoResponse(settings, { count++; measurement("{\"objects\":[") }, { _, _, _ -> }, {})
            fail("Must reject incomplete JSON")
        } catch (e: PhotoResponseException) { assertTrue(e.message!!.contains("2 attempts")) }
        assertEquals(2, count)
    }

    @Test fun invalidGeometryIsReportedWithoutClampingOrRetrying() = runTest {
        var count = 0
        try {
            analyzePhotoResponse(settings, { count++; measurement("""{"objects":[{"name":"cat","box":[500,0,0,100]}]}""") },
                { _, _, _ -> }, { fail("Do not retry valid JSON with invalid geometry") })
            fail("Must reject reversed coordinates")
        } catch (e: IllegalArgumentException) { assertFalse(e is PhotoResponseException) }
        assertEquals(1, count)
    }

    @Test fun nativeCancellationDoesNotStartAnotherInvocation() = runTest {
        var count = 0
        try {
            analyzePhotoResponse(settings, { count++; throw CancellationException("Stopped") }, { _, _, _ -> }, { fail() })
            fail("Cancellation should propagate")
        } catch (_: CancellationException) { }
        assertEquals(1, count)
    }
}
