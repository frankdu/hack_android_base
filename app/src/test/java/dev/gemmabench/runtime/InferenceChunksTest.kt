package dev.gemmabench.runtime

import dev.gemmabench.photos.PhotoObjects
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class InferenceChunksTest {
    @Test fun burstyNativeCallbacksDoNotLoseJsonTokensWhenConsumerIsSlow() = runTest {
        val json = "{\"objects\":[" + (1..8).joinToString(",") {
            "{\"name\":\"object $it\",\"box\":[0,0,100,100]}"
        } + "]}"
        val received = StringBuilder()
        // Matches the pinned SDK's unchecked trySend callback; an ordinary 64-slot flow loses data.
        callbackFlow {
            json.forEach { trySend(it.toString()) }
            close()
        }.bufferInferenceChunks().collect {
            delay(1)
            received.append(it)
        }
        assertEquals(json, received.toString())
        assertEquals(8, PhotoObjects.parse(received.toString()).size)
    }
}
