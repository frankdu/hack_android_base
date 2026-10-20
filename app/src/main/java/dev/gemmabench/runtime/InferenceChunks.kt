package dev.gemmabench.runtime

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer

/** LiteRT-LM 0.18.0 uses trySend without checking failures. Fuse its callbackFlow with a lossless queue.
 * Requests have a token limit; the queue only retains that request's pending text chunks. */
internal fun <T> Flow<T>.bufferInferenceChunks(): Flow<T> = buffer(Channel.UNLIMITED)
