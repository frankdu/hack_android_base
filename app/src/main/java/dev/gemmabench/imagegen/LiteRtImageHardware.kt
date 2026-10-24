package dev.gemmabench.imagegen

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.Environment

/** Probe once on the Lab worker, not each Compose render or download-poll tick. */
object LiteRtImageHardware {
    @Volatile private var accelerators: Set<Accelerator>? = null
    fun available(context: Context): Set<Accelerator> = accelerators ?: synchronized(this) {
        accelerators ?: runCatching {
            Environment.create(context, mapOf(Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir)).use {
                it.getAvailableAccelerators().toSet()
            }
        }.getOrDefault(emptySet()).also { accelerators = it }
    }
}
