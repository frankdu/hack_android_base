package dev.gemmabench.imagegen

/** Calls are serialized by LabController; cancellation alone may come from another thread. */
object DiffusionNative {
    init { System.loadLibrary("word_diffusion") }
    fun interface Progress { fun onProgress(step: Int, total: Int, seconds: Float) }
    external fun resetStop()
    external fun requestStop()
    external fun generate(model: String, encoder: String, vae: String, prompt: String,
        flux: Boolean, steps: Int, seed: Long, threads: Int, listener: Progress): IntArray?
}
