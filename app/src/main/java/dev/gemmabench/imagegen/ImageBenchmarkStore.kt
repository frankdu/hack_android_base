package dev.gemmabench.imagegen

import android.content.Context
import android.util.AtomicFile
import com.google.gson.GsonBuilder
import java.io.File

/** All access is on the lab worker. Each checkpoint is an atomic manifest replacement. */
class ImageBenchmarkStore(context: Context) : ImageBenchmarkRepository {
    private val root = File(context.filesDir, "image-benchmarks").apply { mkdirs() }
    private val gson = GsonBuilder().setPrettyPrinting().create()
    private fun directory(id: String): File {
        require(id.matches(Regex("[A-Za-z0-9-]+")))
        return File(root, id).apply { mkdirs() }
    }
    fun images(id: String) = File(directory(id), "images").apply { mkdirs() }
    override fun save(run: ImageBenchmarkRun) {
        val atomic = AtomicFile(File(directory(run.id), "run.json"))
        val stream = atomic.startWrite()
        try { stream.write(gson.toJson(run).toByteArray(Charsets.UTF_8)); atomic.finishWrite(stream) }
        catch (e: Exception) { atomic.failWrite(stream); throw e }
    }
    override fun load(id: String): ImageBenchmarkRun = AtomicFile(File(directory(id), "run.json")).openRead().use {
        gson.fromJson(it.bufferedReader(), ImageBenchmarkRun::class.java)
    }
    fun runs(recover: Boolean = false): List<ImageBenchmarkRun> = root.listFiles().orEmpty().filter { it.isDirectory }.mapNotNull { dir ->
        runCatching {
            var run = load(dir.name)
            if (recover && run.status == ImageRunStatus.RUNNING) {
                run = run.copy(status = ImageRunStatus.INTERRUPTED, samples = run.samples.map {
                    if (it.status == ImageSampleStatus.RUNNING) it.copy(status = ImageSampleStatus.PENDING, error = "Interrupted before completion; resume to retry.") else it
                })
                save(run)
            }
            run
        }.getOrNull()
    }.sortedByDescending { it.startedAtEpochMs }
    fun rate(id: String, sampleId: String, quality: Int?): ImageBenchmarkRun {
        require(quality == null || quality in 1..5)
        val run = load(id)
        require(run.samples.any { it.id == sampleId && it.status == ImageSampleStatus.SUCCESS })
        return run.copy(samples = run.samples.map { if (it.id == sampleId) it.copy(quality = quality) else it },
            updatedAtEpochMs = System.currentTimeMillis()).also(::save)
    }
    fun json(id: String) = gson.toJson(load(id))
}
