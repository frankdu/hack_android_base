package dev.gemmabench.imagegen

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import dev.gemmabench.download.ModelFileVerifier
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File

/** Transfers survive process exit. All sources share the catalog's pinned size and SHA-256. */
class ImageModelStore(private val context: Context) {
    private val manager = context.getSystemService(DownloadManager::class.java)
    private val preferences = context.getSharedPreferences("image-model-downloads", Context.MODE_PRIVATE)
    private val gson = Gson()
    private data class Ticket(val id: Long, val modelId: String, val filename: String,
        val error: String? = null, val sourceIndex: Int = 0)
    fun directory() = requireNotNull(context.getExternalFilesDir("image-models")).apply {
        check(isDirectory || mkdirs()) { "Cannot create image model storage." }
    }
    fun file(spec: ImageFile): File {
        val model = ImageModels.liteRt.firstOrNull { it.pipeline == ImagePipeline.LITERT_PLAN && spec.filename.startsWith(it.id + "/") }
        return if (model == null) LiteRtImageBundles.safeFile(directory(), spec.filename)
        else LiteRtImageBundles.safeFile(LiteRtImageBundles(context, portable = true).directory(model.id), spec.filename.removePrefix(model.id + "/"))
    }
    private fun marker(spec: ImageFile) = File(file(spec).path + ".verified")
    private fun partial(spec: ImageFile) = File(file(spec).path + ".part")
    private fun installed(spec: ImageFile) = file(spec).length() == spec.bytes &&
        marker(spec).takeIf { it.isFile }?.readText() == spec.sha256
    fun ready() = ImageModels.all.filter {
        it.pipeline != ImagePipeline.LITERT_PLAN && it.files.isNotEmpty() && it.files.all(::installed)
    }.map { it.id }.toSet() + LiteRtImageBundles(context, portable = true).installed()

    fun sources(model: ImageModel): Map<String, String> = model.files.associate { spec ->
        spec.filename to runCatching {
            gson.fromJson(File(file(spec).path + ".source.json").readText(), ImageLocation::class.java).url
        }.getOrDefault("verified-local:${spec.filename}")
    }

    fun enqueue(model: ImageModel, wifiOnly: Boolean) {
        require(model.files.isNotEmpty()) { "No downloadable files are available for this model." }
        cancel(model.id)
        preferences.edit().remove("bundle-error-${model.id}").remove("bundle-stamp-${model.id}").apply()
        val missing = model.files.filterNot(::installed)
        check(StatFs(directory().path).availableBytes >= missing.sumOf { it.bytes } + 128L * 1024 * 1024) {
            "Not enough storage. Free ${"%.2f".format(missing.sumOf { it.bytes } / 1e9)} GB plus 128 MiB."
        }
        try {
            for (spec in missing) {
                val ticket = start(model, spec, 0, wifiOnly)
                try { save(tickets() + ticket) }
                catch (e: Exception) { manager.remove(ticket.id); throw e }
            }
        } catch (e: Exception) { cancel(model.id); throw e }
    }

    private fun start(model: ImageModel, spec: ImageFile, sourceIndex: Int, wifiOnly: Boolean): Ticket {
        val destination = file(spec)
        destination.parentFile?.mkdirs()
        partial(spec).delete()
        val location = spec.locations[sourceIndex]
        val relative = partial(spec).relativeTo(requireNotNull(context.getExternalFilesDir(null)).canonicalFile).path
        val request = DownloadManager.Request(Uri.parse(location.url))
            .setTitle("${model.name} · ${destination.name}")
            .setDescription("Offline image model · ${location.repository}")
            .setAllowedOverRoaming(false).setAllowedOverMetered(!wifiOnly)
            .setAllowedNetworkTypes(if (wifiOnly) DownloadManager.Request.NETWORK_WIFI else
                DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationInExternalFilesDir(context, null, relative)
        val id = manager.enqueue(request)
        preferences.edit().putBoolean("wifi-$id", wifiOnly).apply()
        return Ticket(id, model.id, spec.filename, sourceIndex = sourceIndex)
    }

    fun cancel(modelId: String) {
        val matches = tickets().filter { it.modelId == modelId }
        matches.forEach { ticket ->
            manager.remove(ticket.id)
            ImageModels.get(modelId).files.firstOrNull { it.filename == ticket.filename }?.let { partial(it).delete() }
            preferences.edit().remove("wifi-${ticket.id}").apply()
        }
        save(tickets().filterNot { it.modelId == modelId })
    }

    private fun fallback(ticket: Ticket, model: ImageModel, spec: ImageFile): Ticket? {
        if (ticket.sourceIndex + 1 >= spec.locations.size) return null
        val wifiOnly = preferences.getBoolean("wifi-${ticket.id}", true)
        manager.remove(ticket.id)
        val next = start(model, spec, ticket.sourceIndex + 1, wifiOnly)
        try { save(tickets().map { if (it.id == ticket.id) next else it }) }
        catch (e: Exception) { manager.remove(next.id); throw e }
        preferences.edit().remove("wifi-${ticket.id}").apply()
        return next
    }

    /** Runs on the Lab worker. Never performs network probes from a Compose render. */
    fun refresh(onVerifying: (ImageTransfer) -> Unit): List<ImageTransfer> {
        val result = mutableListOf<ImageTransfer>()
        for (ticket in tickets()) {
            val model = ImageModels.get(ticket.modelId)
            val spec = model.files.firstOrNull { it.filename == ticket.filename } ?: continue
            val location = spec.locations[ticket.sourceIndex]
            fun status(bytes: Long, phase: String, error: String? = null) = ImageTransfer(model.id,
                spec.filename, bytes, spec.bytes, phase, error, location.repository)
            if (ticket.error != null) {
                result += status(0, "Failed", ticket.error)
                continue
            }
            manager.query(DownloadManager.Query().setFilterById(ticket.id)).use { cursor ->
                if (cursor == null || !cursor.moveToFirst()) {
                    val message = "Android removed this download. Tap Retry."
                    markFailed(ticket, message); result += status(0, "Failed", message)
                    return@use
                }
                val state = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                val bytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)).coerceAtLeast(0)
                if (state == DownloadManager.STATUS_SUCCESSFUL) {
                    var integrityVerified = false
                    try {
                        onVerifying(status(bytes, "Verifying checksum"))
                        val source = partial(spec).takeIf { it.isFile } ?: file(spec)
                        ModelFileVerifier.verify(source, spec.bytes, spec.sha256)
                        integrityVerified = true
                        if (source == partial(spec)) check(source.renameTo(file(spec))) { "Could not install verified model." }
                        marker(spec).writeText(spec.sha256)
                        File(file(spec).path + ".source.json").writeText(gson.toJson(location))
                        save(tickets().filterNot { it.id == ticket.id })
                        manager.remove(ticket.id)
                        preferences.edit().remove("wifi-${ticket.id}").apply()
                    } catch (e: Exception) {
                        val next = if (!integrityVerified) fallback(ticket, model, spec) else null
                        if (next != null) result += ImageTransfer(model.id, spec.filename, 0, spec.bytes,
                            "Trying verified mirror after integrity failure", repository = spec.locations[next.sourceIndex].repository)
                        else {
                            val message = e.message ?: "Integrity check failed. Retry."
                            markFailed(ticket, message); result += status(bytes, "Failed", message)
                        }
                    }
                } else if (state == DownloadManager.STATUS_FAILED && imageDownloadCanFallback(reason)) {
                    val next = fallback(ticket, model, spec)
                    if (next != null) result += ImageTransfer(model.id, spec.filename, 0, spec.bytes,
                        "Trying verified mirror", repository = spec.locations[next.sourceIndex].repository)
                    else {
                        val message = "Download failed (reason $reason). Check network and retry."
                        markFailed(ticket, message); result += status(bytes, "Failed", message)
                    }
                } else {
                    val error = if (state == DownloadManager.STATUS_FAILED) "Download failed (reason $reason). Check storage/network and retry." else null
                    if (error != null) markFailed(ticket, error)
                    result += status(bytes, when (state) {
                        DownloadManager.STATUS_RUNNING -> "Downloading"
                        DownloadManager.STATUS_PAUSED -> imageDownloadWaitingReason(reason)
                        DownloadManager.STATUS_FAILED -> "Failed"
                        else -> "Queued by Android"
                    }, error)
                }
            }
        }
        val bundles = LiteRtImageBundles(context, portable = true)
        val readyBundles = bundles.installed()
        ImageModels.liteRt.filter { it.pipeline == ImagePipeline.LITERT_PLAN && it.id !in readyBundles && it.files.all(::installed) }.forEach { model ->
            val stamp = model.files.joinToString("|") { "${file(it).length()}:${file(it).lastModified()}" }
            val previous = preferences.getString("bundle-error-${model.id}", null)
            if (previous != null && preferences.getString("bundle-stamp-${model.id}", null) == stamp) {
                result += ImageTransfer(model.id, "manifest.json", 0, 1, "Verification failed", previous)
            } else try {
                onVerifying(ImageTransfer(model.id, "manifest.json", 0, 1, "Verifying model bundle"))
                bundles.verifyModel(model.id)
                preferences.edit().remove("bundle-error-${model.id}").remove("bundle-stamp-${model.id}").apply()
            } catch (e: Exception) {
                val message = e.message ?: "Invalid image bundle"
                preferences.edit().putString("bundle-error-${model.id}", message).putString("bundle-stamp-${model.id}", stamp).apply()
                result += ImageTransfer(model.id, "manifest.json", 0, 1, "Verification failed", message)
            }
        }
        return result
    }
    fun verifyStaged(model: ImageModel) {
        model.files.forEach { spec -> ModelFileVerifier.verify(file(spec), spec.bytes, spec.sha256); marker(spec).writeText(spec.sha256) }
        if (model.pipeline == ImagePipeline.LITERT_PLAN) LiteRtImageBundles(context, portable = true).verifyModel(model.id)
    }
    private fun markFailed(ticket: Ticket, error: String) = save(tickets().map { if (it.id == ticket.id) it.copy(error = error) else it })
    private fun tickets(): List<Ticket> = preferences.getString("tickets", null)?.let {
        gson.fromJson(it, object : TypeToken<List<Ticket>>() {}.type)
    } ?: emptyList()
    private fun save(tickets: List<Ticket>) {
        check(preferences.edit().putString("tickets", gson.toJson(tickets)).commit()) { "Could not save download state." }
    }
}
