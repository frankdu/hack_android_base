package dev.gemmabench.imagegen

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import dev.gemmabench.download.ModelFileVerifier
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File

/** Transfers survive process exit. Only SHA-256 verified files get a ready marker. */
class ImageModelStore(private val context: Context) {
    private val manager = context.getSystemService(DownloadManager::class.java)
    private val preferences = context.getSharedPreferences("image-model-downloads", Context.MODE_PRIVATE)
    private val gson = Gson()
    private data class Ticket(val id: Long, val modelId: String, val filename: String, val error: String? = null)
    fun directory() = requireNotNull(context.getExternalFilesDir("image-models")).apply {
        check(isDirectory || mkdirs()) { "Cannot create image model storage." }
    }
    fun file(spec: ImageFile) = File(directory(), spec.filename)
    private fun marker(spec: ImageFile) = File(directory(), spec.filename + ".verified")
    private fun installed(spec: ImageFile) = file(spec).length() == spec.bytes &&
        marker(spec).takeIf { it.isFile }?.readText() == spec.sha256
    fun ready() = ImageModels.all.filter { it.files.isNotEmpty() && it.files.all(::installed) }.map { it.id }.toSet() +
        LiteRtImageBundles(context, portable = true).installed()

    fun enqueue(model: ImageModel, wifiOnly: Boolean) {
        require(model.files.isNotEmpty()) { "Stage and verify this converted bundle; see docs/IMAGE_LITERT.md." }
        cancel(model.id)
        val missing = model.files.filterNot(::installed)
        check(StatFs(directory().path).availableBytes >= missing.sumOf { it.bytes } + 128L * 1024 * 1024) {
            "Not enough storage. Free ${"%.2f".format(missing.sumOf { it.bytes } / 1e9)} GB plus 128 MiB."
        }
        try {
            for (spec in missing) {
                file(spec).parentFile?.mkdirs()
                File(directory(), spec.filename + ".part").delete()
                val request = DownloadManager.Request(Uri.parse(spec.url))
                    .setTitle("${model.name} · ${spec.filename}")
                    .setDescription("Offline vocabulary image model")
                    .setAllowedOverRoaming(false).setAllowedOverMetered(!wifiOnly)
                    .setAllowedNetworkTypes(if (wifiOnly) DownloadManager.Request.NETWORK_WIFI else
                        DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE)
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                    .setDestinationInExternalFilesDir(context, "image-models", spec.filename + ".part")
                val id = manager.enqueue(request)
                try { save(tickets() + Ticket(id, model.id, spec.filename)) }
                catch (e: Exception) { manager.remove(id); throw e }
            }
        } catch (e: Exception) { cancel(model.id); throw e }
    }

    fun cancel(modelId: String) {
        val matches = tickets().filter { it.modelId == modelId }
        matches.forEach {
            manager.remove(it.id)
            File(directory(), it.filename + ".part").delete()
        }
        save(tickets().filterNot { it.modelId == modelId })
    }

    /** Called on the existing lab worker, never while a native model is active. */
    fun refresh(onVerifying: (String) -> Unit): List<ImageTransfer> {
        val result = mutableListOf<ImageTransfer>()
        for (ticket in tickets()) {
            val model = ImageModels.get(ticket.modelId)
            val spec = model.files.first { it.filename == ticket.filename }
            if (ticket.error != null) {
                result += ImageTransfer(model.id, spec.filename, 0, spec.bytes, "Failed", ticket.error)
                continue
            }
            manager.query(DownloadManager.Query().setFilterById(ticket.id)).use { cursor ->
                if (cursor == null || !cursor.moveToFirst()) {
                    val message = "Android removed this download. Tap Retry."
                    markFailed(ticket, message)
                    result += ImageTransfer(model.id, spec.filename, 0, spec.bytes, "Failed", message)
                    return@use
                }
                val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                val bytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)).coerceAtLeast(0)
                if (status == DownloadManager.STATUS_SUCCESSFUL) {
                    try {
                        onVerifying(spec.filename)
                        val partial = File(directory(), spec.filename + ".part")
                        // Recover a process exit between rename and marker/ticket persistence.
                        val source = if (partial.isFile) partial else file(spec)
                        ModelFileVerifier.verify(source, spec.bytes, spec.sha256)
                        if (source == partial) check(partial.renameTo(file(spec))) { "Could not install verified model." }
                        marker(spec).writeText(spec.sha256)
                        save(tickets().filterNot { it.id == ticket.id })
                        manager.remove(ticket.id)
                    } catch (e: Exception) {
                        val message = e.message ?: "Integrity check failed. Retry."
                        markFailed(ticket, message)
                        result += ImageTransfer(model.id, spec.filename, bytes, spec.bytes, "Failed", message)
                    }
                } else {
                    val error = if (status == DownloadManager.STATUS_FAILED) "Download failed (reason $reason). Check storage/network and retry." else null
                    if (error != null) markFailed(ticket, error)
                    result += ImageTransfer(model.id, spec.filename, bytes, spec.bytes,
                        when (status) {
                            DownloadManager.STATUS_RUNNING -> "Downloading"
                            DownloadManager.STATUS_PAUSED -> "Waiting for network / Wi-Fi (reason $reason)"
                            DownloadManager.STATUS_FAILED -> "Failed"
                            else -> "Queued"
                        }, error)
                }
            }
        }
        return result
    }
    fun verifyStaged(model: ImageModel) {
        model.files.forEach { spec ->
            ModelFileVerifier.verify(file(spec), spec.bytes, spec.sha256)
            marker(spec).writeText(spec.sha256)
        }
    }
    private fun markFailed(ticket: Ticket, error: String) = save(tickets().map { if (it.id == ticket.id) it.copy(error = error) else it })
    private fun tickets(): List<Ticket> = preferences.getString("tickets", null)?.let {
        gson.fromJson(it, object : TypeToken<List<Ticket>>() {}.type)
    } ?: emptyList()
    private fun save(tickets: List<Ticket>) {
        check(preferences.edit().putString("tickets", gson.toJson(tickets)).commit()) { "Could not save download state." }
    }
}
