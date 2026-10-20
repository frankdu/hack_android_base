package dev.gemmabench.download

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.IOException

/** Android owns the transfer, network retries and notification even when our process exits. */
class SystemModelDownloads(private val context: Context) {
    private val manager = context.getSystemService(DownloadManager::class.java)
    private val preferences = context.getSharedPreferences("model_downloads", Context.MODE_PRIVATE)
    private val gson = Gson()
    private data class Ticket(val id: Long, val spec: DownloadSpec, val error: String? = null)

    fun directory(): File = requireNotNull(context.getExternalFilesDir("downloaded-models")) {
        "Model storage is unavailable. Check that device storage is mounted."
    }.apply { check(isDirectory || mkdirs()) { "Cannot create model storage." } }

    fun temporaryFile(spec: DownloadSpec): File = File(directory(), spec.temporaryFilename)

    fun enqueue(spec: DownloadSpec, wifiOnly: Boolean): Long {
        val old = tickets().firstOrNull { it.spec.slot == spec.slot }
        if (old != null) cancel(old.id)
        val file = temporaryFile(spec)
        if (file.exists()) check(file.delete()) { "Cannot replace the previous partial download." }
        check(StatFs(directory().absolutePath).availableBytes >= spec.bytes + 128L * 1024 * 1024) {
            "Not enough free space. ${spec.slot.label} needs about ${"%.2f".format(spec.bytes / 1e9)} GB plus 128 MiB free."
        }
        val request = DownloadManager.Request(Uri.parse(spec.url))
            .setTitle("${spec.slot.label} model")
            .setDescription("Downloading for Gemma Bench")
            .setMimeType("application/octet-stream")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setAllowedOverRoaming(false)
            .setAllowedOverMetered(!wifiOnly)
            .setAllowedNetworkTypes(if (wifiOnly) DownloadManager.Request.NETWORK_WIFI else DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE)
            .setDestinationInExternalFilesDir(context, "downloaded-models", spec.temporaryFilename)
            .addRequestHeader("User-Agent", "GemmaBench/0.1 Android")
        val id = manager.enqueue(request)
        try {
            save(tickets() + Ticket(id, spec))
        } catch (e: Exception) {
            manager.remove(id)
            throw e
        }
        return id
    }

    fun statuses(): List<ModelDownload> = tickets().map { ticket ->
        ticket.error?.let { return@map ModelDownload(ticket.id, ticket.spec, TransferPhase.FAILED, detail = it) }
        manager.query(DownloadManager.Query().setFilterById(ticket.id)).use { cursor ->
            if (cursor == null || !cursor.moveToFirst()) {
                return@map ModelDownload(ticket.id, ticket.spec, TransferPhase.FAILED, detail = "Download was removed by Android. Tap Retry.")
            }
            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
            val bytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)).coerceAtLeast(0)
            when (status) {
                DownloadManager.STATUS_PENDING -> ModelDownload(ticket.id, ticket.spec, TransferPhase.QUEUED, bytes, "Queued by Android")
                DownloadManager.STATUS_RUNNING -> ModelDownload(ticket.id, ticket.spec, TransferPhase.DOWNLOADING, bytes)
                DownloadManager.STATUS_PAUSED -> ModelDownload(ticket.id, ticket.spec, TransferPhase.WAITING, bytes, when (reason) {
                    DownloadManager.PAUSED_WAITING_FOR_NETWORK -> "Waiting for a network connection"
                    DownloadManager.PAUSED_QUEUED_FOR_WIFI -> "Waiting for Wi-Fi"
                    DownloadManager.PAUSED_WAITING_TO_RETRY -> "Connection interrupted · Android will retry"
                    else -> "Waiting for Android to resume the transfer"
                })
                DownloadManager.STATUS_SUCCESSFUL -> ModelDownload(ticket.id, ticket.spec, TransferPhase.VERIFYING, bytes, "Checking model integrity")
                else -> ModelDownload(ticket.id, ticket.spec, TransferPhase.FAILED, bytes, failureReason(reason))
            }
        }
    }

    fun markFailed(id: Long, message: String) = save(tickets().map { if (it.id == id) it.copy(error = message) else it })

    /** Forget the completed Android job; its installed model remains at its separately renamed path. */
    fun installed(id: Long) {
        save(tickets().filterNot { it.id == id })
        manager.remove(id)
    }

    fun cancel(id: Long) {
        val ticket = tickets().firstOrNull { it.id == id } ?: return
        manager.remove(id)
        temporaryFile(ticket.spec).delete()
        save(tickets().filterNot { it.id == id })
    }

    private fun tickets(): List<Ticket> = preferences.getString("tickets", null)?.let {
        gson.fromJson(it, object : TypeToken<List<Ticket>>() {}.type)
    } ?: emptyList()

    private fun save(tickets: List<Ticket>) {
        if (!preferences.edit().putString("tickets", gson.toJson(tickets)).commit()) {
            throw IOException("Could not save download state. Check free device storage.")
        }
    }

    private fun failureReason(reason: Int): String = when (reason) {
        DownloadManager.ERROR_INSUFFICIENT_SPACE -> "Device storage is full. Free space, then tap Retry."
        DownloadManager.ERROR_CANNOT_RESUME -> "The server could not resume this transfer. Tap Retry."
        DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "The model host returned too many redirects. Tap Retry."
        DownloadManager.ERROR_DEVICE_NOT_FOUND -> "Download storage is unavailable. Check device storage and retry."
        401, 403 -> "The model host refused access (HTTP $reason). Tap Retry; the official catalog models are public."
        404 -> "The pinned model is unavailable (HTTP 404). Check the app's model catalog."
        in 400..599 -> "The model host returned HTTP $reason. Tap Retry."
        else -> "Android could not finish the download (reason $reason). Check your connection and tap Retry."
    }
}
