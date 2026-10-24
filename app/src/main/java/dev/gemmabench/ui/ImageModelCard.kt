package dev.gemmabench.ui

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.imagegen.ImageModel
import dev.gemmabench.imagegen.ImagePipeline
import dev.gemmabench.imagegen.ImageMethod
import java.util.Locale

/** One download/status view shared by the Models catalog and the Images playground. */
@Composable
fun ImageModelCard(model: ImageModel, state: LabState, controller: LabController, wifiOnly: Boolean,
    showArtifacts: Boolean = false, selected: Boolean = false, onSelect: (() -> Unit)? = null) {
    val installed = model.id in state.images.ready
    val transfers = state.images.transfers.filter { it.modelId == model.id }
    val context = LocalContext.current
    fun open(url: String) { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
    fun decimal(value: Double) = String.format(Locale.US, "%.2f", value)
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row {
                if (onSelect != null) RadioButton(selected = selected, enabled = !state.busy, onClick = onSelect)
                Column(Modifier.weight(1f)) {
                    Text(model.name, style = MaterialTheme.typography.titleMedium)
                    Text(model.role, style = MaterialTheme.typography.bodySmall)
                    val size = if (model.files.isEmpty()) "Staged bundle"
                        else "${decimal(model.bytes / 1e9)} GB · ${model.files.size} file(s)"
                    val runtime = if (model.pipeline == ImagePipeline.NATIVE) "GGUF" else "LiteRT"
                    Text("$size · ${model.width} × ${model.height} · $runtime", style = MaterialTheme.typography.bodySmall)
                }
            }
            Text(model.description, style = MaterialTheme.typography.bodySmall)
            if (model.pipeline != ImagePipeline.NATIVE) {
                val primary = model.files.firstOrNull()?.repository
                Text("Source: ${primary ?: "Local bundle"}" +
                    if (primary?.startsWith("litert-community/") == true) " · verified frankdu mirror fallback" else "",
                    style = MaterialTheme.typography.bodySmall)
                val tpu = state.imageBenchmark.methods.firstOrNull { it.method == ImageMethod.TPU }
                Text("TPU: ${tpu?.reasonFor(model.id) ?: "Checking runtime…"}", style = MaterialTheme.typography.bodySmall)
            }
            val status = when {
                installed -> "Ready · SHA-256 verified"
                transfers.any { it.error != null } -> "Download needs attention"
                transfers.any { it.active } -> "Download in progress"
                model.files.isEmpty() -> "Bundle not installed"
                else -> "Not downloaded"
            }
            Text(status, color = when {
                installed -> MaterialTheme.colorScheme.primary
                transfers.any { it.error != null } -> MaterialTheme.colorScheme.error
                else -> MaterialTheme.colorScheme.onSurfaceVariant
            })
            transfers.forEach { transfer ->
                Text("${transfer.filename}: ${transfer.phase}${transfer.repository?.let { " · $it" } ?: ""}", style = MaterialTheme.typography.bodySmall)
                if (transfer.active) {
                    LinearProgressIndicator(progress = { (transfer.bytes.toFloat() / transfer.total).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth())
                    Text("${decimal(transfer.bytes / 1e9)} / ${decimal(transfer.total / 1e9)} GB", style = MaterialTheme.typography.labelSmall)
                }
                transfer.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
            if (transfers.isNotEmpty()) {
                OutlinedButton(onClick = { controller.cancelImageDownload(model.id) }, enabled = !state.busy) {
                    Text(if (transfers.any { it.active }) "Cancel download" else "Clear transfer")
                }
            }
            if (!installed && model.files.isNotEmpty() && transfers.none { it.active }) {
                Button(onClick = { controller.downloadImageModel(model.id, wifiOnly) }, enabled = !state.unavailable) {
                    Text(if (transfers.isEmpty()) "Download ${model.name}" else "Retry download")
                }
            }
            Text(model.license, style = MaterialTheme.typography.labelSmall)
            TextButton(onClick = { open(model.source) }) { Text("Model card and license") }
            if (showArtifacts) model.files.forEach { spec ->
                TextButton(onClick = { open("https://huggingface.co/${spec.repository}/blob/${spec.revision}/${spec.remotePath}") }) {
                    Text("Artifact: ${spec.filename}", style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}
