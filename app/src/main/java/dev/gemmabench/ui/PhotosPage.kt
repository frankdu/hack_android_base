package dev.gemmabench.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.benchmark.*
import dev.gemmabench.photos.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

@Composable
fun PhotosPage(state: LabState, controller: LabController, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var access by remember { mutableStateOf(PhotoPermissions.access(context)) }
    var previousAccess by remember { mutableStateOf(state.photos.accessScope ?: access) }
    LifecycleResumeEffect(Unit) {
        access = PhotoPermissions.access(context)
        onPauseOrDispose { }
    }
    LaunchedEffect(access, state.unavailable) {
        if (access != previousAccess && !state.unavailable) {
            previousAccess = access
            controller.resetPhotos()
        }
    }
    val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        val updated = PhotoPermissions.access(context)
        if (updated == access) controller.resetPhotos()
        access = updated
    }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        uri?.let(controller::exportPhotos)
    }
    var batchSize by rememberSaveable { mutableStateOf(PhotoDemoState.DEFAULT_BATCH_SIZE.toString()) }
    var familyName by rememberSaveable { mutableStateOf(ModelSlot.E2B.name) }
    var acceleratorName by rememberSaveable { mutableStateOf(Accelerator.CPU.name) }
    val size = batchSize.toIntOrNull()
    val validSize = size != null && size in 1..PhotoDemoState.MAX_BATCH_SIZE
    val photos = state.photos
    val unfinished = photos.results.any { it.status == PhotoStatus.PENDING || it.status == PhotoStatus.ANALYZING }
    val accelerator = if (unfinished) Accelerator.valueOf(photos.textBackend) else Accelerator.valueOf(acceleratorName)
    val family = if (unfinished) {
        if (photos.modelSlot == ModelSlot.E4B) ModelSlot.E4B else ModelSlot.E2B
    } else ModelSlot.valueOf(familyName)
    val slot = if (unfinished) photos.modelSlot else modelSlot(family, accelerator, state.tpuSupport)
    fun available(family: ModelSlot, backend: Accelerator): Boolean {
        val candidate = modelSlot(family, backend, state.tpuSupport) ?: return false
        return state.models.any { it.slot == candidate && it.unavailableReason(backend, state.tpuSupport, photos = true) == null }
    }
    val model = state.models.firstOrNull { it.slot == slot }
    val reason = when {
        accelerator == Accelerator.TPU && !state.tpuSupport.available -> state.tpuSupport.reason
        slot == null -> "E4B TPU is unavailable: no official Tensor artifact is published."
        model == null -> "Download ${slot.label} in Models first."
        else -> model.unavailableReason(accelerator, state.tpuSupport, photos = true)
    }
    val modelReady = model != null && reason == null
    // Select an installed compatible combination on first entry or after model provisioning changes.
    LaunchedEffect(state.models, state.tpuSupport, unfinished, state.unavailable) {
        if (!unfinished && !state.unavailable && !available(family, accelerator)) {
            val choice = listOf(ModelSlot.E2B, ModelSlot.E4B).flatMap { candidate ->
                Accelerator.entries.map { candidate to it }
            }.firstOrNull { (candidate, backend) -> available(candidate, backend) }
            choice?.let { (candidate, backend) -> familyName = candidate.name; acceleratorName = backend.name }
        }
    }

    LazyColumn(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Text("Find major objects", style = MaterialTheme.typography.titleLarge)
            Text("Read device photos in batches, newest added first. Each photo is analyzed locally with Gemma. Boxes are model estimates.", style = MaterialTheme.typography.bodyMedium)
        }
        item {
            Card {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(when (access) {
                        PhotoAccess.FULL -> "Access: all device photos"
                        PhotoAccess.SELECTED -> "Access: selected photos only"
                        PhotoAccess.DENIED -> "Photo access required"
                    }, style = MaterialTheme.typography.titleSmall)
                    Text("Android lets you grant access to all photos or a selection. Changing access starts a fresh scan.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = { permissions.launch(PhotoPermissions.requested()) }, enabled = !state.unavailable) {
                        Text(if (access == PhotoAccess.DENIED) "Grant photo access" else "Change photo access")
                    }
                    TextButton(onClick = {
                        context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}")))
                    }, enabled = !state.unavailable) { Text("App permission settings") }
                }
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = batchSize, onValueChange = { batchSize = it }, label = { Text("Batch size (default 10)") },
                    singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = !validSize, supportingText = { Text("1–${PhotoDemoState.MAX_BATCH_SIZE} photos per batch") },
                    enabled = !state.unavailable && !unfinished, modifier = Modifier.fillMaxWidth(),
                )
                Text("Model", style = MaterialTheme.typography.titleSmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(ModelSlot.E2B, ModelSlot.E4B).forEach { candidate ->
                        FilterChip(
                            selected = family == candidate,
                            onClick = {
                                familyName = candidate.name
                                if (!available(candidate, accelerator)) {
                                    acceleratorName = Accelerator.entries.first { available(candidate, it) }.name
                                }
                            },
                            label = { Text(candidate.name) },
                            enabled = !state.unavailable && !unfinished && Accelerator.entries.any { available(candidate, it) },
                        )
                    }
                }
                Text("Text accelerator", style = MaterialTheme.typography.titleSmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Accelerator.entries.forEach { backend ->
                        FilterChip(selected = accelerator == backend, onClick = { acceleratorName = backend.name },
                            label = { Text(backend.name) }, enabled = !state.unavailable && !unfinished && available(family, backend))
                    }
                }
                val vision = if (unfinished) photos.visionBackend else model?.metadata?.let {
                    visionAccelerator(it.visionBackends.orEmpty(), accelerator, state.tpuSupport.available)?.name
                }
                Text("Text: ${accelerator.name} · Vision: ${vision ?: "auto"} · One image at a time", style = MaterialTheme.typography.bodySmall)
                model?.let { Text("Using ${it.slot.label}", style = MaterialTheme.typography.bodySmall) }
                if (family == ModelSlot.E4B) Text("E4B TPU is unavailable.", style = MaterialTheme.typography.bodySmall)
                else if (!available(family, Accelerator.TPU)) Text(if (!state.tpuSupport.available) state.tpuSupport.reason
                    else "Download a vision-capable E2B TPU model in Models to enable TPU.", style = MaterialTheme.typography.bodySmall)
                reason?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Button(
                    onClick = { slot?.let { controller.analyzePhotos(it, accelerator, size ?: PhotoDemoState.DEFAULT_BATCH_SIZE, resume = unfinished) } },
                    enabled = !state.unavailable && access != PhotoAccess.DENIED && validSize && modelReady && (photos.hasMore || unfinished),
                    modifier = Modifier.fillMaxWidth(),
                ) { Text(if (unfinished) "Resume batch" else "Analyze next batch") }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = controller::resetPhotos, enabled = !state.unavailable) { Text("Start over") }
                    OutlinedButton(onClick = { export.launch("gemma-photos-batch-${photos.batchNumber}.json") }, enabled = !state.unavailable && photos.results.isNotEmpty()) { Text("Export JSON") }
                }
                if (photos.batchNumber > 0) {
                    Text("Batch ${photos.batchNumber} · ${photos.results.size} photos · ${photos.modelLabel}", style = MaterialTheme.typography.titleSmall)
                    Text("Only the current batch is retained. Export before loading the next batch. Results clear when the app process exits.", style = MaterialTheme.typography.bodySmall)
                }
                if (!photos.hasMore) Text("End of accessible photos. Start over to include newly added photos.")
            }
        }
        if (access != PhotoAccess.DENIED) items(photos.results, key = { it.photo.id }) { result ->
            PhotoResultCard(result, if (result.status == PhotoStatus.ANALYZING) state.output else "")
        }
    }
}

private val boxColors = listOf(Color(0xFFFFD600), Color(0xFF00E5FF), Color(0xFFFF80AB), Color(0xFF76FF03))

@Composable
private fun PhotoResultCard(result: PhotoResult, streaming: String) {
    var rawVisible by rememberSaveable(result.photo.id) { mutableStateOf(false) }
    Card {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(result.photo.name, style = MaterialTheme.typography.titleSmall)
            Text(when (result.status) {
                PhotoStatus.PENDING -> "Waiting"
                PhotoStatus.ANALYZING -> "Analyzing…"
                PhotoStatus.COMPLETED -> "${result.objects.size} object(s) · ${((result.elapsedMs ?: 0.0) / 1000).roundToInt()} s"
                PhotoStatus.FAILED -> "Analysis failed"
            }, style = MaterialTheme.typography.bodySmall)
            if (result.status == PhotoStatus.ANALYZING) LinearProgressIndicator(Modifier.fillMaxWidth())
            PhotoOverlay(result)
            if (result.status == PhotoStatus.COMPLETED && result.objects.isEmpty()) Text("No major objects identified.")
            SelectionContainer {
                Column {
                    result.objects.forEachIndexed { index, box ->
                        Text("${index + 1}. ${box.name} · [${box.left.roundToInt()}, ${box.top.roundToInt()}, ${box.right.roundToInt()}, ${box.bottom.roundToInt()}]")
                    }
                    if (result.objects.isNotEmpty()) Text("[left, top, right, bottom] · 0–1000 · origin top-left", style = MaterialTheme.typography.labelSmall)
                    result.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                }
            }
            val raw = if (streaming.isNotEmpty()) streaming else result.rawOutput
            if (raw.isNotEmpty()) {
                TextButton(onClick = { rawVisible = !rawVisible }) { Text(if (rawVisible) "Hide raw response" else "Show raw response") }
                if (rawVisible) SelectionContainer { Text(raw, style = MaterialTheme.typography.bodySmall) }
            }
        }
    }
}

@Composable
private fun PhotoOverlay(result: PhotoResult) {
    val path = result.previewPath ?: return
    val preview by produceState<androidx.compose.ui.graphics.ImageBitmap?>(null, path) {
        value = withContext(Dispatchers.IO) {
            BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = 2 })?.asImageBitmap()
        }
    }
    val bitmap = preview ?: return
    val labelPaint = remember { Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = android.graphics.Typeface.DEFAULT_BOLD } }
    Canvas(Modifier.fillMaxWidth().aspectRatio(result.width.toFloat() / result.height).semantics {
        contentDescription = "${result.photo.name}, objects: ${result.objects.joinToString { it.name }}"
    }) {
        drawImage(bitmap, dstSize = IntSize(size.width.roundToInt(), size.height.roundToInt()))
        result.objects.forEachIndexed { index, box ->
            val color = boxColors[index % boxColors.size]
            val left = box.left / 1000f * size.width
            val top = box.top / 1000f * size.height
            drawRect(color, Offset(left, top), Size((box.right - box.left) / 1000f * size.width, (box.bottom - box.top) / 1000f * size.height), style = Stroke(2.dp.toPx()))
            labelPaint.textSize = 13.dp.toPx()
            val label = "${index + 1}. ${box.name.take(30)}"
            val textWidth = labelPaint.measureText(label) + 8.dp.toPx()
            val labelLeft = left.coerceAtMost((size.width - textWidth).coerceAtLeast(0f))
            val labelTop = top.coerceAtMost((size.height - 22.dp.toPx()).coerceAtLeast(0f))
            drawRect(Color.Black.copy(alpha = 0.8f), Offset(labelLeft, labelTop), Size(textWidth.coerceAtMost(size.width), 22.dp.toPx()))
            labelPaint.color = android.graphics.Color.WHITE
            drawContext.canvas.nativeCanvas.drawText(label, labelLeft + 4.dp.toPx(), labelTop + 16.dp.toPx(), labelPaint)
        }
    }
}
