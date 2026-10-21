package dev.gemmabench.ui

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.people.*
import dev.gemmabench.photos.PhotoAccess
import dev.gemmabench.photos.PhotoPermissions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CancellationException
import kotlin.math.roundToInt

@Composable
fun PeoplePage(state: LabState, controller: LabController, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var access by remember { mutableStateOf(PhotoPermissions.access(context)) }
    LifecycleResumeEffect(Unit) {
        access = PhotoPermissions.access(context)
        controller.refreshPeople()
        onPauseOrDispose { }
    }
    val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        access = PhotoPermissions.access(context)
        controller.refreshPeople()
    }
    var sizeText by rememberSaveable { mutableStateOf("10") }
    val size = sizeText.toIntOrNull()
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    var renaming by remember { mutableStateOf<PersonGroup?>(null) }
    var merging by remember { mutableStateOf<PersonGroup?>(null) }
    var moving by remember { mutableStateOf<VisibleFace?>(null) }
    var clearing by remember { mutableStateOf(false) }
    val people = state.people
    val selected = people.groups.firstOrNull { it.id == selectedId }
    LazyColumn(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            Text("Find people", style = MaterialTheme.typography.titleLarge)
            Text("Group visible faces across your photos. Names and corrections stay on this device.")
            Text(when (access) {
                PhotoAccess.FULL -> "Access: all photos"
                PhotoAccess.SELECTED -> "Access: selected photos only"
                PhotoAccess.DENIED -> "Grant photo access to begin."
            })
            TextButton(onClick = { permissions.launch(PhotoPermissions.requested()) }, enabled = !state.unavailable) {
                Text(if (access == PhotoAccess.DENIED) "Grant photo access" else "Change photo access")
            }
        }
        if (people.refreshing) item {
            Text("Refreshing accessible photos…")
            if (!state.busy) TextButton(onClick = controller::refreshPeople) { Text("Retry refresh") }
        }
        if (access != PhotoAccess.DENIED && !people.refreshing) {
            item {
                if (!people.calibrated) Text("Automatic matching is unavailable until a validated calibration profile is bundled. You can still find faces and merge groups manually.")
                OutlinedTextField(value = sizeText, onValueChange = { sizeText = it }, label = { Text("Photos per batch (1–50)") },
                    singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    enabled = !state.unavailable && !people.unfinished, isError = size == null || size !in 1..50)
                Button(onClick = { controller.findPeople(size ?: 10, people.unfinished) },
                    enabled = !state.unavailable && size != null && size in 1..50 && (people.hasMore || people.unfinished)) {
                    Text(if (people.unfinished) "Resume people batch" else "Find people in next batch")
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = controller::refreshPeople, enabled = !state.unavailable) { Text("Start over") }
                    TextButton(onClick = { clearing = true }, enabled = !state.unavailable) { Text("Clear person data") }
                }
                if (people.batchNumber > 0) {
                    val done = people.results.count { it.status != "PENDING" }
                    Text("Batch ${people.batchNumber} · $done/${people.results.size} processed")
                    LinearProgressIndicator(progress = { if (people.results.isEmpty()) 0f else done.toFloat() / people.results.size }, modifier = Modifier.fillMaxWidth())
                }
                if (!people.hasMore) Text("End of accessible photos. Start over to scan again.")
            }
            items(people.results, key = { "result:${it.photo.uri}" }) { result ->
                Text("${result.photo.name}: " + when (result.status) {
                    "COMPLETED" -> "${result.faces} face(s)${if (result.cached) " · Cached" else ""}"
                    "FAILED" -> "Failed · ${result.error}"
                    else -> "Waiting"
                }, style = MaterialTheme.typography.bodySmall)
            }
            if (selected != null) {
                item {
                    TextButton(onClick = { selectedId = null }) { Text("All people") }
                    Text(selected.label, style = MaterialTheme.typography.titleLarge)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = { renaming = selected }, enabled = !state.unavailable) { Text("Name") }
                        TextButton(onClick = { merging = selected }, enabled = !state.unavailable && people.groups.size > 1) { Text("Merge with…") }
                    }
                }
                items(selected.faces, key = { it.id }) { face ->
                    Card {
                        Column(Modifier.padding(12.dp)) {
                            FaceThumbnail(face, crop = false)
                            TextButton(onClick = { moving = face }, enabled = !state.unavailable) { Text("Move this face…") }
                        }
                    }
                }
            } else {
                item { Text("People · ${people.groups.size}", style = MaterialTheme.typography.titleMedium) }
                items(people.groups, key = { it.id }) { group ->
                    Card(onClick = { selectedId = group.id }) {
                        Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            FaceThumbnail(group.faces.first(), crop = true)
                            Column { Text(group.label); Text("${group.photoCount} photo(s)") }
                        }
                    }
                }
                if (people.unassigned.isNotEmpty()) {
                    item { Text("Unassigned faces", style = MaterialTheme.typography.titleMedium) }
                    items(people.unassigned, key = { it.id }) { face ->
                        Card {
                            Row(Modifier.padding(12.dp)) {
                                FaceThumbnail(face, crop = true)
                                TextButton(onClick = { moving = face }, enabled = !state.unavailable) { Text("Assign face…") }
                            }
                        }
                    }
                }
            }
        }
    }
    renaming?.let { group ->
        var name by remember(group.id) { mutableStateOf(group.name.orEmpty()) }
        AlertDialog(onDismissRequest = { renaming = null }, title = { Text("Name this person") },
            text = { OutlinedTextField(value = name, onValueChange = { name = it.take(120) }, singleLine = true) },
            confirmButton = { TextButton(onClick = { controller.renamePerson(group.id, name); renaming = null }, enabled = !state.unavailable) { Text("Save") } },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("Cancel") } })
    }
    if (merging != null || moving != null) {
        val source = merging
        val face = moving
        var target by remember(source?.id, face?.id) { mutableStateOf<String?>(null) }
        AlertDialog(onDismissRequest = { merging = null; moving = null }, title = { Text(if (face != null) "Move face" else "Merge person groups") },
            text = {
                LazyColumn(Modifier.heightIn(max = 320.dp)) {
                    if (face != null) item { TextButton(onClick = { target = null }) { Text("${if (target == null) "● " else ""}New person group") } }
                    items(people.groups.filter { it.id != (source?.id ?: face?.personId) }, key = { it.id }) { group ->
                        TextButton(onClick = { target = group.id }) { Text("${if (target == group.id) "● " else ""}${group.label}") }
                    }
                }
            },
            confirmButton = { TextButton(enabled = !state.unavailable && (face != null || target != null), onClick = {
                if (face != null) controller.movePersonFace(face.id, target)
                else source?.let { controller.mergePeople(it.id, requireNotNull(target)); selectedId = target }
                moving = null; merging = null
            }) { Text(if (face != null) "Move" else "Merge") } },
            dismissButton = { TextButton(onClick = { moving = null; merging = null }) { Text("Cancel") } })
    }
    if (clearing) AlertDialog(onDismissRequest = { clearing = false }, title = { Text("Clear person data?") },
        text = { Text("Delete all face embeddings, groups, names, and corrections. Your photos remain on the device.") },
        confirmButton = { TextButton(onClick = { controller.clearPeople(); selectedId = null; clearing = false }, enabled = !state.unavailable) { Text("Clear") } },
        dismissButton = { TextButton(onClick = { clearing = false }) { Text("Cancel") } })
}

@Composable
private fun FaceThumbnail(face: VisibleFace, crop: Boolean) {
    val context = LocalContext.current
    val preview by produceState<androidx.compose.ui.graphics.ImageBitmap?>(null, face.id, crop) {
        value = withContext(Dispatchers.IO) {
            try {
                val bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, Uri.parse(face.photoUri))) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val scale = minOf(1f, 768f / maxOf(info.size.width, info.size.height))
                    decoder.setTargetSize(maxOf(1, (info.size.width * scale).roundToInt()), maxOf(1, (info.size.height * scale).roundToInt()))
                }
                if (crop) {
                    val left = (face.left / 1000 * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
                    val top = (face.top / 1000 * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
                    val width = ((face.right - face.left) / 1000 * bitmap.width).toInt().coerceIn(1, bitmap.width - left)
                    val height = ((face.bottom - face.top) / 1000 * bitmap.height).toInt().coerceIn(1, bitmap.height - top)
                    val cropped = Bitmap.createBitmap(bitmap, left, top, width, height)
                    if (cropped !== bitmap) bitmap.recycle()
                    cropped.asImageBitmap()
                } else bitmap.asImageBitmap()
            } catch (e: CancellationException) { throw e } catch (_: Exception) { null }
        }
    }
    val bitmap = preview
    if (bitmap != null && crop) Image(bitmap, contentDescription = "Person face", modifier = Modifier.size(80.dp))
    else if (bitmap != null) Canvas(Modifier.fillMaxWidth().aspectRatio(bitmap.width.toFloat() / bitmap.height)) {
        drawImage(bitmap, dstSize = IntSize(size.width.roundToInt(), size.height.roundToInt()))
        drawRect(Color.Yellow, Offset(face.left / 1000 * size.width, face.top / 1000 * size.height),
            Size((face.right - face.left) / 1000 * size.width, (face.bottom - face.top) / 1000 * size.height), style = Stroke(3.dp.toPx()))
    }
    else Text("Preview unavailable")
}
