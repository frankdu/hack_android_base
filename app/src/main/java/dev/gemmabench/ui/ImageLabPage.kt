package dev.gemmabench.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.imagegen.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

@Composable
fun ImageLabPage(state: LabState, controller: LabController) {
    var selected by rememberSaveable { mutableStateOf("SD_TURBO") }
    var word by rememberSaveable { mutableStateOf("apple") }
    var category by rememberSaveable { mutableStateOf("Fruit") }
    var prompt by rememberSaveable { mutableStateOf(WordImagePrompt.build("apple", "Fruit")) }
    var seed by rememberSaveable { mutableStateOf("42") }
    var steps by rememberSaveable { mutableIntStateOf(1) }
    var wifiOnly by rememberSaveable { mutableStateOf(true) }
    var research by rememberSaveable { mutableStateOf(false) }
    var modelsExpanded by rememberSaveable { mutableStateOf(false) }
    val model = ImageModels.get(selected)
    val context = LocalContext.current
    fun open(url: String) { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
    fun rebuild() { prompt = runCatching { WordImagePrompt.build(word, category) }.getOrDefault("") }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        uri?.let(controller::exportImageResults)
    }
    var imageToSave by rememberSaveable { mutableStateOf("") }
    val saveImage = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        uri?.let { controller.exportWordImage(it, imageToSave) }
    }
    Text("Word → picture", style = MaterialTheme.typography.headlineSmall)
    Text("Build a visual explanation from a word and its category. Download once, then generate offline on this phone.")
    Card {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Pixel research · 23 Oct 2026", style = MaterialTheme.typography.titleMedium)
            Text("Three offline models, from a small SD baseline to FLUX. ARM CPU inference can take minutes; compare the pictures and timings on your phone.", style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = { research = !research }) { Text(if (research) "Hide research sources" else "Research and sources") }
            if (research) {
                Text("SD-Turbo is the lightweight baseline; SDXL-Turbo targets more detail. FLUX.2 Klein 4B is the closest match to the FLUX goal, with higher memory and latency risk.", style = MaterialTheme.typography.bodySmall)
                Text("Pixel 10 Pro has Tensor G5 / 16 GB RAM; Pixel 11 has Tensor G6 / 12 GB RAM. Published desktop and Snapdragon NPU speeds do not predict Pixel speed. First run includes model loading; the engine is released after each image.", style = MaterialTheme.typography.bodySmall)
                Text("No Pixel 11 timing or FLUX quality parity is assumed. The results below are measurements from this device.", style = MaterialTheme.typography.bodySmall)
                Text("Pixel 10 Pro Fold / Tensor G5 CPU trial, apple at 512 px: SD-Turbo 69.8 s, SDXL-Turbo 143.8 s, FLUX Klein 546.8 s. FLUX gave the clearest single-subject picture in this one prompt/seed trial. Pixel 11 was not tested.", style = MaterialTheme.typography.bodySmall)
                Text("Google MobileDiffusion reports 512 px in ~0.5 s on premium mobile devices, but a reusable public checkpoint was not established in this research. Qualcomm demos use Snapdragon-specific acceleration. Neither is a drop-in Tensor implementation.", style = MaterialTheme.typography.bodySmall)
                listOf(
                    "Pixel 10 Pro specifications" to "https://store.google.com/product/pixel_10_pro_specs",
                    "Pixel 11 specifications" to "https://store.google.com/product/pixel_11_specs",
                    "Google MobileDiffusion research" to "https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/",
                    "Android runtime and supported models" to "https://github.com/leejet/stable-diffusion.cpp",
                ).forEach { (title, url) -> TextButton(onClick = { open(url) }) { Text(title) } }
            }
        }
    }
    Text("1 · Choose a model", style = MaterialTheme.typography.titleLarge)
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ImageModels.all.forEach { candidate ->
            FilterChip(selected = selected == candidate.id, onClick = { selected = candidate.id; steps = candidate.steps },
                enabled = !state.busy, label = { Text(candidate.name) })
        }
    }
    if (state.images.ready.isNotEmpty()) TextButton(onClick = { modelsExpanded = !modelsExpanded }) {
        Text(if (modelsExpanded) "Hide model details" else "Downloads and model details")
    }
    if (modelsExpanded || state.images.ready.isEmpty() || state.images.transfers.isNotEmpty()) ImageModels.all.forEach { candidate ->
        val installed = candidate.id in state.images.ready
        val transfers = state.images.transfers.filter { it.modelId == candidate.id }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row {
                    RadioButton(selected = selected == candidate.id, enabled = !state.busy,
                        onClick = { selected = candidate.id; steps = candidate.steps })
                    Column(Modifier.weight(1f)) {
                        Text(candidate.name, style = MaterialTheme.typography.titleMedium)
                        Text(candidate.role, style = MaterialTheme.typography.bodySmall)
                        Text("${decimal(candidate.bytes / 1e9)} GB · ${candidate.files.size} file(s) · 512 × 512 · CPU", style = MaterialTheme.typography.bodySmall)
                    }
                }
                Text(candidate.description, style = MaterialTheme.typography.bodySmall)
                Text(candidate.license, style = MaterialTheme.typography.labelSmall)
                TextButton(onClick = { open(candidate.source) }) { Text("Model card and license") }
                if (research) candidate.files.forEach { spec ->
                    TextButton(onClick = { open("https://huggingface.co/${spec.repository}/blob/${spec.revision}/${spec.remotePath}") }) {
                        Text("Artifact: ${spec.filename}", style = MaterialTheme.typography.labelSmall)
                    }
                }
                if (installed) Text("Ready · SHA-256 verified", color = MaterialTheme.colorScheme.primary)
                transfers.forEach { transfer ->
                    Text("${transfer.filename}: ${transfer.phase}", style = MaterialTheme.typography.bodySmall)
                    if (transfer.active) {
                        LinearProgressIndicator(progress = { (transfer.bytes.toFloat() / transfer.total).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                        Text("${decimal(transfer.bytes / 1e9)} / ${decimal(transfer.total / 1e9)} GB", style = MaterialTheme.typography.labelSmall)
                    }
                    transfer.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                }
                if (transfers.isNotEmpty()) {
                    OutlinedButton(onClick = { controller.cancelImageDownload(candidate.id) }, enabled = !state.busy) { Text("Cancel / clear transfer") }
                }
                if (!installed && transfers.none { it.active }) {
                    Button(onClick = { controller.downloadImageModel(candidate.id, wifiOnly) }, enabled = !state.unavailable) {
                        Text(if (transfers.isEmpty()) "Download ${candidate.name}" else "Retry download")
                    }
                }
            }
        }
    }
    if (modelsExpanded || state.images.ready.isEmpty() || state.images.transfers.isNotEmpty()) Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Switch(checked = wifiOnly, onCheckedChange = { wifiOnly = it })
        Text("Wi-Fi only for new downloads", Modifier.padding(top = 12.dp))
    }
    if (modelsExpanded || state.images.ready.isEmpty()) Text("Downloads run through Android and recover after restart. Files are checked before use. FLUX needs all three files. Other model operations wait for downloads to finish.", style = MaterialTheme.typography.bodySmall)
    Text("2 · Describe the concept", style = MaterialTheme.typography.titleLarge)
    OutlinedTextField(word, onValueChange = { word = it.take(80); rebuild() }, label = { Text("Word") },
        enabled = !state.busy, singleLine = true, modifier = Modifier.fillMaxWidth())
    OutlinedTextField(category, onValueChange = { category = it.take(80); rebuild() }, label = { Text("Category") },
        placeholder = { Text("Fruit, Actions, Feelings, Colors, Shapes…") },
        enabled = !state.busy, singleLine = true, modifier = Modifier.fillMaxWidth())
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf("apple" to "Fruit", "jump" to "Actions", "happy" to "Feelings", "red" to "Colors").forEach { (w, c) ->
            SuggestionChip(onClick = { word = w; category = c; rebuild() }, label = { Text("$w · $c") }, enabled = !state.busy)
        }
    }
    OutlinedTextField(prompt, { prompt = it.take(2000) }, label = { Text("Image prompt · editable") },
        enabled = !state.busy, minLines = 4, modifier = Modifier.fillMaxWidth())
    TextButton(onClick = { rebuild() }, enabled = !state.busy) { Text("Rebuild prompt from word and category") }
    Text("Category guides the scene: an action is performed, an emotion is expressed, and an object is shown clearly. Review ambiguous words and generated pictures before showing children.", style = MaterialTheme.typography.bodySmall)
    OutlinedTextField(seed, { seed = it.take(19) }, label = { Text("Seed · use the same seed for comparisons") }, singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), enabled = !state.busy, modifier = Modifier.fillMaxWidth())
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(1, 2, 4).forEach { count ->
            FilterChip(selected = steps == count, onClick = { steps = count }, label = { Text("$count step${if (count == 1) "" else "s"}") },
                enabled = !state.busy && (!model.flux || count == 4))
        }
    }
    val seedValue = seed.toLongOrNull()?.takeIf { it >= 0 }
    if (seedValue == null) Text("Enter a non-negative whole-number seed.", color = MaterialTheme.colorScheme.error)
    val ready = selected in state.images.ready
    if (!ready) Text("Download ${model.name} above to enable image generation.", style = MaterialTheme.typography.bodySmall)
    Button(onClick = { controller.generateWordImage(selected, word, category, prompt, requireNotNull(seedValue), steps) },
        enabled = !state.unavailable && ready && word.isNotBlank() && category.isNotBlank() && prompt.isNotBlank() && seedValue != null,
        modifier = Modifier.fillMaxWidth()) { Text("Generate with ${model.name}") }
    if (state.imageGenerating) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
        Text("${state.images.step} / ${state.images.totalSteps} denoising steps. Loading, text encoding and image decoding also take time.", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = controller::stop, enabled = !state.stopRequested) { Text(if (state.stopRequested) "Stopping…" else "Stop image generation") }
    }
    if (state.images.results.isNotEmpty()) {
        Text("3 · Review and compare", style = MaterialTheme.typography.titleLarge)
        Text("Measured here: total includes native model load and PNG creation. Native time includes load, text encoding, denoising and decode. Peak RSS is the process lifetime high-water mark, not per-model memory.", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = { export.launch("word-image-results.json") }, enabled = !state.unavailable) { Text("Export measurements JSON") }
        state.images.results.take(6).forEach { result ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("${result.word} · ${result.category}", style = MaterialTheme.typography.titleLarge)
                    Text("${result.modelName} · seed ${result.seed} · ${result.steps} steps", style = MaterialTheme.typography.bodySmall)
                    val bitmap by produceState<android.graphics.Bitmap?>(null, result.imagePath) {
                        value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(result.imagePath) }
                    }
                    bitmap?.let { Image(it.asImageBitmap(), contentDescription = "${result.modelName}: ${result.word} in ${result.category}",
                        modifier = Modifier.fillMaxWidth().aspectRatio(1f)) }
                    Text("Total ${decimal(result.totalMs / 1000.0)} s · native ${decimal(result.nativeMs / 1000.0)} s\nProcess peak ${decimal(result.peakRssKb / 1048576.0)} GiB · ${result.backend}")
                    if (result.denoiseMs > 0) Text("Load + encode ${decimal(result.loadAndEncodeMs / 1000.0)} s · denoise ${decimal(result.denoiseMs / 1000.0)} s · decode + save ${decimal(result.decodeAndSaveMs / 1000.0)} s", style = MaterialTheme.typography.bodySmall)
                    SelectionContainer { Text(result.prompt, style = MaterialTheme.typography.bodySmall) }
                    Text(result.device, style = MaterialTheme.typography.labelSmall)
                    OutlinedButton(onClick = { imageToSave = result.imagePath; saveImage.launch("${result.modelId}-${result.createdAtEpochMs}.png") },
                        enabled = !state.unavailable) { Text("Save PNG") }
                }
            }
        }
    }
}

private fun decimal(value: Double) = String.format(Locale.US, "%.2f", value)
