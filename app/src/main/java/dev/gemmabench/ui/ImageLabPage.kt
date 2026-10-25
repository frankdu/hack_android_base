package dev.gemmabench.ui

import android.graphics.BitmapFactory
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
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.imagegen.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ImageLabPage(state: LabState, controller: LabController) {
    var selected by rememberSaveable { mutableStateOf("SD_TURBO_LITERT") }
    var compare by rememberSaveable { mutableStateOf(false) }
    var word by rememberSaveable { mutableStateOf("apple") }
    var category by rememberSaveable { mutableStateOf("Fruit") }
    var prompt by rememberSaveable { mutableStateOf(WordImagePrompt.build("apple", "Fruit")) }
    var seed by rememberSaveable { mutableStateOf("42") }
    var methodName by rememberSaveable { mutableStateOf(ImageMethod.CPU.name) }
    var wifiOnly by rememberSaveable { mutableStateOf(true) }
    var details by rememberSaveable { mutableStateOf(false) }
    var vocabulary by rememberSaveable { mutableStateOf(false) }
    val model = ImageModels.liteRt.firstOrNull { it.id == selected } ?: ImageModels.get("SD_TURBO_LITERT")
    val method = ImageMethod.valueOf(methodName)
    val methods = state.imageBenchmark.methods
    LaunchedEffect(model.id) { selected = model.id }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { it?.let(controller::exportImageResults) }
    var imageToSave by rememberSaveable { mutableStateOf("") }
    val saveImage = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        uri?.let { controller.exportWordImage(it, imageToSave) }
    }
    Text("LiteRT image generation", style = MaterialTheme.typography.headlineSmall)
    Text("Test FLUX, Bonsai, SD-Turbo and SDXL-Turbo offline, or compare them with the same prompts and seeds.")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(selected = !compare, onClick = { compare = false }, label = { Text("Generate image") })
        FilterChip(selected = compare, onClick = { compare = true }, label = { Text("Compare models") })
    }
    if (compare) {
        ImageBenchmarkSection(state, controller, model.id)
        return
    }
    Text("1 · Model", style = MaterialTheme.typography.titleLarge)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ImageModels.liteRt.forEach { candidate ->
            FilterChip(modifier = Modifier.testTag("image-model-${candidate.id}"), selected = candidate.id == model.id, onClick = { selected = candidate.id; details = false },
                enabled = !state.busy, label = { Text(candidate.name.removeSuffix(" · LiteRT")) })
        }
    }
    Text("${model.width} × ${model.height} · ${model.steps} step${if (model.steps == 1) "" else "s"} · ${decimal(model.bytes / 1e9)} GB ${if (model.gpuUnavailableReason == null) "shared CPU/GPU" else "CPU"} download",
        style = MaterialTheme.typography.bodySmall)
    Text("Backend", style = MaterialTheme.typography.titleMedium)
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ImageMethod.entries.forEach { candidate ->
            val support = methods.firstOrNull { it.method == candidate }
            FilterChip(selected = method == candidate, onClick = { methodName = candidate.name },
                enabled = !state.busy && support?.compatible(model.id) == true,
                label = { Text("${candidate.label}${if (support?.supports(model.id) == true) " · ready" else ""}") })
        }
    }
    ImageMethod.entries.forEach { candidate ->
        val support = methods.firstOrNull { it.method == candidate }
        Text("${candidate.label}: ${support?.reasonFor(model.id) ?: "Checking runtime…"}", style = MaterialTheme.typography.bodySmall)
    }
    val installed = model.id in state.images.ready
    val transfers = state.images.transfers.filter { it.modelId == model.id }
    if (!installed || transfers.isNotEmpty() || details) {
        ImageModelCard(model, state, controller, wifiOnly, showArtifacts = details)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Switch(checked = wifiOnly, onCheckedChange = { wifiOnly = it })
            Text("Wi-Fi only for new downloads", Modifier.padding(top = 12.dp))
        }
    }
    TextButton(onClick = { details = !details }) { Text(if (details) "Hide model details" else "Model source and download details") }
    if (details) OutlinedButton(onClick = controller::verifyImageTpuBundles, enabled = !state.unavailable) { Text("Verify staged TPU bundles") }
    Text("2 · Prompt", style = MaterialTheme.typography.titleLarge)
    OutlinedTextField(prompt, { prompt = it.take(2000) }, label = { Text("Describe the picture") },
        enabled = !state.busy, minLines = 4, modifier = Modifier.fillMaxWidth())
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(
            Triple("apple", "Fruit", "A red apple on a wooden table, studio lighting."),
            Triple("teapot", "Objects", "A blue ceramic teapot on a wooden table, morning light."),
            Triple("cat", "Animals", "A friendly orange cat sitting in a sunny garden."),
            Triple("happy", "Feelings", WordImagePrompt.build("happy", "Feelings")),
        ).forEach { (w, c, p) -> SuggestionChip(onClick = { word = w; category = c; prompt = p },
            enabled = !state.busy, label = { Text(w.replaceFirstChar { it.uppercase() }) }) }
    }
    TextButton(onClick = { vocabulary = !vocabulary }) { Text(if (vocabulary) "Hide vocabulary helpers" else "Vocabulary word and category") }
    if (vocabulary) {
        OutlinedTextField(word, { word = it.take(80) }, label = { Text("Word") }, singleLine = true,
            enabled = !state.busy, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(category, { category = it.take(80) }, label = { Text("Category") }, singleLine = true,
            enabled = !state.busy, modifier = Modifier.fillMaxWidth())
        TextButton(onClick = { prompt = WordImagePrompt.build(word, category) },
            enabled = !state.busy && word.isNotBlank() && category.isNotBlank()) { Text("Build teaching prompt") }
    }
    OutlinedTextField(seed, { seed = it.take(19) }, label = { Text("Seed · repeat within the same model/backend") }, singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), enabled = !state.busy, modifier = Modifier.fillMaxWidth())
    val seedValue = seed.toLongOrNull()?.takeIf { it >= 0 }
    if (seedValue == null) Text("Enter a non-negative whole-number seed.", color = MaterialTheme.colorScheme.error)
    val ready = methods.any { it.method == method && it.supports(model.id) }
    Button(onClick = { controller.generateWordImage(model.id, word, category, prompt, requireNotNull(seedValue), model.steps, method) },
        enabled = !state.unavailable && ready && prompt.isNotBlank() && word.isNotBlank() && category.isNotBlank() && seedValue != null,
        modifier = Modifier.fillMaxWidth()) { Text("Generate · ${model.name.removeSuffix(" · LiteRT")} · ${method.label}") }
    if (state.imageGenerating) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
        Text("${state.images.step} / ${state.images.totalSteps} steps · loading, text encoding and image decode also take time.", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = controller::stop, enabled = !state.stopRequested) { Text(if (state.stopRequested) "Stopping…" else "Stop generation") }
    }
    if (state.images.results.isNotEmpty()) {
        Text("Recent images", style = MaterialTheme.typography.titleLarge)
        Text("Timings include loading and PNG creation. Process peak memory is the lifetime high-water mark.", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = { export.launch("word-image-results.json") }, enabled = !state.unavailable) { Text("Export measurements JSON") }
        state.images.results.take(6).forEach { result ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(result.modelName, style = MaterialTheme.typography.titleMedium)
                    Text("${result.backend} · ${result.width} × ${result.height} · seed ${result.seed} · ${result.steps} steps", style = MaterialTheme.typography.bodySmall)
                    val bitmap by produceState<android.graphics.Bitmap?>(null, result.imagePath) {
                        value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(result.imagePath) }
                    }
                    bitmap?.let { Image(it.asImageBitmap(), contentDescription = "${result.modelName}: ${result.prompt}",
                        modifier = Modifier.fillMaxWidth().aspectRatio(result.width.toFloat() / result.height)) }
                    Text("Total ${decimal(result.totalMs / 1000.0)} s · process peak ${decimal(result.peakRssKb / 1048576.0)} GiB")
                    Text("Encode ${decimal(result.loadAndEncodeMs / 1000.0)} s · denoise ${decimal(result.denoiseMs / 1000.0)} s · decode/save ${decimal(result.decodeAndSaveMs / 1000.0)} s", style = MaterialTheme.typography.bodySmall)
                    SelectionContainer { Text(result.prompt, style = MaterialTheme.typography.bodySmall) }
                    OutlinedButton(onClick = { imageToSave = result.imagePath; saveImage.launch("${result.modelId}-${result.createdAtEpochMs}.png") },
                        enabled = !state.unavailable) { Text("Save PNG") }
                }
            }
        }
    }
}

private fun decimal(value: Double) = String.format(Locale.US, "%.2f", value)
