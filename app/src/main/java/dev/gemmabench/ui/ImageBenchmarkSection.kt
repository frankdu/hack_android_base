package dev.gemmabench.ui

import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.imagegen.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.DateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ImageBenchmarkSection(state: LabState, controller: LabController, selectedModelId: String) {
    val benchmark = state.imageBenchmark
    var models by rememberSaveable(selectedModelId) { mutableStateOf(setOf(selectedModelId)) }
    var methods by rememberSaveable { mutableStateOf(setOf(ImageMethod.CPU.name, ImageMethod.GPU.name)) }
    var timeout by rememberSaveable { mutableIntStateOf(20) }
    var inputsExpanded by rememberSaveable { mutableStateOf(false) }
    var viewedId by rememberSaveable { mutableStateOf("") }
    var historyLimit by rememberSaveable { mutableIntStateOf(10) }
    var exportId by rememberSaveable { mutableStateOf("") }
    var savePath by rememberSaveable { mutableStateOf("") }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if (uri != null && exportId.isNotEmpty()) controller.exportImageBenchmark(uri, exportId)
    }
    val saveImage = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        if (uri != null && savePath.isNotEmpty()) controller.exportWordImage(uri, savePath)
    }
    HorizontalDivider()
    Text("30-word generation benchmark", style = MaterialTheme.typography.headlineSmall)
    Text("One click runs 30 fixed word prompts for every selected model and method. The same word uses the same prompt and seed across comparisons.")
    Text("Speed includes worker startup, model loading, encoding, denoising, decoding and PNG saving. A fresh engine is used for every image; no warm-up is excluded. Resolution: 512 × 512. Average and mean are the same statistic; median is shown too.", style = MaterialTheme.typography.bodySmall)
    Text("Models", style = MaterialTheme.typography.titleMedium)
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ImageModels.all.forEach { model ->
            FilterChip(selected = model.id in models, onClick = { models = if (model.id in models) models - model.id else models + model.id },
                enabled = !state.busy && model.id in state.images.ready, label = { Text("${model.name} · ${model.steps} step${if (model.steps == 1) "" else "s"}") })
        }
    }
    if (models.any { it !in state.images.ready }) Text("Download the selected model above to enable the benchmark.", style = MaterialTheme.typography.bodySmall)
    Text("Generation methods", style = MaterialTheme.typography.titleMedium)
    benchmark.methods.forEach { support ->
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Checkbox(checked = support.available && support.method.name in methods,
                onCheckedChange = { checked -> methods = if (checked) methods + support.method.name else methods - support.method.name },
                enabled = support.available && !state.busy)
            Column(Modifier.weight(1f).padding(top = 8.dp)) {
                Text("${support.method.label} · ${if (support.available) "available" else "unavailable"}", style = MaterialTheme.typography.titleSmall)
                Text(support.method.detail, style = MaterialTheme.typography.bodySmall)
                Text(support.reason, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
    Text("A GPU failure skips the remaining GPU jobs in that run; CPU jobs can continue. Failed/skipped jobs have no speed or quality score.", style = MaterialTheme.typography.bodySmall)
    Text("Per-image timeout", style = MaterialTheme.typography.titleSmall)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(5, 10, 20).forEach { minutes -> FilterChip(selected = timeout == minutes,
            onClick = { timeout = minutes }, enabled = !state.busy, label = { Text("$minutes min") }) }
    }
    TextButton(onClick = { inputsExpanded = !inputsExpanded }) { Text(if (inputsExpanded) "Hide fixed prompts" else "Preview the 30 words and prompts") }
    if (inputsExpanded) ImageBenchmarkSuite.prompts.forEachIndexed { index, input ->
        Text("${index + 1}. ${input.word} · ${input.category} · seed ${input.seed}", style = MaterialTheme.typography.titleSmall)
        SelectionContainer { Text(input.prompt, style = MaterialTheme.typography.bodySmall) }
    }
    val runnableMethods = benchmark.methods.filter { it.available && it.method.name in methods }.map { it.method }.toSet()
    val jobs = models.size * runnableMethods.size * ImageBenchmarkSuite.prompts.size
    Text("$jobs images planned · 30 words × ${models.size} model(s) × ${runnableMethods.size} method(s). This can take hours. Keep Research Lab foreground; leaving stops the run and preserves completed images for resuming.", style = MaterialTheme.typography.bodySmall)
    Button(onClick = { viewedId = ""; controller.startImageBenchmark(models, runnableMethods, timeout) },
        enabled = !state.unavailable && models.isNotEmpty() && models.all { it in state.images.ready } && runnableMethods.isNotEmpty(),
        modifier = Modifier.fillMaxWidth()) { Text("Run 30-word benchmark") }
    if (state.imageBenchmarkRunning) {
        val active = benchmark.runs.firstOrNull { it.id == benchmark.activeRunId }
        val finished = active?.samples?.count { it.status in listOf(ImageSampleStatus.SUCCESS, ImageSampleStatus.FAILED, ImageSampleStatus.SKIPPED) } ?: 0
        LinearProgressIndicator(progress = { if (active == null || active.samples.isEmpty()) 0f else finished.toFloat() / active.samples.size }, modifier = Modifier.fillMaxWidth())
        Text("$finished / ${active?.samples?.size ?: jobs} jobs finished · current image step ${benchmark.step} / ${benchmark.steps}", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = controller::stop, enabled = !state.stopRequested) { Text(if (state.stopRequested) "Stopping worker…" else "Stop and keep results") }
    }
    val latest = benchmark.runs.firstOrNull()
    val displayed = benchmark.runs.firstOrNull { it.id == viewedId } ?: latest
    if (displayed == null) Text("No image benchmark runs yet.", style = MaterialTheme.typography.bodySmall)
    else {
        Text(if (displayed.id == latest?.id) "Most recent run" else "Historical run", style = MaterialTheme.typography.titleLarge)
        if (displayed.id != latest?.id) TextButton(onClick = { viewedId = "" }) { Text("View most recent run") }
        Text("${date(displayed.startedAtEpochMs)} · ${displayed.status.name.lowercase().replaceFirstChar { it.uppercase() }}", style = MaterialTheme.typography.titleMedium)
        Text("${displayed.successful} generated · ${displayed.failed} failed · ${displayed.pending} pending · ${displayed.samples.count { it.status == ImageSampleStatus.SKIPPED }} skipped", style = MaterialTheme.typography.bodySmall)
        Text(displayed.device, style = MaterialTheme.typography.labelSmall)
        Text("${displayed.suiteVersion} · timeout ${displayed.timeoutMinutes} min/image", style = MaterialTheme.typography.labelSmall)
        displayed.skippedMethods.forEach { Text("${it.method.label} unavailable: ${it.reason}", style = MaterialTheme.typography.bodySmall) }
        Text("Scroll the statistics table sideways for all timing and quality columns.", style = MaterialTheme.typography.labelSmall)
        ImageBenchmarkStatistics(displayed)
        Text("Quality is a human review score: 1 = wrong/unrecognizable; 3 = recognizable with problems; 5 = accurate, clear teaching picture. Review concept accuracy, clutter, anatomy and unwanted text. Unrated images are excluded from quality averages.", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { exportId = displayed.id; export.launch("image-benchmark-${displayed.id}.json") }, enabled = !state.unavailable) { Text("Export run JSON") }
            if (displayed.resumable) OutlinedButton(onClick = { controller.resumeImageBenchmark(displayed.id) }, enabled = !state.unavailable) { Text("Resume pending") }
        }
        val words = displayed.samples.map { it.input }.distinctBy { it.word to it.category }
        var wordIndex by rememberSaveable(displayed.id) { mutableIntStateOf(0) }
        wordIndex = wordIndex.coerceIn(0, words.lastIndex.coerceAtLeast(0))
        if (words.isNotEmpty()) {
            Text("Generated images · word ${wordIndex + 1} / ${words.size}", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { wordIndex-- }, enabled = wordIndex > 0) { Text("Previous word") }
                OutlinedButton(onClick = { wordIndex++ }, enabled = wordIndex < words.lastIndex) { Text("Next word") }
            }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                words.forEachIndexed { index, input -> FilterChip(selected = wordIndex == index, onClick = { wordIndex = index }, label = { Text(input.word) }) }
            }
            val input = words[wordIndex]
            Text("${input.word} · ${input.category} · seed ${input.seed}", style = MaterialTheme.typography.titleMedium)
            SelectionContainer { Text(input.prompt, style = MaterialTheme.typography.bodySmall) }
            displayed.samples.filter { it.input.word == input.word && it.input.category == input.category }.forEach { sample ->
                ImageBenchmarkSampleCard(sample, !state.busy,
                    onRate = { controller.rateImageBenchmark(displayed.id, sample.id, it) },
                    onSave = { savePath = it; saveImage.launch("${sample.modelId}-${sample.method.name}-${sample.input.word}.png") })
            }
        }
    }
    Text("Historical runs", style = MaterialTheme.typography.titleLarge)
    Text("Runs and completed images are saved on this device after every sample. Interrupted runs retain pending jobs and can be resumed.", style = MaterialTheme.typography.bodySmall)
    benchmark.runs.take(historyLimit).forEach { run ->
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("${date(run.startedAtEpochMs)}${if (run.id == latest?.id) " · latest" else ""}", style = MaterialTheme.typography.titleSmall)
                Text("${run.status} · ${run.successful}/${run.samples.size} generated · ${run.failed} failed · ${run.pending} pending", style = MaterialTheme.typography.bodySmall)
                run.groups().forEach { group ->
                    Text("${group.modelName} / ${group.method.label}: ${group.speed?.let { "mean ${seconds(it.meanMs)} s · min ${seconds(it.minMs.toDouble())} · max ${seconds(it.maxMs.toDouble())}" } ?: "No successful timings"}" +
                        (group.meanQuality?.let { " · quality ${String.format(Locale.US, "%.1f", it)}/5 (${group.ratings.size} reviewed)" } ?: " · quality unrated"),
                        style = MaterialTheme.typography.bodySmall)
                }
                TextButton(onClick = { viewedId = run.id }) { Text("View images and statistics") }
            }
        }
    }
    if (benchmark.runs.size > historyLimit) TextButton(onClick = { historyLimit += 10 }) { Text("Show more history") }
}

@Composable
private fun ImageBenchmarkStatistics(run: ImageBenchmarkRun) {
    Column(Modifier.horizontalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row {
            listOf("Model / method" to 210, "Success" to 70, "Avg (mean)" to 105, "Median" to 100, "Min" to 90, "Max" to 90, "Quality" to 140).forEach { (title, width) ->
                Text(title, Modifier.width(width.dp), style = MaterialTheme.typography.labelMedium)
            }
        }
        run.groups().forEach { group ->
            val speed = group.speed
            Row {
                Text("${group.modelName}\n${group.method.label}", Modifier.width(210.dp), style = MaterialTheme.typography.bodySmall)
                Text("${speed?.count ?: 0}/${group.samples.size}", Modifier.width(70.dp), style = MaterialTheme.typography.bodySmall)
                listOf(speed?.meanMs to 105, speed?.medianMs to 100, speed?.minMs?.toDouble() to 90, speed?.maxMs?.toDouble() to 90).forEach { (value, width) ->
                    Text(value?.let { "${seconds(it)} s" } ?: "—", Modifier.width(width.dp), style = MaterialTheme.typography.bodySmall)
                }
                Text(group.meanQuality?.let { "${String.format(Locale.US, "%.1f", it)}/5\n${group.ratings.size} reviewed" } ?: "Unrated", Modifier.width(140.dp), style = MaterialTheme.typography.bodySmall)
            }
            val failed = group.samples.count { it.status == ImageSampleStatus.FAILED }
            val skipped = group.samples.count { it.status == ImageSampleStatus.SKIPPED }
            if (failed + skipped > 0) Text("${group.modelName} / ${group.method.label}: $failed failed, $skipped skipped", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun ImageBenchmarkSampleCard(sample: ImageBenchmarkSample, enabled: Boolean, onRate: (Int?) -> Unit, onSave: (String) -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("${sample.modelName} · ${sample.method.label}", style = MaterialTheme.typography.titleMedium)
            Text("${sample.status} · ${sample.steps} steps · seed ${sample.input.seed}", style = MaterialTheme.typography.bodySmall)
            sample.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            sample.result?.let { result ->
                val bitmap by produceState<android.graphics.Bitmap?>(null, result.imagePath) {
                    value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(result.imagePath) }
                }
                bitmap?.let { Image(it.asImageBitmap(), contentDescription = "${sample.modelName} ${sample.method.label} benchmark: ${sample.input.word}", modifier = Modifier.fillMaxWidth().aspectRatio(1f)) }
                    ?: Text("Image file is unavailable.", style = MaterialTheme.typography.bodySmall)
                Text("End-to-end ${sample.elapsedMs?.let { seconds(it.toDouble()) } ?: "—"} s · generation ${seconds(result.totalMs.toDouble())} s", style = MaterialTheme.typography.bodySmall)
                Text("Load/encode ${seconds(result.loadAndEncodeMs.toDouble())} s · denoise ${seconds(result.denoiseMs.toDouble())} s · decode/save ${seconds(result.decodeAndSaveMs.toDouble())} s", style = MaterialTheme.typography.bodySmall)
                Text("${result.backend} · 512 × 512 · process peak ${String.format(Locale.US, "%.2f", result.peakRssKb / 1048576.0)} GiB", style = MaterialTheme.typography.bodySmall)
                Text("Quality rating", style = MaterialTheme.typography.titleSmall)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    (1..5).forEach { rating -> FilterChip(selected = sample.quality == rating, onClick = { onRate(rating) }, enabled = enabled, label = { Text("$rating") }) }
                    if (sample.quality != null) TextButton(onClick = { onRate(null) }, enabled = enabled) { Text("Clear") }
                }
                OutlinedButton(onClick = { onSave(result.imagePath) }, enabled = enabled) { Text("Save PNG") }
            }
        }
    }
}
private fun seconds(value: Double) = String.format(Locale.US, "%.2f", value / 1000.0)
private fun date(value: Long) = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(value))
