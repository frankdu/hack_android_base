package dev.gemmabench.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import dev.gemmabench.LabController
import dev.gemmabench.LabState
import dev.gemmabench.benchmark.*
import dev.gemmabench.download.*
import java.util.Locale

private val Ink = Color(0xFF153B32)
private val Mint = Color(0xFFBFEBD1)
private val Paper = Color(0xFFF6F7F1)
private val colors = lightColorScheme(
    primary = Color(0xFF176B55), onPrimary = Color.White,
    primaryContainer = Mint, onPrimaryContainer = Ink,
    background = Paper, surface = Color.White, surfaceContainer = Color(0xFFEBF0E7),
    onBackground = Color(0xFF1E2923), onSurface = Color(0xFF1E2923),
)

@Composable
fun LabApp(state: LabState, controller: LabController) {
    MaterialTheme(colorScheme = colors) {
        var tab by rememberSaveable { mutableIntStateOf(0) }
        val tabs = listOf("Models", "Benchmark", "Playground", "Results")
        Scaffold(
            containerColor = Paper,
            bottomBar = {
                NavigationBar(containerColor = Color.White) {
                    tabs.forEachIndexed { index, title ->
                        NavigationBarItem(
                            selected = tab == index, onClick = { tab = index },
                            icon = { Text(listOf("◈", "▥", "✳", "≡")[index], style = MaterialTheme.typography.titleLarge) },
                            label = { Text(title) },
                        )
                    }
                }
            },
        ) { insets ->
            Column(
                Modifier.padding(insets).fillMaxSize().verticalScroll(rememberScrollState())
                    .padding(horizontal = 20.dp, vertical = 16.dp).widthIn(max = 840.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column {
                        Text("GEMMA / DEVICE LAB", color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.labelMedium)
                        Text(tabs[tab], style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.SemiBold)
                    }
                    Surface(color = Mint, shape = MaterialTheme.shapes.extraLarge) {
                        Text("LOCAL INFERENCE", Modifier.padding(horizontal = 12.dp, vertical = 8.dp), style = MaterialTheme.typography.labelSmall, color = Ink)
                    }
                }
                state.device?.let {
                    Text("${it.model} · ${it.soc} · Android ${it.androidRelease}\n${it.totalRamMb / 1024.0 format 1} GiB RAM · LiteRT-LM ${it.runtimeVersion}", style = MaterialTheme.typography.bodySmall)
                }
                StatusCard(state, controller)
                when (tab) {
                    0 -> ModelsPage(state, controller)
                    1 -> RunPage(state, controller, playground = false)
                    2 -> RunPage(state, controller, playground = true)
                    3 -> ResultsPage(state, controller)
                }
                Text("Prototype 0.1 · Native token counts · No automatic backend fallback", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun StatusCard(state: LabState, controller: LabController) {
    Card(colors = CardDefaults.cardColors(containerColor = Ink)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(if (state.downloadsActive && !state.busy) "Model downloads in progress · you can leave the app." else state.progress,
                color = Color.White, style = MaterialTheme.typography.bodyMedium)
            if (state.busy) LinearProgressIndicator(Modifier.fillMaxWidth(), color = Mint, trackColor = Color(0xFF39584D))
            if (state.runningInference) {
                OutlinedButton(onClick = controller::stop, enabled = !state.stopRequested,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Mint)) {
                    Text(if (state.stopRequested) "Stop requested" else "Stop run")
                }
            }
        }
    }
    state.error?.let {
        Surface(color = MaterialTheme.colorScheme.errorContainer, shape = MaterialTheme.shapes.medium) {
            SelectionContainer { Text(it, Modifier.padding(16.dp), color = MaterialTheme.colorScheme.onErrorContainer) }
        }
    }
}

@Composable
private fun ModelsPage(state: LabState, controller: LabController) {
    var pending by rememberSaveable { mutableStateOf(ModelSlot.E2B.name) }
    var advanced by rememberSaveable { mutableStateOf(false) }
    var wifiOnly by rememberSaveable { mutableStateOf(true) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { controller.importModel(ModelSlot.valueOf(pending), it) }
    }
    Text("Download a model and start benchmarking. Models come directly from Hugging Face; your prompts and responses stay on this device.")
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
        Column(Modifier.weight(1f)) {
            Text("Wi-Fi only", style = MaterialTheme.typography.titleSmall)
            Text(if (wifiOnly) "Downloads wait for Wi-Fi" else "Mobile data allowed for new downloads", style = MaterialTheme.typography.bodySmall)
        }
        Switch(checked = wifiOnly, onCheckedChange = { wifiOnly = it })
    }
    listOf(ModelSlot.E2B, ModelSlot.E4B).forEach { slot ->
        ModelCard(slot, state, controller, wifiOnly, advanced) { pending = slot.name; picker.launch(arrayOf("*/*")) }
    }
    Text("Downloads continue when you leave the app. Android retries interrupted transfers; return here to see progress. Completed files are checked before becoming ready.", style = MaterialTheme.typography.bodySmall)
    TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Hide advanced options" else "Advanced · manual import and TPU") }
    if (advanced) {
        OutlinedButton(onClick = controller::importStaged, enabled = !state.unavailable, modifier = Modifier.fillMaxWidth()) { Text("Import files staged by ADB") }
        Text("Manual imports make a private copy and use the slot label you select. Downloaded official models are verified against a pinned SHA-256 checksum.", style = MaterialTheme.typography.bodySmall)
        Text(if (state.tensorLibraries) "Tensor dispatch library detected. Runtime, firmware and artifact versions still need to match." else "This build has CPU/GPU support. To enable TPU, bundle matching Tensor native libraries using the documented Gradle option.", style = MaterialTheme.typography.bodyMedium)
        listOf(ModelSlot.E2B_G5, ModelSlot.E2B_G6).forEach { slot ->
            ModelCard(slot, state, controller, wifiOnly, true) { pending = slot.name; picker.launch(arrayOf("*/*")) }
        }
        Text("E4B TPU is not offered: the current official Tensor model list includes E2B. See docs/BENCHMARKING.md.", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ModelCard(slot: ModelSlot, state: LabState, controller: LabController, wifiOnly: Boolean, advanced: Boolean, onImport: () -> Unit) {
    val model = state.models.firstOrNull { it.slot == slot }
    val transfer = state.downloads.firstOrNull { it.spec.slot == slot }
    val spec = ModelCatalog.models.firstOrNull { it.slot == slot }
    val context = LocalContext.current
    val family = if (slot == ModelSlot.E4B) "E4B" else "E2B"
    val repo = "https://huggingface.co/litert-community/gemma-4-$family-it-litert-lm"
    val url = when (slot) {
        ModelSlot.E2B_G5 -> "$repo/blob/main/gemma-4-E2B-it_Google_Tensor_G5.litertlm"
        ModelSlot.E2B_G6 -> "$repo/blob/main/gemma-4-E2B-it_Google_Tensor_G6.litertlm"
        else -> "$repo/tree/main"
    }
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(slot.label, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(if (slot.tpu) "Device-specific TPU artifact" else if (slot == ModelSlot.E2B) "Fast baseline · 2.59 GB" else "Higher capacity · 3.66 GB", style = MaterialTheme.typography.bodySmall)
            if (model != null) {
                Text(if (model.downloaded) "Ready · verified download" else "Ready · local model", color = MaterialTheme.colorScheme.primary)
            }
            if (transfer != null) {
                when (transfer.phase) {
                    TransferPhase.VERIFYING -> {
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                        Text(transfer.detail ?: "Checking downloaded model…", style = MaterialTheme.typography.bodySmall)
                    }
                    TransferPhase.FAILED -> {
                        Text(transfer.detail ?: "Download failed. Tap Retry.", color = MaterialTheme.colorScheme.error)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { controller.downloadModel(slot, wifiOnly) }, enabled = !state.busy) { Text("Retry download") }
                            TextButton(onClick = { controller.cancelDownload(transfer.id) }, enabled = !state.busy) { Text("Dismiss") }
                        }
                    }
                    else -> {
                        LinearProgressIndicator(progress = { transfer.fraction }, modifier = Modifier.fillMaxWidth())
                        Text("${(transfer.fraction * 100).toInt()}% · ${(transfer.receivedBytes / 1e9) format 2} / ${(transfer.spec.bytes / 1e9) format 2} GB", style = MaterialTheme.typography.bodyMedium)
                        transfer.detail?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                        OutlinedButton(onClick = { controller.cancelDownload(transfer.id) }, enabled = !state.busy && transfer.phase != TransferPhase.CANCELING) { Text("Cancel download") }
                    }
                }
            } else if (model == null && spec != null) {
                Button(onClick = { controller.downloadModel(slot, wifiOnly) }, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) { Text("Download ${if (slot == ModelSlot.E2B) "E2B" else "E4B"}") }
            }
            if (model != null && advanced) {
                model.metadata?.let { meta ->
                    Text("Backends: ${meta.textBackends.joinToString().ifBlank { "unspecified" }} · context ${meta.maxContextTokens.takeIf { it > 0 } ?: "unspecified"}${if (meta.dynamicContext) " (dynamic)" else ""}", style = MaterialTheme.typography.bodySmall)
                }
                SelectionContainer { Text("SHA-256 ${model.sha256}", style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace) }
            }
            if (advanced) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onImport, enabled = !state.unavailable) { Text(if (model == null) "Import file" else "Replace file") }
                    TextButton(onClick = { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }) { Text("Official files ↗") }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RunPage(state: LabState, controller: LabController, playground: Boolean) {
    var acceleratorName by rememberSaveable(playground) { mutableStateOf(Accelerator.CPU.name) }
    val accelerator = Accelerator.valueOf(acceleratorName)
    var selectedNames by rememberSaveable(playground) { mutableStateOf(listOf(ModelSlot.E2B.name, ModelSlot.E4B.name)) }
    var inputTokens by rememberSaveable(playground) { mutableStateOf("256") }
    var outputTokens by rememberSaveable(playground) { mutableStateOf(if (playground) "128" else "256") }
    var contextTokens by rememberSaveable(playground) { mutableStateOf("2048") }
    var repeats by rememberSaveable(playground) { mutableStateOf("3") }
    var warmups by rememberSaveable(playground) { mutableStateOf("1") }
    var cooldown by rememberSaveable(playground) { mutableStateOf("2") }
    var mtp by rememberSaveable(playground) { mutableStateOf(false) }
    var prompt by rememberSaveable(playground) { mutableStateOf(PromptPresets.explain) }
    var error by rememberSaveable(playground) { mutableStateOf<String?>(null) }
    var advanced by rememberSaveable(playground) { mutableStateOf(false) }

    Text(if (playground) "Try a real task with a warm engine and a fresh conversation for each request. Select both models to compare the same prompt." else "Run a repeatable token workload. The native harness pads or truncates input to the requested count; generated text is not a quality test.")
    Text("Accelerator", style = MaterialTheme.typography.titleMedium)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Accelerator.entries.forEach { backend ->
            FilterChip(
                selected = backend == accelerator,
                onClick = { acceleratorName = backend.name; if (backend != Accelerator.GPU) mtp = false },
                enabled = !state.unavailable && (backend != Accelerator.TPU || state.tensorLibraries),
                label = { Text(backend.name) },
            )
        }
    }
    if (!state.tensorLibraries) Text("TPU needs an optional build with Tensor libraries. CPU/GPU work with standard artifacts.", style = MaterialTheme.typography.bodySmall)
    if (state.downloadsActive) Text("Finish or cancel model downloads before running inference.", style = MaterialTheme.typography.bodySmall)
    Text("Models to run", style = MaterialTheme.typography.titleMedium)
    val slots = if (accelerator == Accelerator.TPU) listOf(ModelSlot.E2B_G5, ModelSlot.E2B_G6) else listOf(ModelSlot.E2B, ModelSlot.E4B)
    slots.forEach { slot ->
        val ready = state.models.any { it.slot == slot }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = slot.name in selectedNames, enabled = ready && !state.unavailable, onCheckedChange = { checked ->
                selectedNames = if (checked) selectedNames + slot.name else selectedNames - slot.name
            })
            Column {
                Text(slot.label)
                Text(if (ready) "Ready" else "Download from Models first", style = MaterialTheme.typography.labelSmall)
            }
        }
    }
    if (playground) {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("Explain" to PromptPresets.explain, "Extract JSON" to PromptPresets.extract, "Summarize" to PromptPresets.summarize).forEach { (label, value) ->
                AssistChip(onClick = { prompt = value }, enabled = !state.unavailable, label = { Text(label) })
            }
        }
        OutlinedTextField(value = prompt, onValueChange = { prompt = it }, label = { Text("Prompt") }, minLines = 5, maxLines = 12, enabled = !state.unavailable, modifier = Modifier.fillMaxWidth())
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            NumberField("Output limit", outputTokens, { outputTokens = it }, !state.unavailable, Modifier.weight(1f))
            NumberField("Context tokens", contextTokens, { contextTokens = it }, !state.unavailable, Modifier.weight(1f))
        }
    } else {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            AssistChip(onClick = { inputTokens = "256"; outputTokens = "128" }, enabled = !state.unavailable, label = { Text("Quick · 256 / 128") })
            AssistChip(onClick = { inputTokens = "1024"; outputTokens = "256" }, enabled = !state.unavailable, label = { Text("Reference · 1024 / 256") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            NumberField("Input tokens", inputTokens, { inputTokens = it }, !state.unavailable, Modifier.weight(1f))
            NumberField("Output tokens", outputTokens, { outputTokens = it }, !state.unavailable, Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            NumberField("Measured runs", repeats, { repeats = it }, !state.unavailable, Modifier.weight(1f))
            NumberField("Warm-up runs", warmups, { warmups = it }, !state.unavailable, Modifier.weight(1f))
        }
        Text("Every fixed-token invocation reloads its engine. Warm-ups prime runtime/OS caches and are excluded from summaries. This is not a retained-engine latency test.", style = MaterialTheme.typography.bodySmall)
    }
    TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Hide options" else "More options") }
    if (advanced) {
        if (!playground) NumberField("Cooldown seconds", cooldown, { cooldown = it }, !state.unavailable)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(checked = mtp, onCheckedChange = { mtp = it }, enabled = !state.unavailable && accelerator == Accelerator.GPU)
            Text("  Speculative decoding (GPU experiment)")
        }
        Text("Baseline: speculative decoding off. Prompt mode disables thinking and uses top-k 1, temperature 0, seed 0. Fixed-token mode uses the native harness defaults and its own derived context size. CPU thread count uses runtime defaults.", style = MaterialTheme.typography.bodySmall)
    }
    error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
    Button(
        enabled = !state.unavailable,
        modifier = Modifier.fillMaxWidth().height(52.dp),
        onClick = {
            try {
                val fixedInput = if (playground) 256 else inputTokens.toInt()
                val output = outputTokens.toInt()
                val settings = RunSettings(
                    accelerator = accelerator, workload = if (playground) Workload.PROMPT else Workload.FIXED_TOKENS,
                    prefillTokens = fixedInput, outputTokens = output,
                    contextTokens = if (playground) contextTokens.toInt() else maxOf(2048, fixedInput + output + 64),
                    repetitions = if (playground) 1 else repeats.toInt(), warmups = if (playground) 0 else warmups.toInt(),
                    cooldownSeconds = if (playground) 0 else cooldown.toInt(), speculativeDecoding = mtp,
                    prompt = prompt,
                )
                settings.validate()
                val selected = slots.filter { it.name in selectedNames && state.models.any { model -> model.slot == it } }.toSet()
                require(selected.isNotEmpty()) { "Download and select at least one model." }
                error = null
                controller.run(selected, settings)
            } catch (e: Exception) { error = if (e is NumberFormatException) "Enter whole numbers for each setting." else e.message }
        },
    ) { Text(if (playground) "Run prompt" else "Run benchmark") }
    if (playground) {
        OutlinedButton(onClick = controller::unload, enabled = !state.unavailable, modifier = Modifier.fillMaxWidth()) { Text("Unload engine") }
        Text(if (state.engineRetained) "Last prompt engine retained for the next request." else "The engine will load if no matching engine is retained.", style = MaterialTheme.typography.bodySmall)
        if (state.output.isNotBlank()) Card {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Response", style = MaterialTheme.typography.titleMedium)
                SelectionContainer { Text(state.output) }
            }
        }
    }
    state.current?.let { ResultCard(it) }
}

@Composable
private fun NumberField(label: String, value: String, onValueChange: (String) -> Unit, enabled: Boolean, modifier: Modifier = Modifier) {
    OutlinedTextField(value = value, onValueChange = onValueChange, label = { Text(label) }, enabled = enabled,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), singleLine = true, modifier = modifier)
}

@Composable
private fun ResultsPage(state: LabState, controller: LabController) {
    val json = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { it?.let { uri -> controller.export(uri, false) } }
    val csv = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { it?.let { uri -> controller.export(uri, true) } }
    Text("${state.results.size} saved result(s). Medians and p95 exclude warm-ups. Compare runs only when model artifact, workload, backend and settings match.")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = { json.launch("gemma-bench.json") }, enabled = !state.unavailable && state.results.isNotEmpty()) { Text("Export JSON") }
        OutlinedButton(onClick = { csv.launch("gemma-bench.csv") }, enabled = !state.unavailable && state.results.isNotEmpty()) { Text("Export CSV") }
    }
    Text("Exports include prompts, responses, model hashes and device build details. Results remain local until you choose a destination.", style = MaterialTheme.typography.bodySmall)
    if (state.results.isEmpty()) {
        Card { Text("No device measurements yet. Download a model, then run a benchmark or a prompt.", Modifier.padding(24.dp)) }
    }
    state.results.take(20).forEach { ResultCard(it) }
    if (state.results.size > 20) Text("Showing the latest 20. Exports include all saved results (up to 100).")
}

@Composable
private fun ResultCard(result: BenchmarkResult) {
    var expanded by rememberSaveable(result.id) { mutableStateOf(false) }
    val measured = result.measuredSamples()
    val decode = summarize(measured.map { it.metrics.decodeTokensPerSecond })
    val prefill = summarize(measured.map { it.metrics.prefillTokensPerSecond })
    val ttft = summarize(measured.map { it.metrics.ttftSeconds })
    val init = summarize(measured.map { it.metrics.initSeconds })
    Card {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(result.model.slot.label, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text("${result.settings.accelerator} · ${if (result.settings.workload == Workload.PROMPT) "Prompt" else "Fixed tokens"} · ${result.status}", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Metric("Decode", decode?.median, "tok/s", Modifier.weight(1f))
                Metric("TTFT", ttft?.median, "s", Modifier.weight(1f))
                Metric("Prefill", prefill?.median, "tok/s", Modifier.weight(1f))
            }
            Text("${measured.size} measured · ${result.samples.count { it.warmup }} warm-up · MTP ${if (result.settings.speculativeDecoding) "on" else "off"}", style = MaterialTheme.typography.bodySmall)
            if (decode != null) Text("Decode p95 ${decode.p95 format 2} · range ${decode.min format 2}–${decode.max format 2} tok/s", style = MaterialTheme.typography.bodySmall)
            if (ttft != null) Text("TTFT p95 ${ttft.p95 format 3} s · native init median ${init?.median?.let { it format 2 } ?: "—"} s", style = MaterialTheme.typography.bodySmall)
            result.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            TextButton(onClick = { expanded = !expanded }) { Text(if (expanded) "Hide run details" else "Run details") }
            if (expanded) {
                SelectionContainer {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("${result.startedAt}\n${result.device.model} / ${result.device.soc}\nSHA-256 ${result.model.sha256}", style = MaterialTheme.typography.labelSmall)
                        Text(if (result.settings.workload == Workload.FIXED_TOKENS) "Requested ${result.settings.prefillTokens} input / ${result.settings.outputTokens} output tokens. Context is native-harness controlled." else "Output limit ${result.settings.outputTokens}; context ${result.settings.contextTokens}.", style = MaterialTheme.typography.bodySmall)
                        result.samples.forEach { sample ->
                            HorizontalDivider()
                            Text("${if (sample.warmup) "Warm-up" else "Run"} ${sample.index} · ${sample.metrics.prefillTokens} in / ${sample.metrics.decodeTokens} out", fontWeight = FontWeight.Medium)
                            Text("Wall ${sample.elapsedMs format 1} ms · visible text ${sample.firstVisibleTextMs?.let { it format 1 } ?: "—"} ms\nEngine load ${sample.engineLoadMs?.let { "${it format 1} ms" } ?: if (sample.engineReused) "reused" else "native harness"}\nThermal ${sample.before.thermalStatus} → ${sample.after.thermalStatus} · PSS ${sample.after.processPssMb format 0} MiB (end sample, not peak)", style = MaterialTheme.typography.bodySmall)
                        }
                        if (result.settings.workload == Workload.PROMPT) {
                            Text("Prompt: ${result.settings.prompt}", style = MaterialTheme.typography.bodySmall)
                            result.samples.lastOrNull()?.output?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Metric(label: String, value: Double?, unit: String, modifier: Modifier) {
    Column(modifier.background(Paper, MaterialTheme.shapes.small).padding(10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall)
        Text(value?.let { it format 2 } ?: "—", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Text(unit, style = MaterialTheme.typography.labelSmall)
    }
}

private infix fun Double.format(digits: Int): String = String.format(Locale.US, "%.${digits}f", this)
