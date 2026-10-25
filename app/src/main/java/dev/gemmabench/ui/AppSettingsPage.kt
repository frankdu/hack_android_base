package dev.gemmabench.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.gemmabench.GemmaBenchApplication
import dev.gemmabench.benchmark.ModelSlot
import dev.gemmabench.settings.AppPreferencesStore
import dev.gemmabench.settings.settingsLabel
import dev.gemmabench.tts.*

@Composable
fun AppSettingsPage(preferences: AppPreferencesStore, speech: SpeechController,
                    modifier: Modifier = Modifier, onOpenResearchLab: () -> Unit) {
    val application = LocalContext.current.applicationContext as GemmaBenchApplication
    val lab by application.lab.state.collectAsStateWithLifecycle()
    val prefs by preferences.state.collectAsStateWithLifecycle()
    val voices by speech.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { speech.refreshSystemAvailability() }
    Column(modifier.verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Card(onClick = onOpenResearchLab, modifier = Modifier.fillMaxWidth()) {
            ListItem(headlineContent = { Text("Research Lab") },
                supportingContent = { Text("Models, benchmarks, photos, and image generation") },
                trailingContent = { Text("›", style = MaterialTheme.typography.headlineSmall) })
        }
        if (dev.gemmabench.BuildConfig.DEBUG) {
            val context = LocalContext.current
            TextButton(onClick = {
                context.startActivity(android.content.Intent().setClassName(context,
                    "dev.gemmabench.ui.WordAssistantPreviewActivity"))
            }) { Text("Preview word assistant layout") }
        }
        Card(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().toggleable(value = prefs.assistantShowTextInput,
                role = Role.Switch, onValueChange = preferences::setAssistantShowTextInput)
                .padding(16.dp), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Assistant: show text input box", modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.titleMedium)
                Switch(checked = prefs.assistantShowTextInput, onCheckedChange = null)
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().toggleable(value = prefs.assistantIncludeImage,
                role = Role.Switch, onValueChange = preferences::setAssistantIncludeImage)
                .padding(16.dp), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Column(Modifier.weight(1f)) {
                    Text("Assistant: include image in Gemma input", style = MaterialTheme.typography.titleMedium)
                    Text("Off by default for faster replies. Gemma describes the image in the background and uses the description as context. Turn on to send the image with every reply.",
                        style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = prefs.assistantIncludeImage, onCheckedChange = null)
            }
        }
        Text("Text-to-speech Engine", style = MaterialTheme.typography.titleLarge)
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.selectableGroup()) {
                listOf(SpeechOption.SYSTEM, SpeechOption.KITTEN, SpeechOption.KOKORO).forEach { option ->
                    val detail = when (option) {
                        SpeechOption.SYSTEM -> voices.systemVoiceDetail
                        SpeechOption.KITTEN -> "Kiki · " + if (option in voices.ready) "Ready offline" else "24 MB download on first use"
                        SpeechOption.KOKORO -> "Adam · " + if (option in voices.ready) "Ready offline" else "92 MB download on first use"
                    }
                    SettingsChoice("Text-to-speech Engine", option.settingsLabel, detail,
                        selected = option == (prefs.speechEngine ?: voices.defaultOption ?: SpeechOption.SYSTEM)) {
                        speech.selectPreferred(option)
                    }
                }
            }
        }
        Text(when {
            voices.systemVoice == SystemVoiceAvailability.CHECKING && voices.defaultOption == SpeechOption.SYSTEM -> "Checking the system voice…"
            voices.defaultOption == null -> "No voice is ready. Select a voice above and try again."
            else -> "Using ${voices.defaultOption!!.settingsLabel} for word playback and the word assistant."
        }, style = MaterialTheme.typography.bodyMedium)
        Text("If the preferred voice is unavailable, playback tries System, Kitten TTS, then Kokoro TTS. Local downloads may use mobile data.",
            style = MaterialTheme.typography.bodySmall)

        Text("Default Local LLM", style = MaterialTheme.typography.titleLarge)
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.selectableGroup()) {
                listOf(ModelSlot.E2B, ModelSlot.E4B).forEach { family ->
                    SettingsChoice("Default Local LLM", family.label,
                        if (lab.models.any { if (family == ModelSlot.E2B) it.slot != ModelSlot.E4B else it.slot == family })
                            "Installed model available" else "Download from Research Lab",
                        selected = family == (prefs.llmFamily ?: lab.defaultLlm?.family ?: ModelSlot.E2B)) {
                        preferences.setLlmFamily(family)
                    }
                }
            }
        }
        Text(when {
            lab.device == null && lab.busy -> "Checking local models…"
            lab.defaultLlm != null -> "Using ${lab.defaultLlm!!.label} for the word assistant and Research Lab."
            else -> "No compatible local model is installed. Download one for the word assistant and Research Lab."
        }, style = MaterialTheme.typography.bodyMedium)
        Text("Automatic defaults prefer TPU, then GPU, then CPU, with E2B before E4B. A saved model choice is preferred when compatible. Settings do not start model downloads.",
            style = MaterialTheme.typography.bodySmall)
        TextButton(onClick = speech::resetDefaults) { Text("Reset defaults") }
    }
}

@Composable
private fun SettingsChoice(group: String, title: String, detail: String, selected: Boolean, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().selectable(selected, role = Role.RadioButton, onClick = onClick)
        .semantics { contentDescription = "$group: $title" }.padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        RadioButton(selected = selected, onClick = null)
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(detail, style = MaterialTheme.typography.bodySmall)
        }
    }
}
