package dev.gemmabench.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.keepScreenOn
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.gemmabench.R
import dev.gemmabench.voice.*

@Composable
fun WordAssistantPage(state: VoiceAgentState, onTalk: () -> Unit, onCancel: () -> Unit,
                      onStopListening: () -> Unit, onSend: (String) -> Unit,
                      onMicrophoneSettings: () -> Unit,
                      modifier: Modifier = Modifier, showTextInput: Boolean = false) {
    val scroll = rememberLazyListState()
    var question by rememberSaveable(state.word) { mutableStateOf("") }
    val busy = state.phase in setOf(VoicePhase.THINKING, VoicePhase.TRANSCRIBING, VoicePhase.SPEAKING, VoicePhase.LISTENING)
    val listening = state.phase == VoicePhase.LISTENING
    val activeGreen = Color(0xFF66EE66)
    val actionLabel = when (state.phase) {
        VoicePhase.LISTENING -> "Done"
        VoicePhase.TRANSCRIBING, VoicePhase.THINKING -> "Cancel"
        VoicePhase.SPEAKING -> "Stop speaking"
        VoicePhase.PAUSED, VoicePhase.ERROR -> "Talk"
    }
    val status = when (state.phase) {
        VoicePhase.THINKING -> "Thinking about ${state.word}…"
        VoicePhase.SPEAKING -> "Gemma is talking"
        VoicePhase.LISTENING -> "Listening — your turn!"
        VoicePhase.TRANSCRIBING -> "Listening to your words…"
        VoicePhase.PAUSED -> "Ready when you are"
        VoicePhase.ERROR -> "Let’s try again"
    }
    LaunchedEffect(state.messages.size, state.speakingText, state.phase) {
        scroll.animateScrollToItem((scroll.layoutInfo.totalItemsCount - 1).coerceAtLeast(0))
    }
    // Android restarts its normal screen timeout when the last keep-screen-on request is removed.
    val awakeModifier = if (busy) Modifier.keepScreenOn() else Modifier
    BoxWithConstraints(modifier.then(awakeModifier).imePadding()) {
        val imageHeight = (maxHeight * 0.25f).coerceAtMost(220.dp)
        Column(Modifier.fillMaxSize().padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(state.word.orEmpty(), style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.SemiBold)
            state.word?.let { word ->
                BundledWordImage(word, state.category, size = imageHeight,
                    modifier = Modifier.fillMaxWidth().height(imageHeight))
            }
            Surface(color = MaterialTheme.colorScheme.primaryContainer, shape = MaterialTheme.shapes.large) {
                Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(status, style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
                    if (state.phase == VoicePhase.LISTENING) {
                        LinearProgressIndicator(progress = { state.level }, modifier = Modifier.fillMaxWidth())
                        Text("Speak, then pause. Tap Done when you finish, or Cancel to discard.", style = MaterialTheme.typography.bodySmall)
                    } else if (state.phase == VoicePhase.THINKING || state.phase == VoicePhase.TRANSCRIBING) {
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                    }
                    state.detail?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
                    if (state.detail?.startsWith("Allow microphone") == true) {
                        TextButton(onClick = onMicrophoneSettings) { Text("Microphone settings") }
                    }
                }
            }
            LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = scroll,
                verticalArrangement = Arrangement.spacedBy(12.dp), contentPadding = PaddingValues(vertical = 8.dp)) {
                items(state.messages) { message ->
                    Surface(shape = MaterialTheme.shapes.large,
                        color = if (message.child) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface) {
                        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(if (message.child) "You" else "Gemma", style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary)
                            Text(message.text, style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
                if (state.speakingText.isNotEmpty()) item {
                    Surface(shape = MaterialTheme.shapes.large) {
                        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("Gemma", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
                            Text(buildAnnotatedString {
                                append(state.speakingText)
                                state.highlight?.let { range ->
                                    val end = (range.last + 1).coerceAtMost(length)
                                    if (range.first in 0 until end) addStyle(SpanStyle(
                                        background = MaterialTheme.colorScheme.primaryContainer,
                                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                                        fontWeight = FontWeight.Bold), range.first, end)
                                }
                            }, style = MaterialTheme.typography.headlineSmall)
                        }
                    }
                }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = when {
                        listening -> onStopListening
                        busy -> onCancel
                        else -> onTalk
                    }, modifier = Modifier.weight(1f),
                    colors = if (busy) ButtonDefaults.buttonColors(containerColor = activeGreen, contentColor = Color(0xFF153B32))
                        else ButtonDefaults.buttonColors()) {
                    Icon(painterResource(if (busy && !listening) R.drawable.ic_stop_24 else R.drawable.ic_mic_24),
                        contentDescription = null)
                    Spacer(Modifier.width(8.dp)); Text(actionLabel)
                }
                if (listening) {
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
            }
            if (showTextInput) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(value = question, onValueChange = { question = it.take(600) },
                        label = { Text("Type a question") }, modifier = Modifier.weight(1f), maxLines = 3, enabled = !busy)
                    IconButton(onClick = { onSend(question); question = "" }, enabled = !busy && question.isNotBlank()) {
                        Icon(painterResource(R.drawable.ic_send_24), contentDescription = "Send")
                    }
                }
            }
            Text("Voice and conversation stay on this device. Gemma can make mistakes.",
                style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(bottom = 12.dp))
        }
    }
}
