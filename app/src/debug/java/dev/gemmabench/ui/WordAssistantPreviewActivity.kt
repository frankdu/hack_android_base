package dev.gemmabench.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.gemmabench.GemmaBenchApplication
import dev.gemmabench.R
import dev.gemmabench.voice.*

/** Debug-only entry point: the real assistant layout with explicit sample data, no mic/model/audio. */
class WordAssistantPreviewActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val prefs by (application as GemmaBenchApplication).preferences.state.collectAsStateWithLifecycle()
            AssistantLayoutPreview(onBack = ::finish, showTextInput = prefs.assistantShowTextInput)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AssistantLayoutPreview(initialPhase: VoicePhase = VoicePhase.SPEAKING, onBack: () -> Unit = {},
                                   showTextInput: Boolean = false) {
    var phase by remember { mutableStateOf(initialPhase) }
    var menu by remember { mutableStateOf(false) }
    val sample = VoiceAgentState(word = "butterfly", category = "bug", phase = phase,
        messages = listOf(
            VoiceMessage(false, "A butterfly is an insect with wide, colorful wings. What would you like to know about butterflies?"),
            VoiceMessage(true, "Why are their wings so colorful?"),
        ),
        speakingText = if (phase == VoicePhase.SPEAKING) "Those bright colors can help butterflies find each other." else "",
        highlight = if (phase == VoicePhase.SPEAKING) 6 until 19 else null,
        level = 0.45f,
    )
    MaterialTheme(colorScheme = WordColors) {
        Scaffold(topBar = {
            TopAppBar(title = {
                Column {
                    Text("Word assistant")
                    Text("Layout preview · sample conversation", style = MaterialTheme.typography.labelSmall)
                }
            }, navigationIcon = {
                IconButton(onClick = onBack) { Icon(painterResource(R.drawable.ic_arrow_back_24), "Back") }
            }, actions = {
                Box {
                    TextButton(onClick = { menu = true }) { Text("State") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        listOf("Speaking" to VoicePhase.SPEAKING, "Listening" to VoicePhase.LISTENING,
                            "Transcribing" to VoicePhase.TRANSCRIBING, "Thinking" to VoicePhase.THINKING,
                            "Ready" to VoicePhase.PAUSED).forEach { (title, value) ->
                            DropdownMenuItem(text = { Text(title) }, onClick = { phase = value; menu = false })
                        }
                    }
                }
            }, colors = TopAppBarDefaults.topAppBarColors(containerColor = WordColors.background))
        }) { padding ->
            Box(Modifier.fillMaxSize().padding(padding).consumeWindowInsets(padding), contentAlignment = Alignment.TopCenter) {
                WordAssistantPage(sample, onTalk = { phase = VoicePhase.LISTENING },
                    onCancel = { phase = VoicePhase.PAUSED }, onStopListening = { phase = VoicePhase.TRANSCRIBING },
                    onSend = { phase = VoicePhase.SPEAKING }, onMicrophoneSettings = {},
                    modifier = Modifier.widthIn(max = 720.dp).fillMaxSize(),
                    showTextInput = showTextInput)
            }
        }
    }
}

@Preview(name = "Word assistant · speaking", showBackground = true, widthDp = 412, heightDp = 892)
@Composable
private fun SpeakingAssistantPreview() { AssistantLayoutPreview() }

@Preview(name = "Word assistant · listening", showBackground = true, widthDp = 412, heightDp = 892)
@Composable
private fun ListeningAssistantPreview() { AssistantLayoutPreview(VoicePhase.LISTENING) }

@Preview(name = "Word assistant · transcribing", showBackground = true, widthDp = 412, heightDp = 892)
@Composable
private fun TranscribingAssistantPreview() { AssistantLayoutPreview(VoicePhase.TRANSCRIBING) }

@Preview(name = "Word assistant · thinking", showBackground = true, widthDp = 412, heightDp = 892)
@Composable
private fun ThinkingAssistantPreview() { AssistantLayoutPreview(VoicePhase.THINKING) }

@Preview(name = "Word assistant · ready", showBackground = true, widthDp = 412, heightDp = 892)
@Composable
private fun ReadyAssistantPreview() { AssistantLayoutPreview(VoicePhase.PAUSED) }
