package dev.gemmabench.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.platform.LocalContext
import dev.gemmabench.voice.VoiceAgentController
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.gemmabench.data.WordCategory
import dev.gemmabench.R
import dev.gemmabench.tts.SpeechController
import dev.gemmabench.settings.AppPreferencesStore
import dev.gemmabench.settings.settingsLabel

sealed interface DictionaryState {
    data object Loading : DictionaryState
    data class Ready(val categories: List<WordCategory>) : DictionaryState
    data object Error : DictionaryState
}

internal val WordColors = lightColorScheme(
    primary = Color(0xFF176B55),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCEEEDD),
    onPrimaryContainer = Color(0xFF153B32),
    background = Color(0xFFF6F7F1),
    surface = Color.White,
    onBackground = Color(0xFF1E2923),
    onSurface = Color(0xFF1E2923),
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun KidsEnglishApp(dictionary: DictionaryState, speech: SpeechController, preferences: AppPreferencesStore, agent: VoiceAgentController,
                   onOpenResearchLab: () -> Unit) {
    var selectedCategory by rememberSaveable { mutableStateOf<Int?>(null) }
    var settingsOpen by rememberSaveable { mutableStateOf(false) }
    val speechState by speech.state.collectAsStateWithLifecycle()
    val prefs by preferences.state.collectAsStateWithLifecycle()
    val voice by agent.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var pendingWord by remember { mutableStateOf<Pair<String, String>?>(null) }
    val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        pendingWord?.let { (word, category) -> agent.start(word, category, granted) }
        pendingWord = null
    }
    fun openAssistant(word: String, category: String) {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            agent.start(word, category, true)
        } else {
            pendingWord = word to category
            microphone.launch(Manifest.permission.RECORD_AUDIO)
        }
    }
    LaunchedEffect(selectedCategory, settingsOpen) { speech.stop(); agent.end() }
    val categoryListState = rememberLazyListState()
    val wordListState = key(selectedCategory) { rememberLazyListState() }
    val categories = (dictionary as? DictionaryState.Ready)?.categories.orEmpty()
    val category = selectedCategory?.let(categories::getOrNull)
    val canGoBack = settingsOpen || voice.word != null || selectedCategory != null
    val goBack: () -> Unit = {
        if (settingsOpen) settingsOpen = false else if (voice.word != null) agent.end() else selectedCategory = null
    }
    BackHandler(enabled = canGoBack, onBack = goBack)

    MaterialTheme(colorScheme = WordColors) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text(when {
                            settingsOpen -> "Settings"
                            voice.word != null -> "Word assistant"
                            else -> selectedCategory?.let { categories.getOrNull(it)?.name } ?: "English words"
                        }, maxLines = 2)
                    },
                    navigationIcon = {
                        if (settingsOpen || voice.word != null || selectedCategory != null) {
                            IconButton(onClick = goBack) {
                                Icon(painterResource(R.drawable.ic_arrow_back_24), contentDescription = "Back")
                            }
                        }
                    },
                    actions = {
                        if (!settingsOpen) {
                            if (speechState.activeWord != null && voice.word == null) {
                                IconButton(onClick = speech::stop) {
                                    Icon(painterResource(R.drawable.ic_stop_24), contentDescription = "Stop playback",
                                        tint = Color(0xFFD32F2F))
                                }
                            }
                            IconButton(onClick = { settingsOpen = true }) {
                                Icon(painterResource(R.drawable.ic_settings_24), contentDescription = "Settings")
                            }
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = WordColors.background),
                )
            },
        ) { insets ->
            Box(Modifier.fillMaxSize().padding(insets).consumeWindowInsets(insets), contentAlignment = Alignment.TopCenter) {
                val contentModifier = Modifier.widthIn(max = 720.dp).fillMaxSize()
                when {
                    settingsOpen -> AppSettingsPage(preferences, speech, contentModifier, onOpenResearchLab)
                    voice.word != null -> WordAssistantPage(voice,
                        onTalk = {
                            if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) agent.listen()
                            else { pendingWord = voice.word!! to voice.category; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                        }, onCancel = agent::cancel, onStopListening = agent::finishListening, onSend = agent::send,
                        onMicrophoneSettings = {
                            context.startActivity(android.content.Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                android.net.Uri.parse("package:${context.packageName}")))
                        },
                        modifier = contentModifier, showTextInput = prefs.assistantShowTextInput)
                    dictionary is DictionaryState.Loading -> Box(contentModifier, contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                            CircularProgressIndicator()
                            Text("Opening your words…")
                        }
                    }
                    dictionary is DictionaryState.Error -> Box(contentModifier.padding(24.dp), contentAlignment = Alignment.Center) {
                        Text("We couldn’t open the words. Please reopen the app to try again.")
                    }
                    category != null -> key(selectedCategory) {
                        LazyColumn(
                            modifier = contentModifier,
                            state = wordListState,
                            contentPadding = PaddingValues(20.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp),
                        ) {
                            item {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text("${category.items.size} words to explore", style = MaterialTheme.typography.bodyLarge)
                                    Text("Tap a word to hear ${speechState.defaultOption?.settingsLabel ?: "the default voice"}. Tap the sparkle to explore it with Gemma.",
                                        style = MaterialTheme.typography.bodyMedium)
                                }
                            }
                            itemsIndexed(category.items) { index, word ->
                                Surface(shape = MaterialTheme.shapes.large, modifier = Modifier.fillMaxWidth()) {
                                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                            Text("${index + 1}", color = MaterialTheme.colorScheme.primary,
                                                style = MaterialTheme.typography.labelLarge)
                                            BundledWordImage(word, category.name, size = 56.dp,
                                                modifier = Modifier.size(56.dp))
                                            Text(word, style = MaterialTheme.typography.headlineSmall,
                                                modifier = Modifier.weight(1f).clickable(role = androidx.compose.ui.semantics.Role.Button) {
                                                    agent.end(); speech.speakDefault(word)
                                                }.semantics { contentDescription = "Play $word with default voice" })
                                            IconButton(onClick = { agent.end(); speech.speakDefault(word) }) {
                                                SpeakingIcon(playing = speechState.activeWord == word && speechState.isSpeaking,
                                                    contentDescription = "Speak $word")
                                            }
                                            FilledTonalIconButton(onClick = { openAssistant(word, category.name) }) {
                                                Icon(painterResource(R.drawable.ic_assistant_24), contentDescription = "Talk about $word with Gemma")
                                            }
                                        }
                                        speechState.lastError?.takeIf { it.first == word }?.let { (_, message) ->
                                            Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> LazyColumn(
                        modifier = contentModifier,
                        state = categoryListState,
                        contentPadding = PaddingValues(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        item {
                            Column(Modifier.padding(bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Let’s learn a new word!", style = MaterialTheme.typography.headlineMedium,
                                    fontWeight = FontWeight.SemiBold)
                                Text("Choose a category to explore.", style = MaterialTheme.typography.bodyLarge)
                                Text("${categories.size} categories", color = MaterialTheme.colorScheme.primary,
                                    style = MaterialTheme.typography.labelLarge)
                            }
                        }
                        itemsIndexed(categories) { index, entry ->
                            Card(onClick = { selectedCategory = index }, modifier = Modifier.fillMaxWidth()) {
                                ListItem(
                                    headlineContent = { Text(entry.name, style = MaterialTheme.typography.titleLarge) },
                                    supportingContent = { Text("${entry.items.size} words") },
                                    leadingContent = {
                                        Surface(color = MaterialTheme.colorScheme.primaryContainer, shape = MaterialTheme.shapes.medium) {
                                            Box(Modifier.size(48.dp), contentAlignment = Alignment.Center) {
                                                Text(entry.name.take(1).uppercase(), style = MaterialTheme.typography.titleLarge)
                                            }
                                        }
                                    },
                                    trailingContent = { Text("›", style = MaterialTheme.typography.headlineSmall) },
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
