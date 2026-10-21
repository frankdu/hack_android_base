package dev.gemmabench.ui

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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.gemmabench.data.WordCategory
import dev.gemmabench.tts.SpeechController
import dev.gemmabench.tts.SpeechOption
import dev.gemmabench.tts.SystemVoiceAvailability
import dev.gemmabench.settings.AppPreferencesStore
import dev.gemmabench.settings.settingsLabel

sealed interface DictionaryState {
    data object Loading : DictionaryState
    data class Ready(val categories: List<WordCategory>) : DictionaryState
    data object Error : DictionaryState
}

private val WordColors = lightColorScheme(
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
fun KidsEnglishApp(dictionary: DictionaryState, speech: SpeechController, preferences: AppPreferencesStore,
                   onOpenResearchLab: () -> Unit) {
    var selectedCategory by rememberSaveable { mutableStateOf<Int?>(null) }
    var settingsOpen by rememberSaveable { mutableStateOf(false) }
    val speechState by speech.state.collectAsStateWithLifecycle()
    LaunchedEffect(selectedCategory, settingsOpen) { speech.stop() }
    val categoryListState = rememberLazyListState()
    val wordListState = key(selectedCategory) { rememberLazyListState() }
    val categories = (dictionary as? DictionaryState.Ready)?.categories.orEmpty()
    val category = selectedCategory?.let(categories::getOrNull)
    val canGoBack = settingsOpen || selectedCategory != null
    val goBack: () -> Unit = {
        if (settingsOpen) settingsOpen = false else selectedCategory = null
    }
    BackHandler(enabled = canGoBack, onBack = goBack)

    MaterialTheme(colorScheme = WordColors) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text(when {
                            settingsOpen -> "Settings"
                            else -> selectedCategory?.let { categories.getOrNull(it)?.name } ?: "English words"
                        }, maxLines = 2)
                    },
                    navigationIcon = {
                        if (settingsOpen || selectedCategory != null) TextButton(onClick = goBack) { Text("Back") }
                    },
                    actions = {
                        if (!settingsOpen) TextButton(onClick = { settingsOpen = true }) { Text("Settings") }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = WordColors.background),
                )
            },
        ) { insets ->
            Box(Modifier.fillMaxSize().padding(insets), contentAlignment = Alignment.TopCenter) {
                val contentModifier = Modifier.widthIn(max = 720.dp).fillMaxSize()
                when {
                    settingsOpen -> AppSettingsPage(preferences, speech, contentModifier, onOpenResearchLab)
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
                                    Text("Tap a word to hear ${speechState.defaultOption?.settingsLabel ?: "the default voice"}, or use the buttons to compare.",
                                        style = MaterialTheme.typography.bodyMedium)
                                    Text("Tap a voice to compare. Local models download once (Kokoro 92 MB · Kitten 24 MB); mobile data may be used.",
                                        style = MaterialTheme.typography.bodyMedium)
                                    if (speechState.activeWord != null) TextButton(onClick = speech::stop) { Text("Stop playback / download") }
                                }
                            }
                            itemsIndexed(category.items) { index, word ->
                                Surface(shape = MaterialTheme.shapes.large, modifier = Modifier.fillMaxWidth()) {
                                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).clickable(role = androidx.compose.ui.semantics.Role.Button) {
                                            speech.speakDefault(word)
                                        }.semantics { contentDescription = "Play $word with default voice" },
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                                            Text("${index + 1}", color = MaterialTheme.colorScheme.primary,
                                                style = MaterialTheme.typography.labelLarge)
                                            Text(word, style = MaterialTheme.typography.headlineSmall)
                                        }
                                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                            SpeechOption.entries.forEach { option ->
                                                val active = speechState.activeWord == word && speechState.activeOption == option
                                                Column(Modifier.weight(1f)) {
                                                    OutlinedButton(
                                                        onClick = { if (active) speech.stop() else speech.speak(word, option) },
                                                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 8.dp),
                                                        modifier = Modifier.fillMaxWidth().semantics {
                                                            contentDescription = "${if (active) "Stop" else "Play"} $word with ${option.label}"
                                                        },
                                                    ) { Text(option.label, maxLines = 1) }
                                                    Text(speechState.results[word to option] ?: when {
                                                        option == SpeechOption.SYSTEM && speechState.systemVoice != SystemVoiceAvailability.AVAILABLE ->
                                                            if (speechState.systemVoice == SystemVoiceAvailability.CHECKING) "Checking…" else "Unavailable"
                                                        option !in speechState.ready -> "Download ${option.bytes / 1_000_000} MB"
                                                        else -> "Ready"
                                                    }, style = MaterialTheme.typography.labelSmall,
                                                        color = MaterialTheme.colorScheme.primary)
                                                }
                                            }
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
