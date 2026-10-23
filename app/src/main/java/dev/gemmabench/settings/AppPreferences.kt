package dev.gemmabench.settings

import android.content.Context
import android.content.SharedPreferences
import dev.gemmabench.benchmark.ModelSlot
import dev.gemmabench.tts.SpeechOption
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Null preferences retain automatic selection as device/model availability changes. */
data class AppPreferences(
    val speechEngine: SpeechOption? = null,
    val llmFamily: ModelSlot? = null,
    val assistantShowTextInput: Boolean = false,
)

class AppPreferencesStore(context: Context) {
    private val storage = context.getSharedPreferences("app_defaults", Context.MODE_PRIVATE)
    private fun read() = AppPreferences(
        SpeechOption.entries.firstOrNull { it.name == storage.getString("speech_engine", null) },
        listOf(ModelSlot.E2B, ModelSlot.E4B).firstOrNull { it.name == storage.getString("llm_family", null) },
        storage.getBoolean("assistant_show_text_input", false),
    )
    private val mutable = MutableStateFlow(read())
    val state = mutable.asStateFlow()
    private val listener = SharedPreferences.OnSharedPreferenceChangeListener { _, _ -> mutable.value = read() }

    init { storage.registerOnSharedPreferenceChangeListener(listener) }

    fun setSpeechEngine(engine: SpeechOption) { storage.edit().putString("speech_engine", engine.name).apply() }
    fun setLlmFamily(family: ModelSlot) {
        require(family == ModelSlot.E2B || family == ModelSlot.E4B)
        storage.edit().putString("llm_family", family.name).apply()
    }
    fun setAssistantShowTextInput(show: Boolean) {
        storage.edit().putBoolean("assistant_show_text_input", show).apply()
    }
    fun reset() {
        storage.edit().remove("speech_engine").remove("llm_family").remove("assistant_show_text_input").apply()
    }
}
