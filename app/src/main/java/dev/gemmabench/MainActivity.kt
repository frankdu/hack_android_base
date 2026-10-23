package dev.gemmabench

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import dev.gemmabench.data.KidsEnglishDictionary
import dev.gemmabench.ui.DictionaryState
import dev.gemmabench.ui.KidsEnglishApp
import dev.gemmabench.tts.SpeechController
import dev.gemmabench.voice.VoiceAgentController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : ComponentActivity() {
    private lateinit var speech: SpeechController
    private lateinit var agent: VoiceAgentController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        speech = SpeechController(this)
        agent = VoiceAgentController(this, (application as GemmaBenchApplication).lab, speech)
        enableEdgeToEdge()
        setContent {
            val dictionary by produceState<DictionaryState>(DictionaryState.Loading) {
                value = try {
                    val categories = withContext(Dispatchers.IO) {
                        assets.open(KidsEnglishDictionary.ASSET_NAME).bufferedReader(Charsets.UTF_8).use {
                            KidsEnglishDictionary.parse(it)
                        }
                    }
                    DictionaryState.Ready(categories)
                } catch (_: IOException) {
                    DictionaryState.Error
                } catch (_: IllegalArgumentException) {
                    DictionaryState.Error
                }
            }
            KidsEnglishApp(dictionary, speech, (application as GemmaBenchApplication).preferences, agent) {
                startActivity(Intent(this, ResearchLabActivity::class.java))
            }
        }
    }

    override fun onStop() {
        agent.cancel()
        speech.stop()
        super.onStop()
    }

    override fun onDestroy() {
        agent.close()
        speech.close()
        super.onDestroy()
    }
}
