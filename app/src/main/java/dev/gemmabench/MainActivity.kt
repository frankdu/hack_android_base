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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
            KidsEnglishApp(dictionary) {
                startActivity(Intent(this, ResearchLabActivity::class.java))
            }
        }
    }
}
