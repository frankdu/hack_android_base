package dev.gemmabench

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.gemmabench.ui.LabApp

class MainActivity : ComponentActivity() {
    private val lab get() = (application as GemmaBenchApplication).lab

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val state by lab.state.collectAsStateWithLifecycle()
            LaunchedEffect(state.runningInference) {
                if (state.runningInference) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            DisposableEffect(Unit) {
                onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
            LabApp(state, lab)
        }
    }

    override fun onStop() {
        super.onStop()
        // Keep comparative runs in the foreground; stop without destroying JNI resources mid-call.
        if (!isChangingConfigurations) lab.stop()
    }
}
