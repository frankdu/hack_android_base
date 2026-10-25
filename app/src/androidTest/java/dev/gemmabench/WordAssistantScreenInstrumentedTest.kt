package dev.gemmabench

import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.ui.WordAssistantPage
import dev.gemmabench.ui.WordAssistantPreviewActivity
import dev.gemmabench.voice.VoiceAgentState
import dev.gemmabench.voice.VoicePhase
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WordAssistantScreenInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val state = mutableStateOf(VoiceAgentState(word = "dog", category = "animal"))
    private val visible = mutableStateOf(true)
    private val activePhases = listOf(VoicePhase.THINKING, VoicePhase.TRANSCRIBING,
        VoicePhase.SPEAKING, VoicePhase.LISTENING)

    private fun launch(): ActivityScenario<WordAssistantPreviewActivity> =
        ActivityScenario.launch(WordAssistantPreviewActivity::class.java).also { scenario ->
            scenario.onActivity { activity ->
                activity.setContent {
                    MaterialTheme {
                        if (visible.value) WordAssistantPage(state.value, onTalk = {}, onCancel = {},
                            onStopListening = {}, onSend = {}, onMicrophoneSettings = {},
                            modifier = Modifier.fillMaxSize())
                    }
                }
            }
        }

    private fun View.hasScreenOnRequest(): Boolean = keepScreenOn ||
        (this is ViewGroup && (0 until childCount).any { getChildAt(it).hasScreenOnRequest() })

    private fun awaitRequest(scenario: ActivityScenario<WordAssistantPreviewActivity>, expected: Boolean) {
        val deadline = SystemClock.uptimeMillis() + 5_000
        do {
            var actual = !expected
            scenario.onActivity { actual = it.window.decorView.hasScreenOnRequest() }
            if (actual == expected) return
            SystemClock.sleep(20)
        } while (SystemClock.uptimeMillis() < deadline)
        fail("Expected keep-screen-on request: $expected for ${state.value.phase}")
    }

    @Test fun activePhasesHoldScreenAndReadyErrorAndLeavingReleaseIt() {
        launch().use { scenario ->
            awaitRequest(scenario, false)
            for (phase in activePhases) {
                scenario.onActivity { state.value = state.value.copy(phase = phase) }
                awaitRequest(scenario, true)
                scenario.onActivity { state.value = state.value.copy(phase = VoicePhase.PAUSED) }
                awaitRequest(scenario, false)
            }
            scenario.onActivity { state.value = state.value.copy(phase = VoicePhase.THINKING) }
            awaitRequest(scenario, true)
            scenario.onActivity { state.value = state.value.copy(phase = VoicePhase.ERROR) }
            awaitRequest(scenario, false)
            scenario.onActivity { state.value = state.value.copy(phase = VoicePhase.LISTENING) }
            awaitRequest(scenario, true)
            scenario.onActivity { visible.value = false }
            awaitRequest(scenario, false)
        }
    }

    private fun shell(command: String): String =
        instrumentation.uiAutomation.executeShellCommand(command).use { descriptor ->
            android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).bufferedReader().use { it.readText().trim() }
        }

    /** Opt-in emulator check: temporarily shorten the real system timer and restore it afterward. */
    @Test fun activeConversationOutlastsTimeoutAndIdleGetsFreshCountdown() {
        assumeTrue("Enable with -e screenTimeout true",
            InstrumentationRegistry.getArguments().getString("screenTimeout") == "true")
        assumeTrue("Changes timeout settings only on an emulator", Build.HARDWARE in listOf("ranchu", "goldfish"))
        val timeout = shell("settings get system screen_off_timeout")
        val stayOn = shell("settings get global stay_on_while_plugged_in")
        try {
            shell("settings put system screen_off_timeout 15000")
            shell("settings put global stay_on_while_plugged_in 0")
            shell("input keyevent KEYCODE_WAKEUP")
            shell("wm dismiss-keyguard")
            val power = instrumentation.targetContext.getSystemService(PowerManager::class.java)
            launch().use { scenario ->
                // No input events after launch: span more than the 15-second system timeout.
                for (phase in activePhases) {
                    scenario.onActivity { state.value = state.value.copy(phase = phase) }
                    awaitRequest(scenario, true)
                    SystemClock.sleep(6_000)
                    assertTrue("Screen timed out during $phase", power.isInteractive)
                }
                scenario.onActivity { state.value = state.value.copy(phase = VoicePhase.PAUSED) }
                awaitRequest(scenario, false)
                SystemClock.sleep(3_000)
                assertTrue("Idle should start a fresh countdown instead of sleeping immediately", power.isInteractive)
                val deadline = SystemClock.uptimeMillis() + 20_000
                while (power.isInteractive && SystemClock.uptimeMillis() < deadline) SystemClock.sleep(100)
                assertFalse("Idle screen should time out normally", power.isInteractive)
            }
        } finally {
            shell(if (timeout == "null") "settings delete system screen_off_timeout"
                else "settings put system screen_off_timeout $timeout")
            shell(if (stayOn == "null") "settings delete global stay_on_while_plugged_in"
                else "settings put global stay_on_while_plugged_in $stayOn")
            shell("input keyevent KEYCODE_WAKEUP")
        }
    }
}
