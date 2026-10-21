package dev.gemmabench

import android.app.KeyguardManager
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.gemmabench.benchmark.ModelSlot
import dev.gemmabench.settings.AppPreferencesStore
import dev.gemmabench.tts.SpeechOption
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Uses platform accessibility APIs so navigation also works on Android preview builds. */
@RunWith(AndroidJUnit4::class)
class KidsEnglishInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Before fun requireUnlockedDevice() {
        val keyguard = instrumentation.targetContext.getSystemService(KeyguardManager::class.java)
        assertFalse("Unlock the device before running UI navigation tests.", keyguard.isKeyguardLocked)
    }

    private fun findText(node: AccessibilityNodeInfo?, text: String): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isVisibleToUser && node.text?.contains(text) == true) return node
        for (index in 0 until node.childCount) {
            findText(node.getChild(index), text)?.let { return it }
        }
        return null
    }

    private fun awaitText(text: String): AccessibilityNodeInfo {
        val deadline = SystemClock.uptimeMillis() + 10_000
        while (SystemClock.uptimeMillis() < deadline) {
            val match = findText(instrumentation.uiAutomation.rootInActiveWindow, text)
            if (match != null) return match
            SystemClock.sleep(50)
        }
        fun snapshot(node: AccessibilityNodeInfo?): String {
            if (node == null) return ""
            return listOfNotNull(node.text?.toString(), node.contentDescription?.toString())
                .joinToString(" / ") + (0 until node.childCount).joinToString(" | ") { snapshot(node.getChild(it)) }
        }
        throw AssertionError("Expected visible text: $text. UI: ${snapshot(instrumentation.uiAutomation.rootInActiveWindow)}")
    }

    private fun clickText(text: String) {
        var node: AccessibilityNodeInfo? = awaitText(text)
        while (node != null && !node.isClickable) node = node.parent
        assertNotNull("Expected a clickable control for: $text", node)
        assertTrue(node!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
        instrumentation.waitForIdleSync()
    }

    private fun findDescription(node: AccessibilityNodeInfo?, description: String): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isVisibleToUser && node.contentDescription?.toString() == description) return node
        for (index in 0 until node.childCount) {
            findDescription(node.getChild(index), description)?.let { return it }
        }
        return null
    }

    private fun scroll(node: AccessibilityNodeInfo?): Boolean {
        if (node == null) return false
        if (node.isScrollable && node.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)) return true
        for (index in 0 until node.childCount) if (scroll(node.getChild(index))) return true
        return false
    }

    private fun clickDescription(description: String) {
        val deadline = SystemClock.uptimeMillis() + 10_000
        while (SystemClock.uptimeMillis() < deadline) {
            var node = findDescription(instrumentation.uiAutomation.rootInActiveWindow, description)
            if (node != null) {
                while (node != null && !node.isClickable) node = node.parent
                assertNotNull("Expected a clickable control: $description", node)
                assertTrue(node!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
                instrumentation.waitForIdleSync()
                return
            }
            scroll(instrumentation.uiAutomation.rootInActiveWindow)
            instrumentation.waitForIdleSync()
            SystemClock.sleep(100)
        }
        throw AssertionError("Expected visible control: $description")
    }

    @Test fun settingsPersistAcrossRecreationAndChangeWordAndLabDefaults() {
        val app = instrumentation.targetContext.applicationContext as GemmaBenchApplication
        val saved = app.preferences.state.value
        try {
            instrumentation.runOnMainSync { app.preferences.reset() }
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                awaitText("100 categories")
                clickText("Settings")
                awaitText("Text-to-speech Engine")
                clickDescription("Text-to-speech Engine: Kokoro TTS")
                assertEquals(SpeechOption.KOKORO, app.preferences.state.value.speechEngine)
                clickDescription("Default Local LLM: Gemma 4 E4B")
                assertEquals(ModelSlot.E4B, app.preferences.state.value.llmFamily)
                scenario.recreate()
                awaitText("Settings")
                val reloaded = AppPreferencesStore(instrumentation.targetContext).state.value
                assertEquals(SpeechOption.KOKORO, reloaded.speechEngine)
                assertEquals(ModelSlot.E4B, reloaded.llmFamily)
                clickText("Back")
                clickText("animal")
                awaitText("Tap a word to hear Kokoro TTS")
                awaitText("animal")
                clickText("Back")
                clickText("Settings")
                val monitor = instrumentation.addMonitor(ResearchLabActivity::class.java.name, null, false)
                try {
                    clickText("Research Lab")
                    val labActivity = instrumentation.waitForMonitorWithTimeout(monitor, 10_000)
                    assertNotNull(labActivity)
                    runBlocking { withTimeout(10_000) { while (app.lab.state.value.preferredLlmFamily != ModelSlot.E4B) delay(50) } }
                    instrumentation.runOnMainSync { labActivity.finish() }
                } finally { instrumentation.removeMonitor(monitor) }
            }
        } finally {
            instrumentation.runOnMainSync {
                app.preferences.reset()
                saved.speechEngine?.let(app.preferences::setSpeechEngine)
                saved.llmFamily?.let(app.preferences::setLlmFamily)
            }
        }
    }

    @Test fun everyVisibleWordHasThreeDistinctSpeechButtons() {
        ActivityScenario.launch(MainActivity::class.java).use {
            awaitText("100 categories")
            clickText("animal")
            awaitText("dog")
            for (voice in listOf("Android", "Kokoro", "Kitten")) {
                val description = "Play dog with $voice"
                val deadline = SystemClock.uptimeMillis() + 10_000
                var button: AccessibilityNodeInfo? = null
                while (SystemClock.uptimeMillis() < deadline && button == null) {
                    button = findDescription(instrumentation.uiAutomation.rootInActiveWindow, description)
                    if (button == null) SystemClock.sleep(50)
                }
                assertNotNull("Missing distinct speech button: $description", button)
                var clickable = button
                while (clickable != null && !clickable.isClickable) clickable = clickable.parent
                assertNotNull("Expected a clickable control for: $description", clickable)
            }
        }
    }

    @Test fun opensCategoryRestoresAfterRecreationAndReturnsWithSystemBack() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            awaitText("100 categories")
            clickText("animal")
            awaitText("dog")
            awaitText("cat")
            scenario.recreate()
            awaitText("dog")
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            awaitText("100 categories")
            clickText("bird")
            awaitText("robin")
            clickText("Back")
            awaitText("100 categories")
        }
    }

    @Test fun opensResearchLabFromSettingsAndReturnsToDictionary() {
        ActivityScenario.launch(MainActivity::class.java).use {
            awaitText("100 categories")
            clickText("Settings")
            val monitor = instrumentation.addMonitor(ResearchLabActivity::class.java.name, null, false)
            try {
                clickText("Research Lab")
                val lab = instrumentation.waitForMonitorWithTimeout(monitor, 10_000)
                assertNotNull("Settings should launch ResearchLabActivity", lab)
                awaitText("GEMMA / DEVICE LAB")
                instrumentation.runOnMainSync { (lab as ResearchLabActivity).onBackPressedDispatcher.onBackPressed() }
                awaitText("Research Lab")
                clickText("Back")
                awaitText("100 categories")
            } finally {
                instrumentation.removeMonitor(monitor)
            }
        }
    }
}
