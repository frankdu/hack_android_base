package dev.gemmabench

import android.app.KeyguardManager
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
        throw AssertionError("Expected visible text: $text")
    }

    private fun clickText(text: String) {
        var node: AccessibilityNodeInfo? = awaitText(text)
        while (node != null && !node.isClickable) node = node.parent
        assertNotNull("Expected a clickable control for: $text", node)
        assertTrue(node!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
        instrumentation.waitForIdleSync()
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
