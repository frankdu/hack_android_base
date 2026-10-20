package dev.gemmabench.photos

import org.junit.Assert.*
import org.junit.Test

class PhotoEligibilityTest {
    @Test fun recognizesScreenshotNamesAndFolders() {
        listOf("Screenshot_20261020.png", "Screenshot20261020.jpg", "sCrEeNsHoT (2).png",
            "Screen_shot-2026.png", "Screen Capture 2026.jpg").forEach { name ->
            assertTrue(name, PhotoEligibility.isScreenshot(name, "Download/", null))
        }
        assertTrue(PhotoEligibility.isScreenshot("renamed.jpg", "Pictures/Screenshots/", null))
        assertTrue(PhotoEligibility.isScreenshot("renamed.jpg", "DCIM/sCrEeN_sHoTs/edited/", null))
        assertTrue(PhotoEligibility.isScreenshot("renamed.jpg", null, "Screenshots"))
    }

    @Test fun preservesCameraPhotosAndImagesWithoutScreenshotMetadata() {
        listOf("IMG_20261020.jpg", "PXL_20261020.png", "my_screenshot_notes.jpg", "Screenshotting birds.jpg").forEach { name ->
            assertFalse(name, PhotoEligibility.isScreenshot(name, "DCIM/Camera/", "Camera"))
        }
        assertFalse(PhotoEligibility.isScreenshot(null, null, null))
        assertFalse(PhotoEligibility.isScreenshot("bird.jpg", "Pictures/My Screenshots Album/", null))
    }
}
