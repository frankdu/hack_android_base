package dev.gemmabench.photos

/** Metadata-based filtering; renamed screenshots outside these folders cannot be identified. */
object PhotoEligibility {
    private val screenshotName = Regex("^screen[ _-]?(?:shots?|captures?)(?:$|[\\s_.(\\-]|\\d)", RegexOption.IGNORE_CASE)
    private val folderSeparators = Regex("[ _-]")

    fun isScreenshot(name: String?, relativePath: String?, bucket: String?): Boolean {
        fun screenshotFolder(folder: String): Boolean = folder.trim().replace(folderSeparators, "")
            .let { it.equals("screenshots", true) || it.equals("screenshot", true) ||
                it.equals("screencaptures", true) || it.equals("screencapture", true) }
        return screenshotName.containsMatchIn(name.orEmpty()) ||
            relativePath.orEmpty().split('/').any(::screenshotFolder) ||
            screenshotFolder(bucket.orEmpty())
    }
}
