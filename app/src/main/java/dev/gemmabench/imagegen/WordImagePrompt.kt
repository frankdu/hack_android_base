package dev.gemmabench.imagegen

object WordImagePrompt {
    fun build(word: String, category: String): String {
        val subject = word.trim().replace(Regex("\\s+"), " ")
        val context = category.trim().replace(Regex("\\s+"), " ")
        require(subject.isNotEmpty() && context.isNotEmpty()) { "Enter both a word and a category." }
        require(subject.length <= 80 && context.length <= 80) { "Use at most 80 characters per field." }
        val scene = when {
            Regex("action|verb|movement", RegexOption.IGNORE_CASE).containsMatchIn(context) ->
                "one friendly character clearly performing $subject, with the action easy to recognize"
            Regex("emotion|feeling", RegexOption.IGNORE_CASE).containsMatchIn(context) ->
                "a friendly character showing $subject through facial expression and body language in a simple everyday scene"
            Regex("color|colour", RegexOption.IGNORE_CASE).containsMatchIn(context) ->
                "one familiar object in a clear $subject color, filling most of the frame"
            Regex("shape", RegexOption.IGNORE_CASE).containsMatchIn(context) ->
                "one large $subject shape with a clean outline on a plain background"
            else -> "$subject, recognizable features, a simple setting appropriate to $context"
        }
        // Keep the main concept first for the shorter CLIP context in SD/SDXL.
        return "$scene. Children's vocabulary illustration, category: $context. " +
            "Clear subject, accurate proportions, bright natural colors, soft light, uncluttered composition. " +
            "Show the meaning visually. No text, letters, labels, watermark or collage."
    }
}
