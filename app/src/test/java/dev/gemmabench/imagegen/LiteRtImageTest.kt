package dev.gemmabench.imagegen

import com.google.gson.Gson
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class LiteRtImageTest {
    @Test fun socAndBundlePathsCannotCrossTargetsOrEscapeTheStore() {
        assertEquals("Tensor_G5", LiteRtImageBundles.tensorSoc("Tensor G5"))
        assertEquals("Tensor_G6", LiteRtImageBundles.tensorSoc("Tensor_G6"))
        assertNull(LiteRtImageBundles.tensorSoc("Tensor G3"))
        assertNull(LiteRtImageBundles.tensorSoc("Snapdragon 8 Elite"))
        val root = File(System.getProperty("java.io.tmpdir"), "image-bundle-test")
        assertThrows(IllegalArgumentException::class.java) { LiteRtImageBundles.safeFile(root, "../outside") }
        assertThrows(IllegalArgumentException::class.java) { LiteRtImageBundles.safeFile(root, "/outside") }
        assertTrue(LiteRtImageBundles.safeFile(root, "graphs/part.tflite").path.startsWith(root.canonicalPath))
    }

    @Test fun installedMethodsScheduleOnlyCompatibleModelPairs() {
        val methods = listOf(ImageMethodSupport(ImageMethod.CPU, true, "installed GGUF", setOf("SD_TURBO", "SDXL_TURBO")),
            ImageMethodSupport(ImageMethod.TPU, true, "G5 bundle", setOf("SD_TURBO")))
        val run = newImageBenchmarkRun("test", 1, "test", ImageModels.all.take(2), methods, 5)
        assertEquals(90, run.samples.size)
        assertEquals(30, run.samples.count { it.method == ImageMethod.TPU })
        assertFalse(run.samples.any { it.method == ImageMethod.TPU && it.modelId == "SDXL_TURBO" })
        assertTrue(run.samples.filter { it.method == ImageMethod.TPU }.all { it.modelId == "SD_TURBO" })
    }

    @Test fun oldMethodSupportJsonRemainsCompatible() {
        val old = Gson().fromJson("""{"method":"CPU","available":true,"reason":"ARM"}""", ImageMethodSupport::class.java)
        assertTrue(old.supports("SD_TURBO"))
    }

    @Test fun clipBpeUsesMergesEosTruncationAndPadding() {
        val spec = Gson().fromJson("""{"kind":"clip","length":4,"pad":0,"bos":1,"eos":2,"prefix":"","suffix":"","paddingSide":"right",
            "vocab":{"a</w>":3,"c":4,"a":5,"t</w>":6,"ca":7,"cat</w>":8},"merges":["c a","ca t</w>"],
            "specialTokens":[{"text":"<|endoftext|>","id":2}],"fixtures":[]}""", com.google.gson.JsonObject::class.java)
        val tokenizer = ImageBpeTokenizer(spec)
        assertArrayEquals(longArrayOf(1, 8, 2, 0), tokenizer.encode("  CAT ").ids)
        assertArrayEquals(longArrayOf(1, 1, 1, 0), tokenizer.encode("cat").mask)
        assertArrayEquals(longArrayOf(1, 3, 3, 2), tokenizer.encode("a a a a a").ids)
    }

    /** Compact test-only specs retain exactly the vocab/merges used by the real Python fixtures. */
    @Test fun exportedTokenizerFixturesMatchPython() {
        listOf("sd", "xl", "flux").forEach { name ->
            val text = requireNotNull(requireNotNull(javaClass.classLoader).getResourceAsStream("image-tokenizers/$name.json")).bufferedReader().use { it.readText() }
            ImageBpeTokenizer(Gson().fromJson(text, com.google.gson.JsonObject::class.java)).validateFixtures()
        }
    }
}
