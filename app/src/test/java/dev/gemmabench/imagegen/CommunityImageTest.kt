package dev.gemmabench.imagegen

import com.google.gson.JsonParser
import dev.gemmabench.imagegen.bonsai.BonsaiMath
import dev.gemmabench.imagegen.bonsai.QwenTokenizer as BonsaiTokenizer
import dev.gemmabench.imagegen.iris.HostOps
import dev.gemmabench.imagegen.iris.QwenTokenizer as IrisTokenizer
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.util.zip.ZipInputStream

class CommunityImageTest {
    private fun <T> tokenizers(block: (File) -> T): T {
        val root = Files.createTempDirectory("litert-tokenizers").toFile()
        try {
            ZipInputStream(requireNotNull(javaClass.classLoader!!.getResourceAsStream("image-litert/tokenizers.zip"))).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    val file = LiteRtImageBundles.safeFile(root, entry.name)
                    file.parentFile!!.mkdirs(); file.outputStream().use { zip.copyTo(it) }
                }
            }
            return block(root)
        } finally { root.deleteRecursively() }
    }

    @Test fun fluxTokenizerMatchesPublishedPythonFixturesIncludingWhitespaceUnicodeAndDigits() = tokenizers { root ->
        val dir = File(root, "flux")
        val tokenizer = IrisTokenizer(dir, "qwen_special.txt")
        File(dir, "tokenizer_fixture.txt").forEachLine { line ->
            val pair = line.split("\t", limit = 2)
            val prompt = pair[0].replace("\\n", "\n").replace("\\t", "\t")
            val want = pair[1].split(',').map(String::toInt)
            assertEquals(prompt, want, tokenizer.encode(prompt))
        }
    }

    @Test fun bonsaiTokenizerMatchesPublishedGoldenCasesAndPreservesSuffixOnTruncation() = tokenizers { root ->
        val dir = File(root, "bonsai")
        val tokenizer = File(dir, "vocab.json").inputStream().use { vocab ->
            File(dir, "merges.txt").inputStream().use { merges -> BonsaiTokenizer(vocab, merges) }
        }
        val golden = JsonParser.parseString(File(dir, "golden.json").readText()).asJsonObject
        golden.getAsJsonArray("cases").forEach { entry ->
            val c = entry.asJsonObject
            val prompt = c["prompt"].asString
            assertEquals(prompt, c.getAsJsonArray("body_ids").map { it.asInt }, tokenizer.encode("user\n$prompt").toList())
            val result = tokenizer.encodePrompt(prompt)
            assertEquals(256, result.ids.size)
            assertEquals(256, result.mask.size)
            assertEquals(151644, result.ids.first())
        }
        val long = tokenizer.encodePrompt("apple ".repeat(500))
        assertArrayEquals(BonsaiTokenizer.SUFFIX_IDS, long.ids.takeLast(BonsaiTokenizer.SUFFIX_IDS.size).toIntArray())
        assertTrue(long.mask.all { it == 1 })
    }

    @Test fun catalogKeepsModelFormatsAndActualDimensionsExplicit() {
        assertEquals(5, CommunityImageModels.all.size)
        assertEquals(256, ImageModels.get("FLUX_KLEIN_LITERT").width)
        assertEquals(768, ImageModels.get("IRIS_LITERT").width)
        val files = ImageModels.all.flatMap { it.files }
        assertEquals(files.size, files.map { it.filename }.distinct().size)
        CommunityImageModels.all.forEach { m ->
            if (m.pipeline != ImagePipeline.LITERT_PLAN) {
                assertTrue(m.files.any { it.filename.endsWith(".tflite") })
                assertTrue(m.files.none { it.filename.endsWith(".gguf") || it.filename.endsWith(".litertlm") })
                assertTrue(m.files.all { it.filename.startsWith(m.id + "/") })
            } else {
                assertTrue(m.files.any { it.filename.endsWith("/manifest.json") })
                assertTrue(m.files.any { it.filename.endsWith(".tflite") })
                assertTrue(m.files.all { it.repository.startsWith("frankdu/") })
            }
        }
    }

    @Test fun fluxRopeMaskAndHalfEmbeddingConversionMatchTheGraphLayouts() {
        val mask = CommunityFlux.encoderMask(12)
        assertEquals(32 * 512 * 512, mask.size)
        assertEquals(0f, mask[11 * 512 + 11], 0f)
        assertEquals(-Float.MAX_VALUE, mask[11 * 512 + 12], 0f)
        assertEquals(-Float.MAX_VALUE, mask[0 * 512 + 1], 0f)
        assertEquals(mask[11 * 512 + 11], mask[512 * 512 + 11 * 512 + 11], 0f)
        val (cos, sin) = CommunityFlux.jointRope()
        assertEquals(768 * 64, cos.size)
        assertEquals(1f, cos[0], 0f)
        assertEquals(kotlin.math.cos(1f), cos[64 + 48], 1e-7f)
        assertEquals(kotlin.math.sin(1f), sin[(512 + 16) * 64 + 16], 1e-7f)
        assertEquals(1f, CommunityFlux.halfToFloat(0x3c00.toShort()), 0f)
        assertEquals(-2f, CommunityFlux.halfToFloat(0xc000.toShort()), 0f)
        assertEquals(5.9604645e-8f, CommunityFlux.halfToFloat(1), 0f)
    }

    @Test fun bonsaiLatentUnpatchifyAppliesPackedChannelNormalizationBeforePermutation() {
        val lat = FloatArray(4 * 128) { it.toFloat() }
        val out = BonsaiMath.unpatchify(lat, FloatArray(128) { 2f }, FloatArray(128) { 3f }, 2)
        assertEquals(3f, out[0], 0f)
        assertEquals(5f, out[1], 0f)
        assertEquals(7f, out[4], 0f)
        assertEquals(259f, out[2], 0f)
        assertEquals(11f, out[16], 0f)
        assertArrayEquals(floatArrayOf(1f, 0f), BonsaiMath.sigmas(1), 0f)
    }

    @Test fun irisSamplerAndPatchAssemblyKeepThePixelSpaceConvention() {
        val x = floatArrayOf(1f, -2f); val clean = FloatArray(2)
        HostOps.samplerStep(x, floatArrayOf(0.5f, 1f), 1f, 0.5f, clean)
        assertArrayEquals(floatArrayOf(0.5f, -3f), clean, 0f)
        assertArrayEquals(floatArrayOf(0.75f, -2.5f), x, 0f)
        val index = HostOps.patchIndex()
        assertEquals(HostOps.IMAGE_FLOATS, index.toSet().size)
        assertEquals(0, index[0]); assertEquals(768 * 768, index[1]); assertEquals(1, index[3])
        val pix = PortableImagePipeline.pixels(floatArrayOf(-1f, 0f, 1f), 1)
        assertEquals(0xff0080ff.toInt(), pix.single())
    }
}
