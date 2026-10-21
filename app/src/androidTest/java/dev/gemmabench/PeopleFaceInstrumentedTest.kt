package dev.gemmabench

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import dev.gemmabench.people.*
import dev.gemmabench.photos.PhotoCacheRequest
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class PeopleFaceInstrumentedTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun nativePipelineHandlesNoFacesAndChecksModels() {
        val engine = FaceEngine(context)
        assertTrue("Bundled model-specific calibration must enable matching.", engine.policy().validated)
        val blank = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888)
        try { assertTrue(engine.analyze(blank).isEmpty()) } finally { blank.recycle() }
    }

    data class Sample(val identity: String, val split: String, val embedding: FloatArray)
    data class Evaluation(val queries: Int, val correct: Int, val falseMerges: Int, val rejected: Int, val negativeComparisons: Int) {
        val recall = if (queries == 0) 0.0 else correct.toDouble() / queries
    }

    data class ClusterEvaluation(val photos: Int, val groups: Int, val falseMerges: Int, val recoveredPairs: Int, val samePersonPairs: Int, val falsePairs: Int) {
        val precision = if (recoveredPairs + falsePairs == 0) 1.0 else recoveredPairs.toDouble() / (recoveredPairs + falsePairs)
        val recall = if (samePersonPairs == 0) 0.0 else recoveredPairs.toDouble() / samePersonPairs
    }

    private fun cluster(samples: List<Sample>, policy: MatchPolicy): ClusterEvaluation {
        val anchors = linkedMapOf<String, Sample>()
        val assignments = mutableListOf<Pair<Sample, String>>()
        val byIdentity = samples.groupBy { it.identity }.toSortedMap()
        var falseMerges = 0
        // Interleave identities, as photos of different people arrive across batches.
        for (index in 0 until byIdentity.values.maxOf { it.size }) for (photos in byIdentity.values) {
            val sample = photos.getOrNull(index) ?: continue
            val match = FaceMatching.choose(anchors.map { (id, anchor) -> id to FaceMatching.cosine(sample.embedding, anchor.embedding) }, policy)
            if (match != null && anchors.getValue(match).identity != sample.identity) falseMerges++
            val group = match ?: "group-${anchors.size}".also { anchors[it] = sample }
            assignments += sample to group
        }
        var recovered = 0; var same = 0; var falsePairs = 0
        for (i in assignments.indices) for (j in 0 until i) {
            val sameIdentity = assignments[i].first.identity == assignments[j].first.identity
            val sameGroup = assignments[i].second == assignments[j].second
            if (sameIdentity) same++
            if (sameIdentity && sameGroup) recovered++
            if (!sameIdentity && sameGroup) falsePairs++
        }
        return ClusterEvaluation(assignments.size, anchors.size, falseMerges, recovered, same, falsePairs)
    }

    private fun evaluate(samples: List<Sample>, policy: MatchPolicy): Evaluation {
        val identities = samples.groupBy { it.identity }
        val anchors = identities.mapValues { it.value.first().embedding }
        var queries = 0; var correct = 0; var falseMerges = 0; var rejected = 0
        for ((identity, photos) in identities) for (photo in photos.drop(1)) {
            queries++
            val match = FaceMatching.choose(anchors.map { (id, vector) -> id to FaceMatching.cosine(photo.embedding, vector) }, policy)
            when (match) { null -> rejected++; identity -> correct++; else -> falseMerges++ }
        }
        return Evaluation(queries, correct, falseMerges, rejected, queries * (anchors.size - 1))
    }

    /** Opt-in local benchmark; fixtures and outputs stay in app-private test storage. */
    @Test fun calibrateAndValidateOnIdentityDisjointPhotos() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("peopleValidation") == "true")
        val dir = File(context.filesDir, "people-validation")
        val manifest = File(dir, "manifest.json")
        check(manifest.isFile) { "Stage fixtures with scripts/prepare-people-validation.py first." }
        val records = JsonParser.parseString(manifest.readText()).asJsonObject["records"].asJsonArray
        val engine = FaceEngine(context)
        val samples = mutableListOf<Sample>()
        val times = mutableListOf<Double>()
        var detected = 0
        val identitiesBySplit = mutableMapOf<String, MutableSet<String>>()
        var maxHeap = 0L
        var peakPss = 0L
        var firstUsable: File? = null
        for (record in records) {
            val row = record.asJsonObject
            val file = File(dir, row["file"].asString)
            assertEquals(row["sha256"].asString, PhotoCacheRequest.sha256(file.readBytes()))
            val split = row["split"].asString
            val identity = row["identity"].asString
            identitiesBySplit.getOrPut(split) { mutableSetOf() }.add(identity)
            val bitmap = requireNotNull(BitmapFactory.decodeFile(file.absolutePath))
            val started = System.nanoTime()
            val faces = try { engine.analyze(bitmap) } finally { bitmap.recycle() }
            times += (System.nanoTime() - started) / 1_000_000.0
            if (faces.isNotEmpty()) detected++
            // LFW is centered on one labeled face; use the largest face if another appears.
            val embedding = faces.maxByOrNull { (it.right - it.left) * (it.bottom - it.top) }?.embedding
            if (embedding != null) { samples += Sample(identity, split, embedding); if (firstUsable == null) firstUsable = file }
            maxHeap = maxOf(maxHeap, android.os.Debug.getNativeHeapAllocatedSize())
            peakPss = maxOf(peakPss, android.os.Debug.MemoryInfo().also { android.os.Debug.getMemoryInfo(it) }.totalPss.toLong() * 1024)
        }
        assertTrue(identitiesBySplit.getValue("calibration").intersect(identitiesBySplit.getValue("holdout")).isEmpty())
        val calibration = samples.filter { it.split == "calibration" }
        val candidates = (35..85).flatMap { threshold -> (5..20 step 5).map { margin -> MatchPolicy(threshold / 100f, margin / 100f, true) } }
        val selected = candidates.map { it to evaluate(calibration, it) }.filter { it.second.falseMerges == 0 }
            .sortedWith(compareByDescending<Pair<MatchPolicy, Evaluation>> { it.second.correct }.thenByDescending { it.first.threshold }.thenByDescending { it.first.margin })
            .firstOrNull() ?: error("No conservative calibration threshold found.")
        val heldOut = evaluate(samples.filter { it.split == "holdout" }, selected.first)
        val heldOutClusters = cluster(samples.filter { it.split == "holdout" }, selected.first)
        val accepted = heldOut.falseMerges == 0 && heldOut.recall >= 0.60 && heldOut.negativeComparisons >= 1000 &&
            heldOutClusters.falseMerges == 0 && heldOutClusters.recall >= 0.60 && detected.toDouble() / records.size() >= 0.80
        val sorted = times.sorted()
        val report = linkedMapOf<String, Any>(
            "pipelineVersion" to FaceEngine.VERSION, "detectorSha256" to FaceEngine.DETECTOR_SHA,
            "recognizerSha256" to FaceEngine.RECOGNIZER_SHA,
            "threshold" to selected.first.threshold, "margin" to selected.first.margin, "validated" to accepted,
            "dataset" to "LFW funneled, deterministic seed 42, 40 identities, 8 photos each; identity-disjoint splits",
            "manifestSha256" to PhotoCacheRequest.sha256(manifest.readBytes()),
            "photos" to records.size(), "detectedPhotos" to detected, "usablePhotos" to samples.size,
            "calibration" to selected.second, "holdout" to heldOut,
            "holdoutIncrementalGrouping" to heldOutClusters,
            "medianMs" to sorted[sorted.size / 2], "p95Ms" to sorted[(sorted.size * 0.95).toInt()],
            "peakNativeHeapBytes" to maxHeap, "peakProcessPssBytes" to peakPss, "device" to android.os.Build.MODEL,
            "limits" to "Small public-figure fixture; not a demographic or personal-album accuracy guarantee. Native allocated heap is not total process memory."
        )
        File(context.filesDir, "people-calibration.json").writeText(GsonBuilder().setPrettyPrinting().create().toJson(report))
        assertTrue("Held-out calibration acceptance failed; see people-calibration.json", accepted)
        // Test ImageDecoder's EXIF path independently of the in-memory fixture path.
        val source = requireNotNull(firstUsable)
        val oriented = File(context.cacheDir, "people-exif.jpg")
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, android.graphics.Matrix().apply { postRotate(90f) }, true)
        try {
            oriented.outputStream().use { rotated.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            ExifInterface(oriented.absolutePath).apply { setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_270.toString()); saveAttributes() }
            val upright = engine.analyze(android.net.Uri.fromFile(oriented).toString())
            assertTrue(upright.isNotEmpty())
            for (face in upright) {
                assertTrue(face.left >= 0 && face.top >= 0 && face.right <= 1000 && face.bottom <= 1000)
                assertTrue(face.left < face.right && face.top < face.bottom)
            }
        } finally { bitmap.recycle(); rotated.recycle(); oriented.delete() }
    }
}
