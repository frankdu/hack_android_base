package dev.gemmabench.people

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

data class MatchPolicy(val threshold: Float, val margin: Float, val validated: Boolean = false)

object FaceMatching {
    fun normalize(values: FloatArray): FloatArray {
        require(values.isNotEmpty() && values.all { it.isFinite() }) { "Invalid face embedding." }
        val norm = sqrt(values.sumOf { it.toDouble() * it }.toFloat())
        require(norm.isFinite() && norm > 0) { "Empty face embedding." }
        return FloatArray(values.size) { values[it] / norm }
    }

    fun encode(values: FloatArray): ByteArray = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        .also { buffer -> values.forEach(buffer::putFloat) }.array()

    fun decode(bytes: ByteArray): FloatArray {
        require(bytes.isNotEmpty() && bytes.size % 4 == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.float }
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        return a.indices.sumOf { a[it].toDouble() * b[it] }.toFloat().coerceIn(-1f, 1f)
    }

    /** One fixed anchor per person avoids single-link clustering through gradually drifting faces. */
    fun choose(scores: List<Pair<String, Float>>, policy: MatchPolicy): String? {
        if (!policy.validated) return null
        val ranked = scores.filter { it.second.isFinite() }.sortedWith(compareByDescending<Pair<String, Float>> { it.second }.thenBy { it.first })
        val best = ranked.firstOrNull() ?: return null
        val runnerUp = ranked.getOrNull(1)?.second ?: -1f
        return best.first.takeIf { best.second >= policy.threshold && best.second - runnerUp >= policy.margin }
    }
}

data class DetectedFace(
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val embedding: FloatArray?, val quality: Float,
)

data class VisibleFace(
    val id: String, val photoUri: String, val personId: String?,
    val left: Float, val top: Float, val right: Float, val bottom: Float,
)

data class PersonGroup(val id: String, val name: String?, val faces: List<VisibleFace>) {
    val label: String get() = name ?: "Person ${id.take(8)}"
    val photoCount: Int get() = faces.map { it.photoUri }.distinct().size
}

data class PeopleState(
    val groups: List<PersonGroup> = emptyList(), val unassigned: List<VisibleFace> = emptyList(),
    val results: List<PeopleBatchResult> = emptyList(), val beforeId: Long? = null,
    val hasMore: Boolean = true, val batchNumber: Int = 0,
    val accessScope: dev.gemmabench.photos.PhotoAccess? = null,
    val calibrated: Boolean = false,
    val refreshing: Boolean = false,
) {
    val unfinished: Boolean get() = results.any { it.status == "PENDING" }
}

data class PeopleBatchResult(val photo: dev.gemmabench.photos.DevicePhoto, val status: String = "PENDING", val faces: Int = 0, val cached: Boolean = false, val error: String? = null)
