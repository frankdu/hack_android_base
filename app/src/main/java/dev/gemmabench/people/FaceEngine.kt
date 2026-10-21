package dev.gemmabench.people

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import com.google.gson.JsonParser
import dev.gemmabench.photos.PhotoCacheRequest
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import org.opencv.objdetect.FaceRecognizerSF
import java.io.File
import kotlin.math.roundToInt

/** Worker-thread only; images/embeddings never leave the device. */
class FaceEngine(private val context: Context) {
    companion object {
        const val DETECTOR = "face_detection_yunet_2023mar.onnx"
        const val RECOGNIZER = "face_recognition_sface_2021dec.onnx"
        const val DETECTOR_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
        const val RECOGNIZER_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
        const val VERSION = "yunet2023-sface2021-upright2048-min48-score09-v1"
        const val EMBEDDING_VERSION = "$VERSION:$DETECTOR_SHA:$RECOGNIZER_SHA"
    }

    private var detector: FaceDetectorYN? = null
    private var recognizer: FaceRecognizerSF? = null

    fun policy(): MatchPolicy = runCatching {
        val json = context.assets.open("people/calibration.json").bufferedReader().use { JsonParser.parseReader(it).asJsonObject }
        check(json["pipelineVersion"].asString == VERSION && json["recognizerSha256"].asString == RECOGNIZER_SHA && json["detectorSha256"].asString == DETECTOR_SHA)
        val threshold = json["threshold"].asFloat
        val margin = json["margin"].asFloat
        check(threshold in 0f..1f && margin in 0f..2f)
        MatchPolicy(threshold, margin, json["validated"].asBoolean)
    }.getOrDefault(MatchPolicy(1f, 2f))

    private fun model(name: String, sha: String): File {
        val dir = File(context.noBackupFilesDir, "people-models").apply { mkdirs() }
        val target = File(dir, name)
        fun valid() = target.isFile && target.inputStream().use { input ->
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(65536)
            while (true) { val read = input.read(buffer); if (read < 0) break; digest.update(buffer, 0, read) }
            digest.digest().joinToString("") { "%02x".format(it) } == sha
        }
        if (!valid()) {
            val staged = File(dir, "$name.part")
            context.assets.open("people/$name").use { input -> staged.outputStream().use { input.copyTo(it) } }
            check(PhotoCacheRequest.sha256(staged.readBytes()) == sha) { "Bundled face model checksum failed." }
            check(staged.renameTo(target)) { "Cannot install face model." }
        }
        return target
    }

    private fun initialize() {
        if (detector != null) return
        check(OpenCVLoader.initLocal()) { "OpenCV could not initialize." }
        val detection = model(DETECTOR, DETECTOR_SHA)
        val recognition = model(RECOGNIZER, RECOGNIZER_SHA)
        recognizer = FaceRecognizerSF.create(recognition.absolutePath, "", Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_CPU)
        detector = FaceDetectorYN.create(detection.absolutePath, "", Size(320.0, 320.0), 0.9f, 0.3f, 5000, Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_CPU)
    }

    fun analyze(uri: String): List<DetectedFace> {
        val bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, Uri.parse(uri))) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val scale = minOf(1.0, 2048.0 / maxOf(info.size.width, info.size.height))
            decoder.setTargetSize(maxOf(1, (info.size.width * scale).roundToInt()), maxOf(1, (info.size.height * scale).roundToInt()))
        }
        return try { analyze(bitmap) } finally { bitmap.recycle() }
    }

    fun analyze(bitmap: Bitmap): List<DetectedFace> {
        initialize()
        val rgba = Mat()
        val bgr = Mat()
        val scaled = Mat()
        val detections = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            // YuNet's training scale tops out near 300px faces. Detect on a bounded image,
            // then map its landmarks back to the higher resolution upright source for alignment.
            val scale = minOf(1.0, 640.0 / maxOf(bgr.cols(), bgr.rows()))
            Imgproc.resize(bgr, scaled, Size((bgr.cols() * scale).roundToInt().toDouble(), (bgr.rows() * scale).roundToInt().toDouble()))
            detector!!.setInputSize(scaled.size())
            detector!!.detect(scaled, detections)
            return (0 until detections.rows()).mapNotNull { index ->
                val row = detections.row(index)
                val aligned = Mat()
                val feature = Mat()
                try {
                    val coords = FloatArray(15)
                    row.get(0, 0, coords)
                    for (i in 0..13) coords[i] /= scale.toFloat()
                    val left = coords[0].coerceIn(0f, bgr.cols().toFloat())
                    val top = coords[1].coerceIn(0f, bgr.rows().toFloat())
                    val right = (coords[0] + coords[2]).coerceIn(0f, bgr.cols().toFloat())
                    val bottom = (coords[1] + coords[3]).coerceIn(0f, bgr.rows().toFloat())
                    if (left >= right || top >= bottom) return@mapNotNull null
                    row.put(0, 0, coords)
                    val embedding = if (minOf(right - left, bottom - top) >= 48f) {
                        recognizer!!.alignCrop(bgr, row, aligned)
                        recognizer!!.feature(aligned, feature)
                        FaceMatching.normalize(FloatArray(feature.total().toInt()).also { feature.get(0, 0, it) })
                    } else null
                    DetectedFace(left / bgr.cols() * 1000f, top / bgr.rows() * 1000f,
                        right / bgr.cols() * 1000f, bottom / bgr.rows() * 1000f, embedding, coords[14])
                } finally { row.release(); aligned.release(); feature.release() }
            }.sortedWith(compareBy<DetectedFace> { it.left }.thenBy { it.top })
        } finally { rgba.release(); bgr.release(); scaled.release(); detections.release() }
    }
}
