package com.example.benchmarking

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import com.google.gson.Gson
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

enum class MpDelegate { CPU, GPU }
enum class TfDelegate { CPU, GPU, NNAPI }
enum class InputMode { LANDMARKS_126, IMAGE_4D }

data class PipelineBenchmarkResult(
    val deviceName: String,
    val mediapipeDelegate: String,
    val modelDelegate: String,
    val inputMode: String,
    val meanMs: Double,
    val medianMs: Double,
    val stdMs: Double,
    val p95Ms: Double,
    val fps: Double,
    val sampleCount: Int,
    val accuracy: Double,
    val totalCorrect: Int,
    val totalSamples: Int,
    val noDetections: Int,
    val executionInfo: String
)

class PipelineBenchmarkRunner(
    private val context: Context,
    private val datasetTreeUri: Uri,
    private val modelAssetPath: String = "2.tflite",
    private val labelsAssetPath: String = "labels.txt",
    private val handLandmarkerAssetPath: String = "hand_landmarker.task",
    private val mpDelegate: MpDelegate = MpDelegate.CPU,
    private val tfDelegate: TfDelegate = TfDelegate.CPU,
    private val warmupRuns: Int = 5,
    private val totalRunsLimit: Int? = null,
    private val recordPerSample: Boolean = false,
    private val verbosePerSampleLogs: Boolean = false // new toggle
) {

    companion object {
        private const val TAG = "PipelineBenchmark"
    }

    // Core
    private var landmarker: HandLandmarker? = null
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // Model info
    private lateinit var labels: List<String>
    private lateinit var inputTensor: Tensor
    private lateinit var outputTensor: Tensor
    private var inputMode: InputMode = InputMode.LANDMARKS_126

    // Landmark buffers
    private lateinit var landmarkInput: Array<FloatArray>
    private lateinit var landmarkOutput: Array<FloatArray>

    // Image buffers
    private var imageInputBuffer: ByteBuffer? = null
    private var imageOutputBuffer: ByteBuffer? = null
    private var imageInputShape: IntArray = intArrayOf()
    private var imageOutputShape: IntArray = intArrayOf()
    private var imageInputType: DataType = DataType.FLOAT32
    private var imageOutputType: DataType = DataType.FLOAT32
    private var imageQuantScale: Float = 1f
    private var imageQuantZeroPoint: Int = 0

    // Per-sample optional tracking
    private val perSampleTimes = mutableListOf<Double>()
    private val perSamplePreds = mutableListOf<String>()
    private val perSampleGT = mutableListOf<String>()

    fun run(): PipelineBenchmarkResult {
        Log.i(TAG, "Starting pipeline benchmark. modelAssetPath=$modelAssetPath, handLandmarker=$handLandmarkerAssetPath, mpDelegate=$mpDelegate, tfDelegate=$tfDelegate")

        // Load labels
        labels = try {
            Log.d(TAG, "Loading labels from assets: $labelsAssetPath")
            LabelLoader.loadLabels(context, labelsAssetPath)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load labels: ${e.message}. Proceeding with empty labels.", e)
            emptyList()
        }

        // Interpreter + tensor metadata
        try {
            interpreter = createInterpreter().also {
                inputTensor = it.getInputTensor(0)
                outputTensor = it.getOutputTensor(0)
                Log.i(TAG, "Interpreter created. Input tensor shape=${inputTensor.shape().contentToString()} type=${inputTensor.dataType()} Output shape=${outputTensor.shape().contentToString()} type=${outputTensor.dataType()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create Interpreter — aborting benchmark", e)
            throw e
        }

        try {
            determineInputMode()
            Log.i(TAG, "Determined inputMode=$inputMode")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to determine input mode", e)
            throw e
        }

        if (inputMode == InputMode.LANDMARKS_126) {
            // Use the safe creator that posts to main thread and reports errors
            landmarker = createHandLandmarkerSafe()
            if (landmarker == null) {
                // If handlandmarker couldn't be created, abort gracefully with clear log
                Log.e(TAG, "HandLandmarker creation failed. Aborting benchmark to avoid native crash.")
                throw IllegalStateException("HandLandmarker creation failed. Check assets, MediaPipe native libs and ABI packaging.")
            }
            Log.i(TAG, "HandLandmarker created successfully")
            setupLandmarkBuffers()
            Log.d(TAG, "Landmark buffers initialized. Input len=${landmarkInput[0].size}, output len=${landmarkOutput[0].size}")
        } else {
            setupImageBuffers()
            Log.d(TAG, "Image buffers initialized. inputShape=${imageInputShape.contentToString()} outputShape=${imageOutputShape.contentToString()} inputType=${imageInputType} outputType=${imageOutputType} quantScale=$imageQuantScale zeroPoint=$imageQuantZeroPoint")
        }

        // Dataset enumeration
        val items = enumerateDataset(context.contentResolver, datasetTreeUri)
        Log.i(TAG, "Dataset enumerated. total items=${items.size}")
        val totalPlanned = totalRunsLimit?.coerceAtMost(items.size) ?: items.size
        if (totalPlanned == 0) {
            Log.e(TAG, "No images found under selected folder: $datasetTreeUri")
            throw IllegalStateException("No images found under selected folder")
        }

        // Warmup
        try {
            Log.i(TAG, "Running warmup: $warmupRuns runs")
            runWarmup(items)
        } catch (e: Exception) {
            Log.w(TAG, "Warmup encountered an error", e)
        }

        val times = ArrayList<Double>(totalPlanned)
        var correct = 0
        var noDetections = 0
        var processed = 0

        for ((imgUri, gtLabel) in items) {
            if (processed >= totalPlanned) break

            val bmp = try {
                decodeBitmap(context.contentResolver, imgUri)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to decode bitmap for $imgUri — skipping", e)
                null
            }
            if (bmp == null) {
                Log.w(TAG, "Bitmap is null for $imgUri — skipping")
                continue
            }

            val (predictionIdx, elapsedMs, hadDetection) = try {
                if (inputMode == InputMode.LANDMARKS_126) runLandmarkPipeline(bmp) else runImagePipeline(bmp)
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline failed for image $imgUri", e)
                // skip this sample
                continue
            }

            times.add(elapsedMs)

            val predLabel = predictionIdx?.let { labels.getOrNull(it) } ?: "NO_DETECTION"
            if (!hadDetection) noDetections++
            if (gtLabel == predLabel) correct++

            if (recordPerSample) {
                perSampleTimes.add(elapsedMs)
                perSamplePreds.add(predLabel)
                perSampleGT.add(gtLabel)
                if (verbosePerSampleLogs) {
                    Log.d(TAG, "sample #$processed gt='$gtLabel' pred='$predLabel' time=${"%.3f".format(elapsedMs)}ms detected=$hadDetection")
                }
            }

            processed++
        }

        if (times.isEmpty()) {
            Log.e(TAG, "No timings captured — aborting result creation")
            throw IllegalStateException("No timings captured")
        }

        val stats = computeStats(times)
        val fps = if (stats.meanMs > 0) 1000.0 / stats.meanMs else 0.0
        val accuracy = if (processed > 0) correct.toDouble() / processed else 0.0

        val result = PipelineBenchmarkResult(
            deviceName = DeviceInfo.getDeviceName(),
            mediapipeDelegate = mpDelegate.name,
            modelDelegate = tfDelegate.name,
            inputMode = inputMode.name,
            meanMs = stats.meanMs,
            medianMs = stats.medianMs,
            stdMs = stats.stdMs,
            p95Ms = stats.p95Ms,
            fps = fps,
            sampleCount = times.size,
            accuracy = accuracy,
            totalCorrect = correct,
            totalSamples = processed,
            noDetections = noDetections,
            executionInfo = "InputMode=$inputMode; Samples=$processed"
        )

        Log.i(TAG, "Benchmark finished. mean=${"%.3f".format(result.meanMs)}ms median=${"%.3f".format(result.medianMs)}ms p95=${"%.3f".format(result.p95Ms)}ms fps=${"%.2f".format(result.fps)} accuracy=${"%.3f".format(result.accuracy)} samples=${result.sampleCount} noDetections=${result.noDetections}")

        // Cleanup
        try { landmarker?.close(); Log.d(TAG, "HandLandmarker closed") } catch (e: Throwable) { Log.w(TAG, "Error closing HandLandmarker", e) }
        try { interpreter?.close(); Log.d(TAG, "Interpreter closed") } catch (e: Throwable) { Log.w(TAG, "Error closing Interpreter", e) }
        try { gpuDelegate?.close(); Log.d(TAG, "GpuDelegate closed") } catch (e: Throwable) { Log.w(TAG, "Error closing GpuDelegate", e) }
        if (recordPerSample) {
            try {
                savePerSampleCsv(result)
                Log.d(TAG, "Per-sample CSV saved")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to save per-sample CSV", e)
            }
        }
        return result
    }

    fun saveResultToJson(result: PipelineBenchmarkResult) {
        try {
            val dir = File(context.getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS), "BenchmarkResults")
            dir.mkdirs()
            val file = File(dir, "pipeline_${result.inputMode}_${result.mediapipeDelegate}_${result.modelDelegate}_${System.currentTimeMillis()}.json")
            FileWriter(file).use { it.write(Gson().toJson(result)) }
            Log.i(TAG, "Result JSON saved: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save result JSON", e)
        }
    }

    // ---------------------- Setup ----------------------

    private fun createInterpreter(): Interpreter {
        val opts = Interpreter.Options()
        Log.d(TAG, "Creating interpreter with tfDelegate=$tfDelegate")
        when (tfDelegate) {
            TfDelegate.GPU -> {
                try {
                    gpuDelegate = GpuDelegate()
                    opts.addDelegate(gpuDelegate)
                    Log.i(TAG, "Added GpuDelegate to Interpreter options")
                } catch (e: Throwable) {
                    Log.w(TAG, "Failed to add GpuDelegate - falling back to CPU", e)
                }
            }
            TfDelegate.NNAPI -> {
                try {
                    opts.setUseNNAPI(true)
                    Log.i(TAG, "NNAPI requested in Interpreter options")
                } catch (e: Throwable) {
                    Log.w(TAG, "Failed to enable NNAPI - falling back to CPU", e)
                }
            }
            else -> { Log.d(TAG, "Using CPU interpreter") }
        }
        val model = try {
            loadModelFromAssets(modelAssetPath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from assets: $modelAssetPath", e)
            throw e
        }
        return Interpreter(model, opts)
    }

    private fun determineInputMode() {
        val shape = inputTensor.shape() // e.g. [1,126] or [1,H,W,C]
        Log.d(TAG, "Input tensor shape: ${shape.contentToString()}")
        if (shape.size == 2 && shape[1] == 126) {
            inputMode = InputMode.LANDMARKS_126
        } else if (shape.size == 4) {
            inputMode = InputMode.IMAGE_4D
            imageInputShape = shape
            imageOutputShape = outputTensor.shape()
            imageInputType = inputTensor.dataType()
            imageOutputType = outputTensor.dataType()
            if (imageInputType == DataType.UINT8 || imageInputType == DataType.INT8) {
                imageQuantScale = inputTensor.quantizationParams().scale
                imageQuantZeroPoint = inputTensor.quantizationParams().zeroPoint
            }
        } else {
            throw IllegalStateException("Unsupported model input shape: ${shape.contentToString()} rank=${shape.size}")
        }
    }

    /**
     * Safe creation of HandLandmarker: checks asset presence and creates on UI thread,
     * returning null on failure instead of letting native code crash the app.
     */
    private fun createHandLandmarkerSafe(): HandLandmarker? {
        try {
            // quick existence check for the asset
            try {
                context.assets.open(handLandmarkerAssetPath).close()
                Log.d(TAG, "hand_landmarker.task found in assets")
            } catch (e: Exception) {
                Log.e(TAG, "hand_landmarker.task not found or unreadable: $handLandmarkerAssetPath", e)
                return null
            }

            val latch = CountDownLatch(1)
            var created: HandLandmarker? = null
            var creationEx: Throwable? = null

            // create on main thread (some MediaPipe native init expects UI thread)
            Handler(Looper.getMainLooper()).post {
                try {
                    val baseOptions = BaseOptions.builder()
                        .setModelAssetPath(handLandmarkerAssetPath)
                        .setDelegate(if (mpDelegate == MpDelegate.GPU) Delegate.GPU else Delegate.CPU)
                        .build()
                    val options = HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setNumHands(2)
                        .setMinHandDetectionConfidence(0.5f)
                        .setMinHandPresenceConfidence(0.5f)
                        .setMinTrackingConfidence(0.5f)
                        .build()
                    created = HandLandmarker.createFromOptions(context, options)
                    Log.i(TAG, "HandLandmarker.createFromOptions succeeded on main thread")
                } catch (t: Throwable) {
                    creationEx = t
                    Log.e(TAG, "HandLandmarker.createFromOptions threw", t)
                } finally {
                    latch.countDown()
                }
            }

            // Wait up to 5s for creation to avoid deadlocks; adjust if needed
            val ok = latch.await(5, TimeUnit.SECONDS)
            if (!ok) {
                Log.e(TAG, "Timed out waiting for HandLandmarker creation on UI thread")
                return null
            }
            if (creationEx != null) {
                Log.e(TAG, "Exception while creating HandLandmarker", creationEx)
                return null
            }
            return created
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error in createHandLandmarkerSafe", e)
            return null
        }
    }

    private fun setupLandmarkBuffers() {
        landmarkInput = Array(1) { FloatArray(126) }
        val numClasses = outputTensor.shape().last()
        landmarkOutput = Array(1) { FloatArray(numClasses) }
    }

    private fun setupImageBuffers() {
        val inBytesPerElem = if (imageInputType == DataType.UINT8 || imageInputType == DataType.INT8) 1 else 4
        val inSize = imageInputShape.reduce { acc, i -> acc * i }
        imageInputBuffer = ByteBuffer.allocateDirect(inSize * inBytesPerElem).order(ByteOrder.nativeOrder())

        val outBytesPerElem = if (imageOutputType == DataType.UINT8 || imageOutputType == DataType.INT8) 1 else 4
        val outSize = imageOutputShape.reduce { acc, i -> acc * i }
        imageOutputBuffer = ByteBuffer.allocateDirect(outSize * outBytesPerElem).order(ByteOrder.nativeOrder())
    }

    // ---------------------- Warmup ----------------------

    private fun runWarmup(items: List<Pair<Uri, String>>) {
        if (items.isEmpty() || warmupRuns <= 0) {
            Log.d(TAG, "Skipping warmup: items=${items.size} warmupRuns=$warmupRuns")
            return
        }
        var done = 0
        for ((uri, _) in items) {
            if (done >= warmupRuns) break
            val bmp = decodeBitmap(context.contentResolver, uri) ?: continue
            try {
                if (inputMode == InputMode.LANDMARKS_126) runLandmarkPipeline(bmp) else runImagePipeline(bmp)
                Log.d(TAG, "Warmup run #${done + 1} completed")
            } catch (e: Exception) {
                Log.w(TAG, "Warmup run failed for $uri", e)
            }
            done++
        }
    }

    // ---------------------- Pipelines ----------------------

    private fun runLandmarkPipeline(bitmap: Bitmap): Triple<Int?, Double, Boolean> {
        // landmarker is guaranteed non-null at this point due to earlier checks,
        // but we still coerce-check to avoid NPEs.
        if (landmarker == null) {
            Log.e(TAG, "runLandmarkPipeline called but landmarker == null")
            return Triple(null, 0.0, false)
        }

        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()
        var predIdx: Int? = null
        var hadDetection = false
        val elapsedMs = measureNanoTime {
            try {
                val result = landmarker!!.detect(mpImage)
                val features = extractFeatures(result)
                hadDetection = result.landmarks()?.isNotEmpty() == true
                landmarkInput[0] = features
                interpreter!!.run(landmarkInput, landmarkOutput)
                val out = landmarkOutput[0]
                var best = out[0]; var bestIdx = 0
                for (i in 1 until out.size) if (out[i] > best) { best = out[i]; bestIdx = i }
                predIdx = if (hadDetection) bestIdx else null
            } catch (e: Exception) {
                Log.e(TAG, "Error during landmark pipeline", e)
                throw e
            }
        } / 1e6
        if (verbosePerSampleLogs) Log.d(TAG, "Landmark pipeline took ${"%.3f".format(elapsedMs)}ms predIdx=$predIdx hadDetection=$hadDetection")
        return Triple(predIdx, elapsedMs, hadDetection)
    }

    private fun runImagePipeline(bitmap: Bitmap): Triple<Int?, Double, Boolean> {
        val (_, h, w, _) = imageInputShape // shape is [1,H,W,C]
        val resized = Bitmap.createScaledBitmap(bitmap, w, h, true)
        val buf = imageInputBuffer!!
        buf.rewind()
        try {
            if (imageInputType == DataType.FLOAT32) {
                val pixels = IntArray(w * h)
                resized.getPixels(pixels, 0, w, 0, 0, w, h)
                var idx = 0
                for (py in 0 until h) {
                    for (px in 0 until w) {
                        val argb = pixels[idx++]
                        val r = ((argb shr 16) and 0xFF) / 255f
                        val g = ((argb shr 8) and 0xFF) / 255f
                        val bch = (argb and 0xFF) / 255f
                        buf.putFloat(r); buf.putFloat(g); buf.putFloat(bch)
                    }
                }
            } else { // UINT8 / INT8
                val pixels = IntArray(w * h)
                resized.getPixels(pixels, 0, w, 0, 0, w, h)
                var idx = 0
                for (py in 0 until h) {
                    for (px in 0 until w) {
                        val argb = pixels[idx++]
                        val r = (argb shr 16) and 0xFF
                        val g = (argb shr 8) and 0xFF
                        val bch = argb and 0xFF
                        buf.put(r.toByte()); buf.put(g.toByte()); buf.put(bch.toByte())
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing image buffer", e)
            throw e
        }

        var predIdx: Int? = null
        val elapsedMs = measureNanoTime {
            try {
                interpreter!!.run(buf, imageOutputBuffer!!)
                val outBuf = imageOutputBuffer!!
                outBuf.rewind()
                val outSize = imageOutputShape.reduce { acc, i -> acc * i }
                var bestIdx = 0
                var bestVal = readOutputValue(outBuf, 0)
                for (i in 1 until outSize) {
                    val v = readOutputValue(outBuf, i)
                    if (v > bestVal) { bestVal = v; bestIdx = i }
                }
                predIdx = bestIdx
            } catch (e: Exception) {
                Log.e(TAG, "Interpreter run failed in image pipeline", e)
                throw e
            }
        } / 1e6

        if (verbosePerSampleLogs) Log.d(TAG, "Image pipeline took ${"%.3f".format(elapsedMs)}ms predIdx=$predIdx")
        return Triple(predIdx, elapsedMs, true)
    }

    private fun readOutputValue(buf: ByteBuffer, index: Int): Float {
        // We rewind earlier; read sequentially
        return if (imageOutputType == DataType.FLOAT32) {
            buf.float
        } else {
            (buf.get().toInt() and 0xFF).toFloat()
        }
    }

    // ---------------------- Landmark Feature Extraction ----------------------

    private fun extractFeatures(result: HandLandmarkerResult): FloatArray {
        val data = FloatArray(126) { 0f }
        val hands: List<List<NormalizedLandmark>> = result.landmarks() ?: emptyList()
        val handednessLists: List<List<Category>>? = result.handednesses()

        for (i in hands.indices) {
            if (i >= 2) break
            val lmList = hands[i]
            val label = handednessLists?.getOrNull(i)?.getOrNull(0)?.categoryName()
                ?.trim()?.lowercase() ?: if (i == 0) "left" else "right"
            val offset = if (label == "left") 0 else 63

            var j = 0
            for (lm in lmList) {
                if (j >= 21) break
                val base = offset + j * 3
                data[base] = lm.x()
                data[base + 1] = lm.y()
                data[base + 2] = lm.z()
                j++
            }
        }
        return data
    }

    // ---------------------- Stats & Helpers ----------------------

    private data class Stats(val meanMs: Double, val medianMs: Double, val stdMs: Double, val p95Ms: Double)
    private fun computeStats(times: List<Double>): Stats {
        val sorted = times.sorted()
        val mean = times.average()
        val median = sorted[sorted.size / 2]
        val std = sqrt(times.map { (it - mean).pow(2) }.average())
        val p95Index = (sorted.size * 0.95).toInt().coerceAtMost(sorted.size - 1)
        val p95 = sorted[p95Index]
        return Stats(mean, median, std, p95)
    }

    private fun loadModelFromAssets(assetPath: String): MappedByteBuffer {
        Log.d(TAG, "Loading model from assets: $assetPath")
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    private fun enumerateDataset(resolver: ContentResolver, treeUri: Uri): List<Pair<Uri, String>> {
        Log.d(TAG, "Enumerating dataset at $treeUri")
        val root = DocumentFile.fromTreeUri(context, treeUri) ?: run {
            Log.w(TAG, "DocumentFile.fromTreeUri returned null for $treeUri")
            return emptyList()
        }
        val results = mutableListOf<Pair<Uri, String>>()
        val children = root.listFiles().toList()

        fun isImage(f: DocumentFile): Boolean {
            if (!f.isFile) return false
            val n = (f.name ?: "").lowercase()
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png") || n.endsWith(".webp")
        }

        val rootImages = children.filter { isImage(it) }
        val labelDirs = children.filter { it.isDirectory }

        if (rootImages.isNotEmpty()) {
            val label = root.name ?: "root"
            rootImages.forEach { results.add(it.uri to label) }
            Log.d(TAG, "Found ${rootImages.size} images in root folder (label=$label)")
        }

        if (labelDirs.isNotEmpty()) {
            labelDirs.forEach { dir ->
                val label = dir.name ?: return@forEach
                val imgs = dir.listFiles().filter { isImage(it) }
                if (imgs.isNotEmpty()) {
                    imgs.forEach { results.add(it.uri to label) }
                    Log.d(TAG, "Found ${imgs.size} images in label folder '$label'")
                } else {
                    dir.listFiles().filter { it.isDirectory }.forEach { sub ->
                        val subImgs = sub.listFiles().filter { isImage(it) }
                        subImgs.forEach { results.add(it.uri to label) }
                        if (subImgs.isNotEmpty()) Log.d(TAG, "Found ${subImgs.size} images in subfolder of '$label'")
                    }
                }
            }
        }
        Log.d(TAG, "Enumerate finished: total ${results.size}")
        return results
    }

    private fun decodeBitmap(resolver: ContentResolver, uri: Uri): Bitmap? =
        try {
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to decode stream for $uri", e)
            null
        }

    private fun savePerSampleCsv(result: PipelineBenchmarkResult) {
        val dir = File(context.getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS), "BenchmarkResults")
        dir.mkdirs()
        val file = File(dir, "samples_${result.inputMode}_${System.currentTimeMillis()}.csv")
        FileWriter(file).use { fw ->
            fw.appendLine("index,ground_truth,prediction,inference_time_ms")
            for (i in perSampleTimes.indices) {
                fw.appendLine("$i,${perSampleGT[i]},${perSamplePreds[i]},${"%.4f".format(perSampleTimes[i])}")
            }
        }
        Log.i(TAG, "Per-sample CSV written to: ${file.absolutePath}")
    }
}
