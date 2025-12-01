package com.example.benchmarking

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DelegateFactory
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

// Assuming DeviceInfo and LabelLoader are defined elsewhere

data class BenchmarkResult(
    val deviceName: String,
    val delegate: String,
    val meanMs: Double,
    val medianMs: Double,
    val stdMs: Double,
    val p95Ms: Double,
    val fps: Double,
    val sampleCount: Int,
    val labels: List<String>,
    val executionInfo: String
)

class BenchmarkRunner(
    private val context: Context,
    private val modelPath: String,
    private val delegateType: String
) {
    companion object {
        private const val TAG = "ModelBenchmark"
    }

    fun runBenchmark(): BenchmarkResult {
        Log.i(TAG, "Starting BenchmarkRunner modelPath=$modelPath delegateType=$delegateType")

        var interpreter: Interpreter? = null
        var gpuDelegate: GpuDelegate? = null
        var executionInfo = "Unknown"
        var delegateUsed = delegateType.uppercase()
        val totalRuns = 30

        val labels = try {
            Log.d(TAG, "Loading labels labels.txt")
            LabelLoader.loadLabels(context, "labels.txt")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels", e)
            emptyList<String>()
        }

        val modelBuffer = try {
            loadModelFile(modelPath).also { Log.d(TAG, "Model loaded OK (bytes=${it.capacity()})") }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $modelPath", e)
            throw RuntimeException("Model load failed: ${e.message}")
        }

        try {
            val options = Interpreter.Options()

            when (delegateType.uppercase()) {
                "GPU" -> {
                    try {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                        Log.i(TAG, "Using direct TFLite GPU delegate")
                        executionInfo = "GPU (TFLite)"
                        delegateUsed = "GPU (TFLite)"
                    } catch (t: Throwable) {
                        Log.w(TAG, "Direct GPU delegate creation failed", t)
                        delegateUsed = "CPU (GPU fallback)"
                        executionInfo = "GPU delegate creation failed - using CPU"
                        try { gpuDelegate?.close() } catch (_: Throwable) {}
                    }
                }

                "NNAPI" -> {
                    try {
                        options.setUseNNAPI(true)
                        Log.i(TAG, "Requested NNAPI delegate")
                        executionInfo = "NNAPI"
                        delegateUsed = "NNAPI"
                    } catch (e: Exception) {
                        Log.w(TAG, "NNAPI not available", e)
                        delegateUsed = "CPU (NNAPI unavailable)"
                        executionInfo = "NNAPI unavailable - using CPU"
                    }
                }

                else -> {
                    Log.i(TAG, "Using CPU interpreter")
                    delegateUsed = "CPU"
                    executionInfo = "CPU"
                }
            }

            interpreter = try {
                Interpreter(modelBuffer, options)
            } catch (e: Exception) {
                Log.e(TAG, "Interpreter creation failed", e)
                throw RuntimeException("Interpreter creation failed: ${e.message}")
            }

            val inputShape = try {
                interpreter!!.getInputTensor(0).shape()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read input tensor shape", e)
                throw e
            }
            val inputSize = inputShape.reduce { acc, i -> acc * i }
            Log.d(TAG, "Input shape=${inputShape.contentToString()} inputSize=$inputSize")

            val inputBuffer = ByteBuffer.allocateDirect(inputSize * 4).order(ByteOrder.nativeOrder())

            val outputShape = interpreter!!.getOutputTensor(0).shape()
            val outputSize = outputShape.reduce { acc, i -> acc * i }
            Log.d(TAG, "Output shape=${outputShape.contentToString()} outputSize=$outputSize")
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4).order(ByteOrder.nativeOrder())

            val times = mutableListOf<Double>()
            val warmupRuns = 5

            Log.i(TAG, "Starting warmup runs ($warmupRuns)")
            repeat(warmupRuns) {
                try {
                    inputBuffer.rewind()
                    outputBuffer.rewind()
                    interpreter!!.run(inputBuffer, outputBuffer)
                } catch (e: Exception) {
                    Log.w(TAG, "Warmup run failed", e)
                }
            }

            Log.i(TAG, "Starting benchmark runs ($totalRuns)")
            repeat(totalRuns) { i ->
                inputBuffer.rewind()
                outputBuffer.rewind()
                val t = try {
                    measureNanoTime {
                        interpreter!!.run(inputBuffer, outputBuffer)
                    } / 1e6
                } catch (e: Exception) {
                    Log.e(TAG, "Interpreter run failed on iteration $i", e)
                    Double.NaN
                }
                if (!t.isNaN()) times.add(t)
                if (i % 5 == 0) Log.d(TAG, "Run #$i time=${"%.3f".format(t)}ms")
            }

            if (times.isEmpty()) throw RuntimeException("Benchmark produced no valid timings.")

            val mean = times.average()
            val median = times.sorted()[times.size / 2]
            val std = sqrt(times.map { (it - mean).pow(2) }.average())
            val p95 = times.sorted()[(times.size * 0.95).toInt().coerceAtMost(times.size - 1)]
            val fps = if (mean > 0) 1000.0 / mean else 0.0

            Log.i(TAG, "Benchmark finished mean=${"%.3f".format(mean)}ms fps=${"%.2f".format(fps)}")

            return BenchmarkResult(
                deviceName = DeviceInfo.getDeviceName(),
                delegate = delegateUsed,
                meanMs = mean,
                medianMs = median,
                stdMs = std,
                p95Ms = p95,
                fps = fps,
                sampleCount = totalRuns,
                labels = labels,
                executionInfo = executionInfo
            )

        } finally {
            try { interpreter?.close(); Log.d(TAG, "Interpreter closed") } catch (e: Exception) { Log.w(TAG, "Error closing interpreter", e) }
            try { gpuDelegate?.close(); Log.d(TAG, "GpuDelegate closed") } catch (e: Exception) { Log.w(TAG, "Error closing GPU delegate", e) }
        }
    }

    private fun loadModelFile(modelPath: String): ByteBuffer {
        Log.d(TAG, "Loading model asset: $modelPath")
        val asset = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(asset.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = asset.startOffset
        val declaredLength = asset.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
