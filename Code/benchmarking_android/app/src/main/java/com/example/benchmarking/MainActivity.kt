package com.example.benchmarking

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import java.io.File
import java.io.FileWriter

class MainActivity : ComponentActivity() {

    private lateinit var resultText: TextView
    private lateinit var runButton: Button
    private lateinit var pickDatasetButton: Button
    private lateinit var mpSpinner: Spinner
    private lateinit var tfSpinner: Spinner

    private val mpDelegates = listOf("CPU", "GPU")
    private val tfDelegates = listOf("CPU", "GPU", "NNAPI")
    private var datasetTreeUri: Uri? = null

    private val pickDatasetLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            )
            datasetTreeUri = uri
            Toast.makeText(this, "Dataset selected.", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "No dataset chosen.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.resultText)
        runButton = findViewById(R.id.runButton)
        pickDatasetButton = findViewById(R.id.pickDatasetButton)
        mpSpinner = findViewById(R.id.mpDelegateSpinner)
        tfSpinner = findViewById(R.id.tfDelegateSpinner)

        mpSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, mpDelegates)
        tfSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, tfDelegates)

        pickDatasetButton.setOnClickListener { pickDatasetLauncher.launch(null) }
        runButton.setOnClickListener { startBenchmark() }
    }

    private fun startBenchmark() {
        val uri = datasetTreeUri
        if (uri == null) {
            Toast.makeText(this, "Select dataset folder first.", Toast.LENGTH_SHORT).show()
            return
        }
        val mp = if (mpSpinner.selectedItem.toString() == "GPU") MpDelegate.GPU else MpDelegate.CPU
        val tf = when (tfSpinner.selectedItem.toString()) {
            "GPU" -> TfDelegate.GPU
            "NNAPI" -> TfDelegate.NNAPI
            else -> TfDelegate.CPU
        }

        resultText.text = "Running benchmark… MediaPipe=$mp Model=$tf"
        runButton.isEnabled = false
        Thread {
            try {
                android.util.Log.d("Benchmark", "Starting benchmark. uri=$datasetTreeUri mp=$mp tf=$tf")
                val runner = PipelineBenchmarkRunner(
                    context = this,
                    datasetTreeUri = uri,
                    modelAssetPath = "2.tflite",
                    labelsAssetPath = "labels.txt",
                    handLandmarkerAssetPath = "hand_landmarker.task",
                    mpDelegate = mp,
                    tfDelegate = tf,
                    warmupRuns = 3,
                    recordPerSample = false
                )
                val result = runner.run()
                runner.saveResultToJson(result)
                runOnUiThread {
                    resultText.text = """
                        ✅ Benchmark Complete
                        Input Mode: ${result.inputMode}
                        Device: ${result.deviceName}
                        MediaPipe: ${result.mediapipeDelegate}
                        Model: ${result.modelDelegate}
                        Mean: ${"%.3f".format(result.meanMs)} ms
                        Median: ${"%.3f".format(result.medianMs)} ms
                        P95: ${"%.3f".format(result.p95Ms)} ms
                        FPS: ${"%.2f".format(result.fps)}
                        Accuracy: ${"%.2f".format(result.accuracy * 100)}%
                        Correct: ${result.totalCorrect}/${result.totalSamples}
                        NO_DETECTION: ${result.noDetections}
                        Saved: /Documents/BenchmarkResults/
                    """.trimIndent()
                        runButton.isEnabled = true
                }
            } catch (e: Throwable) {
                android.util.Log.e("Benchmark", "Benchmark failed", e)
                runOnUiThread {
                    resultText.text = "Exception: ${e::class.java.simpleName}\nMessage: ${e.message ?: "<null>"}"
                    runButton.isEnabled = true
                }
            }
        }.start()
    }
}