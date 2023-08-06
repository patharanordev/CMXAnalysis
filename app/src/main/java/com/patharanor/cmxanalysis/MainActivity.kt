package com.patharanor.cmxanalysis

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.patharanor.cmxanalysis.analyzer.BoundingBox
import com.patharanor.cmxanalysis.analyzer.LuminosityAnalyzer
import com.patharanor.cmxanalysis.analyzer.ObjectDetector
import com.patharanor.cmxanalysis.analyzer.SegmentAnyting
import com.patharanor.cmxanalysis.databinding.ActivityMainBinding
import com.patharanor.cmxanalysis.utils.CameraCapture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val TAG = "CMXAnalyzer"
    private lateinit var viewBinding : ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var outputImage: ImageView
    private var imageCapture: ImageCapture? = null
    private var cameraCapture: CameraCapture? = null
    private var CURRENT_ANALYZER = "OBJECT_DETECTION" // "SEGMENT_ANYTHING" //
    private var bboxes: Array<BoundingBox> = emptyArray()
    private var activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        outputImage = findViewById(R.id.imageView2)
        if (outputImage == null) {
            Log.d(TAG, "outputImage is null")
        }

        if (allPermissionsGranted()) {
            cameraCapture = CameraCapture(baseContext)
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo buttons
        viewBinding.imageCaptureButton.setOnClickListener { cameraCapture?.takePhoto(this.bboxes) }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA


            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(viewBinding.viewFinder.width, viewBinding.viewFinder.height))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    when (CURRENT_ANALYZER) {
                        "SEGMENT_ANYTHING" -> {
                            val modelID = R.raw.yolov8n_seg
                            val classes: List<String> = readClasses();
                            val modelByte: ByteArray = readModel(modelID);
                            val segmentAnyting = SegmentAnyting{ bitmap ->
                                Log.d(TAG, "Segment anything : $bitmap")

                                try {
                                    outputImage.setImageBitmap(bitmap)
                                } catch (e: java.lang.Exception) {
                                    Log.e(TAG, "Set bitmap error : $e.message")
                                }
                            }

                            segmentAnyting.init(modelByte, classes, outputImage)
                            segmentAnyting.setDebug(true)

                            it.setAnalyzer(cameraExecutor, segmentAnyting)
                        }
                        "OBJECT_DETECTION" -> {
                            val modelID = R.raw.yolov8n_with_pre_post_processing
                            val classes: List<String> = readClasses();
                            val modelByte: ByteArray = readModel(modelID);
                            val objectDetector = ObjectDetector{ bitmap, bboxes ->
                                this.bboxes = bboxes
                                Log.d(TAG, "Object detection : $bitmap")

                                try {
                                    outputImage.setImageBitmap(bitmap)
                                } catch (e: java.lang.Exception) {
                                    Log.e(TAG, "Set bitmap error : $e.message")
                                }
                            }

                            objectDetector.init(modelByte, classes, outputImage)
                            objectDetector.setDebug(false)

                            it.setAnalyzer(cameraExecutor, objectDetector)
                        }
                        "LUMA" -> {
                            it.setAnalyzer(cameraExecutor, LuminosityAnalyzer { luma ->
                                Log.d(TAG, "Average luminosity : $luma")
                            })
                        }
                        else -> {
                            Log.d(TAG, "Unknown analyzer.")
                        }
                    }
                }

            imageCapture = cameraCapture?.build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun readModel(modelId: Int): ByteArray {
        Log.d(TAG, "Found model ID : $modelId")
        return resources.openRawResource(modelId).readBytes()
    }

    private fun readClasses(): List<String> {
        val classesID = R.raw.classes
        Log.d(TAG, "Found classes ID : $classesID")
        return resources.openRawResource(classesID).bufferedReader().readLines()
    }

    companion object {
        private const val TAG = "CMXAnalysis"
        // private const val FILENAME_FORMAT = "yyyy-MM-dd HH:mm:ss.SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
}