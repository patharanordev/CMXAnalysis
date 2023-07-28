package com.patharanor.cmxanalysis.analyzer

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.Log
import android.widget.ImageView
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.common.primitives.Ints
import com.patharanor.cmxanalysis.utils.BitmapUtils
import java.nio.ByteBuffer
import java.util.Collections
import java.util.Random


typealias ObjectDetectionListener = (bitmap: Bitmap) -> Unit

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
) {}

class ObjectDetector(private val listener: ObjectDetectionListener) : ImageAnalysis.Analyzer {

    private val TAG = "CMXAnalyzer"
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var classes:List<String>
    private var overlay: ImageView? = null
    private var isDebug = false

    fun init(modelBytes: ByteArray, classes: List<String>, overlay: ImageView?) {
        this.classes = classes
        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)

        if (overlay != null) {
            this.overlay = overlay
        }

        Log.d(TAG, "Initial object detection")
    }

    @ExperimentalGetImage
    private fun detect(image: ImageProxy, ortEnv: OrtEnvironment, ortSession: OrtSession) {
        // Step 1: convert image into byte array (raw image bytes)
        //val rawImageBytes = inputStream.readBytes()
        val bitmap: Bitmap? = BitmapUtils.getBitmap(image)
        if (bitmap != null) {
            val bWidth = bitmap.width
            val bHeight = bitmap.height
            Log.d(TAG, "width: $bWidth, height: $bHeight");
            // Step 2: get the shape of the byte array and make ort tensor
            val bytes: ByteArray? = BitmapUtils.convertBitmapToByteArray(bitmap)

            if (bytes != null) {
                val shape = longArrayOf(bytes.size.toLong())

                val inputTensor = OnnxTensor.createTensor(
                    ortEnv,
                    ByteBuffer.wrap(bytes),
                    shape,
                    OnnxJavaType.UINT8
                )
                inputTensor.use {
                    // Step 3: call ort inferenceSession run
                    val output = ortSession.run(
                        Collections.singletonMap("image", inputTensor),
                        setOf("image_out", "scaled_box_out_next")
                    )

                    // Step 4: output analysis
                    output.use {
                        val rawOutput = (output?.get(0)?.value) as ByteArray
                        val boxOutput = (output?.get(1)?.value) as Array<FloatArray>
                        val outputImageBitmap = byteArrayToBitmap(rawOutput)

                        // Step 5: set output result
                        var result = Result(outputImageBitmap, boxOutput)
                        updateUI(result, bitmap)
                    }
                }
            }
        }
    }

    fun setDebug(isDebug: Boolean) {
        this.isDebug = isDebug
    }

    private fun updateUI(result: Result, originalCameraImage: Bitmap?) {
        val mutableBitmap: Bitmap

        if (isDebug) {
            mutableBitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            mutableBitmap = Bitmap.createBitmap(
                result.outputBitmap.width,
                result.outputBitmap.height, Bitmap.Config.ARGB_8888
            )
        }

        val canvas = Canvas(mutableBitmap)
        val textPaint = Paint()
        textPaint.color = Color.WHITE // Text Color
        textPaint.textSize = 28f // Text Size
        textPaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER) // Text Overlapping Pattern

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, textPaint)
        var boxit = result.outputBox.iterator()
        var xScale: Float = 1.0f
        var yScale: Float = 1.0f

        if (overlay != null) {
            xScale = overlay!!.scaleX
            yScale = overlay!!.scaleY
        }

        while(boxit.hasNext()) {
            var box_info = boxit.next()

            // out => x0,y0,x1,y1,score,cls_id
            var x = box_info[0]
            var y = box_info[1]
            var w = box_info[2]
            var h = box_info[3]
            var score = box_info[4]
            var cls_id = box_info[5]

            var left = ((x - w/2))
            var top = ((y - h/2))
            var width = ((left+w) * xScale)
            var height = ((top+h) * yScale)

            canvas.drawText("%s:%.2f".format(classes[cls_id.toInt()], score),
                x-w/2+8, y-h/2+32, textPaint)

            val boxPaint = Paint()
            val pixel = result.outputBitmap.getPixel(Math.round(width/2), Math.round(height/2)) * cls_id.toInt() / 255

            // border
            boxPaint.strokeWidth = 6f
            boxPaint.style = Paint.Style.STROKE

            //boxPaint.setARGB(255, 255, 255 - pixel, 255 - pixel)
            boxPaint.setARGB(255, 255, pixel, 255 - pixel)

            canvas.drawRect(left, top, (width), (height), boxPaint)
        }

        listener(mutableBitmap)
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }

    @ExperimentalGetImage
    override fun analyze(image: ImageProxy) {
        detect(image, ortEnv, ortSession)
        image.close()
    }
}