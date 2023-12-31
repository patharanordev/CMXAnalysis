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
import android.os.SystemClock
import android.util.Log
import android.widget.ImageView
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.patharanor.cmxanalysis.utils.BitmapUtils
import java.nio.ByteBuffer
import java.util.Collections
import kotlin.math.roundToInt


typealias ObjectDetectionListener = (bitmap: Bitmap) -> Unit

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>,
    var inferenceTime: Long,
) {}

class ObjectDetector(private val listener: ObjectDetectionListener) : ImageAnalysis.Analyzer {

    private val TAG = "CMXAnalyzer"
    private val DEFAUL_MAX_RESOLUTION_TRAIN_IMAGE = 640

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var classes:List<String>
    private var overlay: ImageView? = null
    private var isDebug = false

    // To support slow inference in the large screen/resolution
    // Ex. screen size 8XX x 1,9XX
    private var RESIZING_BITMAP_COMPUTATION = 1

    fun init(modelBytes: ByteArray, classes: List<String>, overlay: ImageView?) {
        this.classes = classes
        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        var providerOptions = mutableMapOf<String, String>()

        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
        sessionOptions.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0")

        providerOptions["intra_op_num_threads"] = "1"
        sessionOptions.addXnnpack(providerOptions)
        sessionOptions.setIntraOpNumThreads(1)

        ortSession = ortEnv.createSession(modelBytes, sessionOptions)

        if (overlay != null) {
            this.overlay = overlay
            this.autoScaleImageIO(this.overlay)
        }

        Log.d(TAG, "Initialized object detection")
    }

    @ExperimentalGetImage
    private fun detect(image: ImageProxy, ortEnv: OrtEnvironment, ortSession: OrtSession) {
        // Step 1: convert image into byte array (raw image bytes)
        //val rawImageBytes = inputStream.readBytes()
        val startInference = SystemClock.uptimeMillis()
        var bitmap: Bitmap? = BitmapUtils.getBitmap(image)

        if (bitmap != null) {
            bitmap = BitmapUtils.getResizedBitmap(bitmap, bitmap.width / RESIZING_BITMAP_COMPUTATION, bitmap.height / RESIZING_BITMAP_COMPUTATION)
        }

        if (bitmap != null) {
            val bWidth = bitmap.width
            val bHeight = bitmap.height
            Log.d(TAG, "width: $bWidth, height: $bHeight")
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
                    val output: OrtSession.Result = ortSession.run(
                        Collections.singletonMap("image", inputTensor),
                        setOf("image_out", "scaled_box_out_next")
                    )

                    // Step 4: output analysis
                    output.use {
                        val rawOutput = (output.get(0)?.value) as ByteArray
                        val boxOutput = (output.get(1)?.value) as Array<FloatArray>
                        val outputImageBitmap = byteArrayToBitmap(rawOutput)
                        val finishInference = SystemClock.uptimeMillis() - startInference

                        // Step 5: set output result
                        val result = Result(outputImageBitmap, boxOutput, finishInference)
                        updateUI(result)
                    }
                }
            }
        }
    }

    private fun autoScaleImageIO(overlay: ImageView?) {
        if (overlay != null) {
            if (overlay.width > DEFAUL_MAX_RESOLUTION_TRAIN_IMAGE || overlay.height > DEFAUL_MAX_RESOLUTION_TRAIN_IMAGE) {
                val maxResolution = Math.max(overlay.width, overlay.height)
                RESIZING_BITMAP_COMPUTATION = maxResolution/DEFAUL_MAX_RESOLUTION_TRAIN_IMAGE
                Log.d(TAG, "Auto-scale image IO with ratio $RESIZING_BITMAP_COMPUTATION")
            }
        }
    }

    fun setDebug(isDebug: Boolean) {
        this.isDebug = isDebug
    }

    private fun updateUI(result: Result) {
        var mutableBitmap: Bitmap?

        Log.d(TAG, "Inference time : ${result.inferenceTime}")

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
        textPaint.textSize = 18f // Text Size
        textPaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER) // Text Overlapping Pattern

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, textPaint)
        val boxit = result.outputBox.iterator()
        var xScale = 1.0f
        var yScale = 1.0f

        if (overlay != null) {
            xScale = overlay!!.scaleX
            yScale = overlay!!.scaleY
        }

        while(boxit.hasNext()) {
            val boxInfo = boxit.next()

            // out => x0,y0,x1,y1,score,cls_id
            val x = boxInfo[0]
            val y = boxInfo[1]
            val w = boxInfo[2]
            val h = boxInfo[3]
            val score = boxInfo[4]
            val classId = boxInfo[5]

            val left = ((x - w/2))
            val top = ((y - h/2))
            val width = ((left+w) * xScale)
            val height = ((top+h) * yScale)

            canvas.drawText("%s:%.2f".format(classes[classId.toInt()], score),
                x-w/2+8, y-h/2+32, textPaint)

            val boxPaint = Paint()
            val pixel = result.outputBitmap.getPixel((width / 2).roundToInt(),
                (height / 2).roundToInt()
            ) * classId.toInt() / 255

            // border
            boxPaint.strokeWidth = 3f
            boxPaint.style = Paint.Style.STROKE
            boxPaint.setARGB(255, 255, pixel, 255 - pixel)
            canvas.drawRect(left, top, (width), (height), boxPaint)
        }

        mutableBitmap = BitmapUtils.getResizedBitmap(
            mutableBitmap,
            mutableBitmap.width * RESIZING_BITMAP_COMPUTATION,
            mutableBitmap.height * RESIZING_BITMAP_COMPUTATION)

        if (mutableBitmap != null) {
            listener(mutableBitmap)
        }
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