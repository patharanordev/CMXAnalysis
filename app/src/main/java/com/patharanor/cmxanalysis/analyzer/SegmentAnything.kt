package com.patharanor.cmxanalysis.analyzer
// Ref. https://github.com/Kotlin/kotlindl/blob/master/onnx/src/commonMain/kotlin/org/jetbrains/kotlinx/dl/onnx/inference/OnnxInferenceModel.kt

/**
 * Segmentation task require more memory usage
 * Ref. https://github.com/ultralytics/ultralytics/issues/811#issuecomment-1435752833
 */

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
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
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.TensorLayout
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.impl.preprocessing.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.rotate
import org.jetbrains.kotlinx.dl.impl.preprocessing.toFloatArray
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.append
import org.jetbrains.kotlinx.multik.ndarray.operations.forEach
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.onEach
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toFloatArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.pytorch.Tensor
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.roundToInt


typealias SegmentAnytingListener = (bitmap: Bitmap) -> Unit

/**
 * output0 - contains detected bounding boxes and object classes, the same as for object detection
 * output1 - contains segmentation masks for detected objects. There are only raw masks and no polygons.
 */
internal data class SegmentResult(
//    var output0: Tensor,
//    var output1: Tensor,
//    var inferenceTime: Long,

    var x1: Float,
    var y1: Float,
    var x2: Float,
    var y2: Float,
    var classId: Int,
    var label: String,
    var prob: Float,
    var mask: D1Array<Float>
) {}

class SegmentAnyting(private val listener: SegmentAnytingListener) : ImageAnalysis.Analyzer {

    private val TAG = "CMXAnalyzer"

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var classes: List<String>
    private var overlay: ImageView? = null
    private var isDebug = false

    fun init(modelBytes: ByteArray, classes: List<String>, overlay: ImageView?) {
        this.classes = classes
        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)

        sessionOptions.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0")

        var providerOptions = mutableMapOf<String, String>()
        providerOptions["intra_op_num_threads"] = "1"
        sessionOptions.addXnnpack(providerOptions)

        sessionOptions.setIntraOpNumThreads(1)

        ortSession = ortEnv.createSession(modelBytes, sessionOptions)

        if (overlay != null) {
            this.overlay = overlay
        }

        Log.d(TAG, "Initial object detection")
    }

    fun transpose(matrix: Array<FloatArray>): Array<FloatArray> {
        val row = matrix.size
        val column = matrix[0].size
        val transpose = Array(column) {
            FloatArray(row)
        }
        for (i in 0 until row) {
            for (j in 0 until column) {
                transpose[j][i] = matrix[i][j]
            }
        }
        return transpose
    }

//    fun intersection(box1, box2): Float {
//        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
//        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
//        x1 = max(box1_x1,box2_x1)
//        y1 = max(box1_y1,box2_y1)
//        x2 = min(box1_x2,box2_x2)
//        y2 = min(box1_y2,box2_y2)
//        return (x2-x1)*(y2-y1)
//    }
//
//    fun union(box1,box2): Float {
//        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
//        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
//        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
//        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
//        return box1_area + box2_area - intersection(box1, box2)
//    }
//
//    fun iou(box1,box2): Float {
//        return intersection(box1, box2) / union(box1, box2)
//    }

    fun getMask(row: MultiArray<Float, D1>): D1Array<Float> {
        val mask = row.reshape(160, 160).toFloatArray()
        return mk.ndarray(mask)
    }

    @ExperimentalGetImage
    private fun detect(image: ImageProxy, ortEnv: OrtEnvironment, ortSession: OrtSession) {
        // Step 1: convert image into byte array (raw image bytes)
        //val rawImageBytes = inputStream.readBytes()
        val startInference = SystemClock.uptimeMillis()
//        var bitmap: Bitmap? = BitmapUtils.getBitmap(image)
//
//        if (bitmap != null) {
//            bitmap = BitmapUtils.getResizedBitmap(bitmap, bitmap.width / 2, bitmap.height / 2)
//        }


        val bitmap = image.toBitmap(applyRotation = true)
        val rotation = image.imageInfo.rotationDegrees.toFloat()

        // Need to add crop :
        // Ref. https://github.com/Kotlin/kotlindl-app-sample/blob/main/app/src/main/java/org/jetbrains/kotlinx/dl/example/app/pipelines.kt#L233
        val preprocessing = pipeline<Bitmap>()
            .resize {
                outputHeight = 224 // bitmap.height / 2
                outputWidth = 224 // bitmap.width / 2
            }
            .rotate { degrees = rotation }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(InputType.TORCH.preprocessing(channelsLast = false))
        val (tensor, shape) = preprocessing.apply(bitmap)
        val firstDim: Long = 1
        val dims = LongArray(1 + shape.dims().size)
        dims[0] = 1
        dims[1] = shape.dims()[0]
        dims[2] = shape.dims()[1]
        dims[3] = shape.dims()[2]

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(tensor),
            dims
        )
        inputTensor.use {
            // Step 3: call ort inferenceSession run
            val output: OrtSession.Result = ortSession.run(
                Collections.singletonMap("images", inputTensor),
                setOf("output0", "output1")
            )

            // Step 4: output analysis
            output.use {
                val output0 = (output.get(0)?.value) as Array<Array<FloatArray>>
                val output1 = (output.get(1)?.value) as Array<Array<Array<FloatArray>>>
                //val (_, numClasses, maskHeight, maskWidth) = output1.shape()

                Log.d(TAG, "Shape output0 : (${output0.size}, ${output0[0].size}, ${output0[0][0].size})")
                Log.d(TAG, "Shape output1 : (${output1.size}, ${output1[0].size}, ${output1[0][0].size}, ${output1[0][0][0].size})")

                // Transpose the matrix
                val transposeOutput0: D2Array<Float> = mk.ndarray(this.transpose(output0[0]))
                Log.d(TAG, "Shape transpose output0 : (${transposeOutput0.shape[0]}, ${transposeOutput0.shape[1]})")

                var rawMasks: D3Array<Float> = mk.zeros<Float>(output1[0].size, output1[0][0].size, output1[0][0][0].size)
                output1[0].forEachIndexed { dim1Idx, dim1Arrays ->
                    rawMasks[dim1Idx] = mk.ndarray(dim1Arrays)
                }
                Log.d(TAG, "Shape rawMasks(output1) : (${rawMasks.shape[0]}, ${rawMasks.shape[1]}, ${rawMasks.shape[2]})")

                var boxes: D2Array<Float> = transposeOutput0[0..transposeOutput0.shape[0], 0..84] as D2Array<Float>
                var masks: D2Array<Float> = transposeOutput0[0..transposeOutput0.shape[0], 84..transposeOutput0.shape[1]] as D2Array<Float>
                Log.d(TAG, "Shape boxes : (${boxes.shape[0]}, ${boxes.shape[1]})")
                Log.d(TAG, "Shape masks : (${masks.shape[0]}, ${masks.shape[1]})")

                val reshapedRawMasks: D2Array<Float> = rawMasks.reshape(rawMasks.shape[0], (rawMasks.shape[1]*rawMasks.shape[2]))
                Log.d(TAG, "Shape reshape raw masks : (${reshapedRawMasks.shape[0]}, ${reshapedRawMasks.shape[1]})")

//                val newMasks = multiplyMatrices(masks.toArray(), reshapedRawMasks.toArray(), masks.shape[0], masks.shape[1], reshapedRawMasks.shape[1])
//                Log.d(TAG, "Shape newMasks : (${newMasks[0].size}, ${newMasks[1].size})")

                val newMasks = masks.dot(reshapedRawMasks)
                Log.d(TAG, "Shape newMasks : (${newMasks.shape[0]}, ${newMasks.shape[1]})")

                /**
                 * The hstack function connects two 2D NumPy arrays horizontally
                 * by appending columns from the second array to the right of the first array.
                 *
                 * Finally, for each detected object, we have the following columns:
                 *  - 0-4 - x_center, y_center, width and height of bounding box
                 *  - 4-84 - Object class probabilities for all 80 classes, that this YOLOv8 model can detect
                 *  - 84-25684 - Pixels of segmentation mask as a single row. Actually, the segmentation mask is a 160x160 matrix, but we just flattened it.
                 */
                boxes = boxes.append(newMasks.asD2Array(), axis = 1)
                Log.d(TAG, "Shape new boxes : (${boxes.shape[0]}, ${boxes.shape[1]})")
                Log.d(TAG, "Boxes : ${boxes[0,0..4]}")

                val objects = emptyArray<SegmentResult>()
                boxes.toArray().forEachIndexed { index, floatArr ->
                    val row = mk.ndarray(floatArr)
                    val prob = row[4..84].max()
                    Log.d(TAG, "prob : $prob")
                    if (prob !=null && prob > 0.5f) {
                        val (xc,yc,w,h) = row[0..4].toFloatArray()
                        val classId = row[4..84].argMax()
                        val x1 = (xc-w/2)/224*bitmap.width
                        val y1 = (yc-h/2)/224*bitmap.height
                        val x2 = (xc+w/2)/224*bitmap.width
                        val y2 = (yc+h/2)/224*bitmap.height
                        val label = classes[classId]
                        val mask = mk.ndarray(row[84..row.size].toFloatArray()) // getMask(row[84..row.size])
                        Log.d(TAG, "Class ID : $classId")
                        Log.d(TAG, "label : $label")
                        objects.plus(SegmentResult(x1,y1,x2,y2,classId,label,prob,mask))
                        objects.sortBy { it.prob }
                    }
                }

                objects.forEach {
                    Log.d(TAG, "----------------------------------------------------")
                    Log.d(TAG, "Label: ${it.label}")
                    Log.d(TAG, "Probability: ${it.prob}")
                    Log.d(TAG, "Rect: (${it.x1}, ${it.y1}, ${it.x2}, ${it.y2})")
                    Log.d(TAG, "Mask: ${it.mask}")
                    Log.d(TAG, "----------------------------------------------------")
                }

//                # parse and filter detected objects
//                objects = []
//                for row in boxes:
//                    prob = row[4:84].max()
//                    if prob < 0.5:
//                    continue
//                    xc,yc,w,h = row[:4]
//                    class_id = row[4:84].argmax()
//                    x1 = (xc-w/2)/640*img_width
//                    y1 = (yc-h/2)/640*img_height
//                    x2 = (xc+w/2)/640*img_width
//                    y2 = (yc+h/2)/640*img_height
//                    label = yolo_classes[class_id]
//                    mask = get_mask(row[84:25684],(x1,y1,x2,y2))
//                    objects.append([x1,y1,x2,y2,label,prob,mask])
//
//                    # apply non-maximum suppression to filter duplicated
//                    # boxes
//                    objects.sort(key=lambda x: x[5], reverse=True)
//                    result = []
//                    while len(objects)>0:
//                      result.append(objects[0])
//                      objects = [object for object in objects if iou(object,objects[0])<0.7]
//
//                    print(len(result))







                updateUI(objects, bitmap)
            }
        }
    }

//    private void PrintOut1(List<YoloPrediction> boxes, DenseTensor<float> output, Image image)
//    {
//        var shape = 160;
//        var dir = @"\output";
//        var (w, h) = (image.Width, image.Height); // image w and h
//        var (xGain, yGain) =( shape / (float)w, shape / (float)h); // x, y gains
//
//        var r = output.Reshape(new[] { 1, 32, 160 * 160 });
//        var mask = MultiplyMatrix(r,boxes.Select(m=>m.MaskPrediction).ToArray(), x=> Sigmoid(x) >= 0.5? 1:0);
//
//        var resultImg = ((Bitmap)image).ToImage<Bgr, byte>();
//
//        for (int i = 0; i < mask.GetLength(1); i++)
//        {
//            var btm = new Bitmap(shape, shape);
//            var box = boxes[i].Rectangle;
//            box = new RectangleF(box.X * xGain, box.Y * yGain, box.Width * xGain, box.Height * yGain);
//
//
//            for (int j = 0; j < mask.GetLength(0); j++)
//            {
//                var color = mask[j, i] == 0 ? Color.White : Color.Black;
//                int y = j / shape; // Integer division
//                int x = j % shape;
//                if (box.Contains(x, y))
//                {
//                    btm.SetPixel(x, y, color);
//                }
//                else
//                {
//                    btm.SetPixel(x, y, Color.White);
//                }
//            }
//
//
//            btm.Save($"{dir}\\{i}.jpg");
//
//            var fullMask = btm.ToImage<Gray, byte>().Resize(640, 640, Emgu.CV.CvEnum.Inter.LinearExact);
//            fullMask.Save($"{dir}\\{i}_f.jpg");
//
//            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
//            Mat hier = new Mat();
//
//            CvInvoke.FindContours(fullMask, contours, hier, Emgu.CV.CvEnum.RetrType.Ccomp, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
//            CvInvoke.DrawContours(resultImg, contours, -1, new MCvScalar(255, 0, 0), 2);
//
//            var fullMaskC = fullMask.Convert<Bgr, byte>();
//            CvInvoke.DrawContours(fullMaskC, contours, -1, new MCvScalar(255, 0, 0), 2);
//            fullMaskC.Save($"{dir}\\{i}_c.jpg");
//        }
//
//
//        resultImg.Save($"{dir}\\result.jpg");
//    }

    fun setDebug(isDebug: Boolean) {
        this.isDebug = isDebug
    }

    private fun updateUI(result: Array<SegmentResult>, oriImg: Bitmap) {
        val mutableBitmap: Bitmap

        //Log.d(TAG, "Inference time : ${result.inferenceTime}")

        if (isDebug) {
            mutableBitmap = oriImg.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            mutableBitmap = Bitmap.createBitmap(
                oriImg.width,
                oriImg.height, Bitmap.Config.ARGB_8888
            )
        }

        val canvas = Canvas(mutableBitmap)
        val textPaint = Paint()
        textPaint.color = Color.WHITE // Text Color
        textPaint.textSize = 28f // Text Size
        textPaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER) // Text Overlapping Pattern

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, textPaint)
        var xScale = 1.0f
        var yScale = 1.0f

        if (overlay != null) {
            xScale = overlay!!.scaleX
            yScale = overlay!!.scaleY
        }

        result.forEach {

            Log.d(TAG, "Rect: (${it.x1}, ${it.y1}, ${it.x2}, ${it.y2})")
            // out => x0,y0,x1,y1,score,cls_id
            val x = it.x1
            val y = it.y1
            val w = it.x2
            val h = it.y2
            val score = it.prob
            val classId = it.classId

            val left = ((x - w/2))
            val top = ((y - h/2))
            val width = ((left+w) * xScale)
            val height = ((top+h) * yScale)

            canvas.drawText("%s:%.2f".format(classes[classId.toInt()], score),
                x-w/2+8, y-h/2+32, textPaint)

            val boxPaint = Paint()
            val pixel = oriImg.getPixel((width / 2).roundToInt(),
                (height / 2).roundToInt()
            ) * classId.toInt() / 255

            // border
            boxPaint.strokeWidth = 6f
            boxPaint.style = Paint.Style.STROKE

            boxPaint.setARGB(255, 255, pixel, 255 - pixel)

            canvas.drawRect(left, top, (width), (height), boxPaint)
        }

        val bitmap: Bitmap? = BitmapUtils.getResizedBitmap(mutableBitmap, mutableBitmap.width * 2, mutableBitmap.height * 2)

        if (bitmap != null) {
            listener(bitmap)
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