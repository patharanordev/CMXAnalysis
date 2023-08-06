package com.patharanor.cmxanalysis.utils

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import androidx.core.graphics.get
import com.patharanor.cmxanalysis.analyzer.BoundingBox
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt


class CameraCapture(baseContext: Context) {
    private val TAG = "CameraCapture"
    private val FILENAME_FORMAT = "yyyy-MM-dd HH:mm:ss.SSS"
    private var ALBUM_NAME = "ObjectDetection"
    private var imageCapture: ImageCapture? = null
    private var context: Context

    init {
        context = baseContext
    }

    fun build(): ImageCapture? {
        imageCapture = ImageCapture.Builder()
            .build()
        return imageCapture
    }

    fun takePhoto(bboxes: Array<BoundingBox>) {

//        bboxes.map {
//            Log.d(TAG, "-----------------------------")
//            Log.d(TAG, "left: ${it.left}")
//            Log.d(TAG, "top: ${it.top}")
//            Log.d(TAG, "width: ${it.width}")
//            Log.d(TAG, "height: ${it.height}")
//        }

        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(context.contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )
    }

    fun recordTargetObject(bitmap: Bitmap, bboxes: Array<BoundingBox>) {
        bboxes.forEach {

            Log.d(TAG, "-----------------------------")
            Log.d(TAG, "left: ${it.left}")
            Log.d(TAG, "top: ${it.top}")
            Log.d(TAG, "width: ${it.width}")
            Log.d(TAG, "height: ${it.height}")
            Log.d(TAG, "real-bitmap width: ${bitmap.width}")
            Log.d(TAG, "real-bitmap height: ${bitmap.height}")

            val mutableBitmap = Bitmap.createBitmap(
                bitmap,
                if (it.left < 0) 0 else it.left.roundToInt(),
                if (it.left < 0) 0 else it.top.roundToInt(),
                it.width.roundToInt() - it.left.roundToInt(),
                it.height.roundToInt() - it.top.roundToInt(),
            )
            saveImage(this.context, mutableBitmap, it.label)
        }
    }

    private fun saveImage(context: Context, bitmap: Bitmap, imageName: String?) {
        val simpleDateFormat = SimpleDateFormat("yyyymmsshhmmss")
        val date = simpleDateFormat.format(Date())
        val fileName = "IMG_" + date + "_${imageName}.jpg"
        val exists: Boolean

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            context.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                arrayOf(MediaStore.Images.Media.DISPLAY_NAME),
                "${MediaStore.Images.Media.DISPLAY_NAME} = '$fileName' ",
                null,
                MediaStore.Images.ImageColumns.DATE_ADDED + " DESC"
            ).let {
                exists = it?.count ?: 0 >= 1
                it?.close()
            }

            if (!exists) {
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis())
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.DISPLAY_NAME, imageName)
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/$ALBUM_NAME/")
                }

                val url = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)!!
                val out = context.contentResolver.openOutputStream(url)
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)

                Toast.makeText(context, "saved", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, "Already saved", Toast.LENGTH_SHORT).show()
            }

        } else {
            val imageDir = File("${Environment.getExternalStorageDirectory()}/$ALBUM_NAME/")
            if (!imageDir.exists())
                imageDir.mkdirs()

            val image = File(imageDir, imageName)

            if (!image.exists()) {
                val outputStream = FileOutputStream(image)
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                outputStream.close()

                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis())
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.DISPLAY_NAME, imageName)
                    put(MediaStore.Images.Media.DATA, image.absolutePath)
                }

                context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                Toast.makeText(context, "saved", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, "Already saved", Toast.LENGTH_SHORT).show()
            }
        }
    }
}