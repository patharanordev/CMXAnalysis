package com.patharanor.cmxanalysis.utils

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import androidx.camera.core.ImageCapture
import androidx.core.content.ContextCompat
import android.widget.Toast
import android.util.Log
import androidx.camera.core.ImageCaptureException
import com.patharanor.cmxanalysis.analyzer.BoundingBox
import java.text.SimpleDateFormat
import java.util.Locale

class CameraCapture(baseContext: Context) {
    private val TAG = "CameraCapture"
    private val FILENAME_FORMAT = "yyyy-MM-dd HH:mm:ss.SSS"
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

        bboxes.map {
            Log.d(TAG, "-----------------------------")
            Log.d(TAG, "left: ${it.left}")
            Log.d(TAG, "top: ${it.top}")
            Log.d(TAG, "width: ${it.width}")
            Log.d(TAG, "height: ${it.height}")
        }

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
                        onImageSaved(output: ImageCapture.OutputFileResults){
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )
    }
}