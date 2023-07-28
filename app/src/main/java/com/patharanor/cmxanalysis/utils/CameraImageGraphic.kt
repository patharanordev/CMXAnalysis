package com.patharanor.cmxanalysis.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import com.patharanor.cmxanalysis.utils.GraphicOverlay.Graphic

/** Draw camera image to background.  */
class CameraImageGraphic(overlay: GraphicOverlay?, private val bitmap: Bitmap) :
    Graphic(overlay!!) {
    override fun draw(canvas: Canvas?) {
        canvas!!.drawBitmap(bitmap, getTransformationMatrix(), null)
    }
}
