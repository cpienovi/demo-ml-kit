package com.pienovi.dday.activity

import android.graphics.*
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.otaliastudios.cameraview.Facing
import com.otaliastudios.cameraview.Frame
import com.otaliastudios.cameraview.Gesture
import com.otaliastudios.cameraview.GestureAction
import com.pienovi.dday.R
import kotlinx.android.synthetic.main.activity_main.*


class MainActivity : AppCompatActivity() {

    companion object {

        private const val RADIUS = 2f

    }

    private var processing = false
    private var bitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private val dotPaint = Paint()
    private val linePaint = Paint()
    private lateinit var faceDetector: FirebaseVisionFaceDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        dotPaint.color = Color.RED
        dotPaint.style = Paint.Style.FILL
        dotPaint.strokeWidth = 2f
        linePaint.color = Color.GREEN
        linePaint.style = Paint.Style.STROKE
        linePaint.strokeWidth = 1f

        cameraView.setLifecycleOwner(this)
        cameraView.mapGesture(Gesture.PINCH, GestureAction.ZOOM)
        cameraView.mapGesture(Gesture.TAP, GestureAction.FOCUS_WITH_MARKER)
        cameraView.addFrameProcessor {
            if (it.data == null) {
                return@addFrameProcessor
            }

            process(it)
        }

        switchCameraButton.setOnClickListener {
            bitmap?.eraseColor(Color.TRANSPARENT)
            imageView.setImageBitmap(null)
            cameraView.toggleFacing()
        }

        val options = FirebaseVisionFaceDetectorOptions.Builder()
            .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
            .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
            .build()
        faceDetector = FirebaseVision.getInstance().getVisionFaceDetector(options)
    }

    private fun process(frame: Frame) {
        if (processing) {
            return
        }

        processing = true
        val width = frame.size.width
        val height = frame.size.height

        val metadata = FirebaseVisionImageMetadata.Builder()
            .setWidth(width)
            .setHeight(height)
            .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
            .setRotation(if (cameraView.facing == Facing.FRONT) FirebaseVisionImageMetadata.ROTATION_270 else FirebaseVisionImageMetadata.ROTATION_90)
            .build()

        val firebaseVisionImage = FirebaseVisionImage.fromByteArray(frame.data, metadata)

        faceDetector.detectInImage(firebaseVisionImage)
            .addOnSuccessListener {
                processing = false
                imageView.setImageBitmap(null)

                if (bitmap == null) {
                    bitmap = Bitmap.createBitmap(height, width, Bitmap.Config.ARGB_8888)
                }

                if (canvas == null) {
                    bitmap!!.eraseColor(Color.TRANSPARENT)
                    canvas = Canvas(bitmap!!)
                } else {
                    canvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                }

                drawFaces(canvas!!, it)
            }
            .addOnFailureListener {
                processing = false
                imageView.setImageBitmap(null)
            }
    }

    private fun drawFaces(canvas: Canvas, faces: List<FirebaseVisionFace>) {
        for (face in faces) {
            val faceContours = face.getContour(FirebaseVisionFaceContour.FACE).points
            for ((i, contour) in faceContours.withIndex()) {
                if (i != faceContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        faceContours[i + 1].x,
                        faceContours[i + 1].y,
                        linePaint
                    )
                else
                    canvas.drawLine(contour.x, contour.y, faceContours[0].x, faceContours[0].y, linePaint)
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val leftEyebrowTopContours = face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_TOP).points
            for ((i, contour) in leftEyebrowTopContours.withIndex()) {
                if (i != leftEyebrowTopContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        leftEyebrowTopContours[i + 1].x,
                        leftEyebrowTopContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val leftEyebrowBottomContours =
                face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_BOTTOM).points
            for ((i, contour) in leftEyebrowBottomContours.withIndex()) {
                if (i != leftEyebrowBottomContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        leftEyebrowBottomContours[i + 1].x,
                        leftEyebrowBottomContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val rightEyebrowTopContours = face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_TOP).points
            for ((i, contour) in rightEyebrowTopContours.withIndex()) {
                if (i != rightEyebrowTopContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        rightEyebrowTopContours[i + 1].x,
                        rightEyebrowTopContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val rightEyebrowBottomContours =
                face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_BOTTOM).points
            for ((i, contour) in rightEyebrowBottomContours.withIndex()) {
                if (i != rightEyebrowBottomContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        rightEyebrowBottomContours[i + 1].x,
                        rightEyebrowBottomContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val leftEyeContours = face.getContour(FirebaseVisionFaceContour.LEFT_EYE).points
            for ((i, contour) in leftEyeContours.withIndex()) {
                if (i != leftEyeContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        leftEyeContours[i + 1].x,
                        leftEyeContours[i + 1].y,
                        linePaint
                    )
                else
                    canvas.drawLine(contour.x, contour.y, leftEyeContours[0].x, leftEyeContours[0].y, linePaint)
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val rightEyeContours = face.getContour(FirebaseVisionFaceContour.RIGHT_EYE).points
            for ((i, contour) in rightEyeContours.withIndex()) {
                if (i != rightEyeContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        rightEyeContours[i + 1].x,
                        rightEyeContours[i + 1].y,
                        linePaint
                    )
                else
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        rightEyeContours[0].x,
                        rightEyeContours[0].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val upperLipTopContours = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_TOP).points
            for ((i, contour) in upperLipTopContours.withIndex()) {
                if (i != upperLipTopContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        upperLipTopContours[i + 1].x,
                        upperLipTopContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val upperLipBottomContours = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_BOTTOM).points
            for ((i, contour) in upperLipBottomContours.withIndex()) {
                if (i != upperLipBottomContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        upperLipBottomContours[i + 1].x,
                        upperLipBottomContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val lowerLipTopContours = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_TOP).points
            for ((i, contour) in lowerLipTopContours.withIndex()) {
                if (i != lowerLipTopContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        lowerLipTopContours[i + 1].x,
                        lowerLipTopContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val lowerLipBottomContours = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_BOTTOM).points
            for ((i, contour) in lowerLipBottomContours.withIndex()) {
                if (i != lowerLipBottomContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        lowerLipBottomContours[i + 1].x,
                        lowerLipBottomContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val noseBridgeContours = face.getContour(FirebaseVisionFaceContour.NOSE_BRIDGE).points
            for ((i, contour) in noseBridgeContours.withIndex()) {
                if (i != noseBridgeContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        noseBridgeContours[i + 1].x,
                        noseBridgeContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            val noseBottomContours = face.getContour(FirebaseVisionFaceContour.NOSE_BOTTOM).points
            for ((i, contour) in noseBottomContours.withIndex()) {
                if (i != noseBottomContours.lastIndex)
                    canvas.drawLine(
                        contour.x,
                        contour.y,
                        noseBottomContours[i + 1].x,
                        noseBottomContours[i + 1].y,
                        linePaint
                    )
                canvas.drawCircle(contour.x, contour.y, RADIUS, dotPaint)
            }

            if (cameraView.facing == Facing.FRONT) {
                val matrix = Matrix()
                matrix.preScale(-1F, 1F)
                val flippedBitmap =
                    Bitmap.createBitmap(bitmap!!, 0, 0, bitmap!!.width, bitmap!!.height, matrix, true)
                imageView.setImageBitmap(flippedBitmap)
            } else {
                imageView.setImageBitmap(bitmap)
            }
        }
    }

}