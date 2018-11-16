package com.pienovi.dday.activity

import android.graphics.*
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.google.firebase.ml.vision.common.FirebaseVisionPoint
import com.google.firebase.ml.vision.face.*
import com.otaliastudios.cameraview.Facing
import com.otaliastudios.cameraview.Frame
import com.otaliastudios.cameraview.Gesture
import com.otaliastudios.cameraview.GestureAction
import com.pienovi.dday.R
import kotlinx.android.synthetic.main.activity_main.*


class MainActivity : AppCompatActivity() {

    companion object {

        private const val RADIUS = 3f
        private const val STROKE_WIDTH = 2f
        private const val COLOR_DOT = Color.WHITE
        private const val COLOR_LINE = Color.GREEN
        private const val COLOR_MOUTH = Color.RED
        private const val COLOR_EYE_IRIS = Color.CYAN
        private const val COLOR_EYE_LID = Color.YELLOW
        private const val MIN_EYE_OPEN_PROBABILITY = 0.5f
        private const val EYE_RADIUS_PROPORTION = 0.45f
        private const val IRIS_RADIUS_PROPORTION = EYE_RADIUS_PROPORTION / 2.0f

    }

    private var processing = false
    private var bitmap: Bitmap? = null
    private var canvas: Canvas? = null

    private val dotPaint = Paint()
    private val linePaint = Paint()
    private val mouthPaint = Paint()

    private var eyeWhitesPaint = Paint()
    private var eyeIrisPaint = Paint()
    private var eyeOutlinePaint = Paint()
    private var eyeLidPaint = Paint()

    private lateinit var faceDetector: FirebaseVisionFaceDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        dotPaint.color = COLOR_DOT
        dotPaint.style = Paint.Style.FILL
        dotPaint.strokeWidth = STROKE_WIDTH

        linePaint.color = COLOR_LINE
        linePaint.style = Paint.Style.STROKE
        linePaint.strokeWidth = STROKE_WIDTH

        mouthPaint.color = COLOR_MOUTH
        mouthPaint.style = Paint.Style.FILL

        eyeWhitesPaint = Paint()
        eyeWhitesPaint.color = Color.WHITE
        eyeWhitesPaint.style = Paint.Style.FILL

        eyeLidPaint = Paint()
        eyeLidPaint.color = COLOR_EYE_LID
        eyeLidPaint.style = Paint.Style.FILL

        eyeIrisPaint = Paint()
        eyeIrisPaint.color = COLOR_EYE_IRIS
        eyeIrisPaint.style = Paint.Style.FILL

        eyeOutlinePaint = Paint()
        eyeOutlinePaint.color = Color.BLACK
        eyeOutlinePaint.style = Paint.Style.STROKE
        eyeOutlinePaint.strokeWidth = 5f

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
            .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
            .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
            .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
            .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
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

                if (contoursCheckBox.isChecked) {
                    drawContours(canvas!!, it)
                }
                if (mouthCheckBox.isChecked) {
                    drawMouth(canvas!!, it)
                }
                if (eyesCheckBox.isChecked) {
                    drawEyes(canvas!!, it)
                }
                if (contoursCheckBox.isChecked || mouthCheckBox.isChecked || eyesCheckBox.isChecked) {
                    updateImageView()
                }
            }
            .addOnFailureListener {
                processing = false
                imageView.setImageBitmap(null)
            }
    }

    private fun updateImageView() {
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

    private fun drawContours(canvas: Canvas, faces: List<FirebaseVisionFace>) {
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
        }
    }

    private fun drawMouth(canvas: Canvas, faces: List<FirebaseVisionFace>) {
        for (face in faces) {
            val upperLipTopContours = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_TOP).points
            val upperLipBottomContours = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_BOTTOM).points
            drawLip(canvas, upperLipTopContours, upperLipBottomContours)

            val lowerLipTopContours = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_TOP).points
            val lowerLipBottomContours = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_BOTTOM).points
            drawLip(canvas, lowerLipTopContours, lowerLipBottomContours)
        }
    }

    private fun drawLip(
        canvas: Canvas,
        topContours: List<FirebaseVisionPoint>,
        bottomContours: List<FirebaseVisionPoint>
    ) {
        if (topContours.isEmpty() || bottomContours.isEmpty()) {
            return
        }

        val pathUpperLip = Path()

        val firstUpperX = topContours[0].x
        val firstUpperY = topContours[0].y
        pathUpperLip.moveTo(firstUpperX, firstUpperY)

        for (contour in topContours) {
            pathUpperLip.lineTo(contour.x, contour.y)
        }

        for (i in (bottomContours.size - 1) downTo 0) {
            val contour = bottomContours[i]
            pathUpperLip.lineTo(contour.x, contour.y)
        }

        pathUpperLip.close()
        canvas.drawPath(pathUpperLip, mouthPaint)
    }

    private fun drawEyes(canvas: Canvas, faces: List<FirebaseVisionFace>) {
        for (face in faces) {
            val leftEye = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EYE)
            val rightEye = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EYE)

            if (leftEye == null || rightEye == null) {
                return
            }

            val detectLeftPosition = leftEye.position
            val detectRightPosition = rightEye.position
            val leftPosition = PointF(detectLeftPosition.x, detectLeftPosition.y)
            val rightPosition = PointF(detectRightPosition.x, detectRightPosition.y)
            val leftOpen = face.leftEyeOpenProbability > MIN_EYE_OPEN_PROBABILITY
            val rightOpen = face.rightEyeOpenProbability > MIN_EYE_OPEN_PROBABILITY

            val distance = Math.sqrt(
                Math.pow(rightPosition.x.toDouble() - leftPosition.x, 2.0)
                        + Math.pow(rightPosition.y.toDouble() - leftPosition.y, 2.0)
            )
            val eyeRadius = EYE_RADIUS_PROPORTION * distance
            val irisRadius = IRIS_RADIUS_PROPORTION * distance

            drawEye(canvas, leftPosition, eyeRadius.toFloat(), leftPosition, irisRadius.toFloat(), leftOpen)
            drawEye(canvas, rightPosition, eyeRadius.toFloat(), rightPosition, irisRadius.toFloat(), rightOpen)
        }
    }

    private fun drawEye(
        canvas: Canvas, eyePosition: PointF, eyeRadius: Float,
        irisPosition: PointF, irisRadius: Float, isOpen: Boolean
    ) {
        if (isOpen) {
            canvas.drawCircle(eyePosition.x, eyePosition.y, eyeRadius, eyeWhitesPaint)
            canvas.drawCircle(irisPosition.x, irisPosition.y, irisRadius, eyeIrisPaint)
        } else {
            canvas.drawCircle(eyePosition.x, eyePosition.y, eyeRadius, eyeLidPaint)
            val y = eyePosition.y
            val start = eyePosition.x - eyeRadius
            val end = eyePosition.x + eyeRadius
            canvas.drawLine(start, y, end, y, eyeOutlinePaint)
        }
        canvas.drawCircle(eyePosition.x, eyePosition.y, eyeRadius, eyeOutlinePaint)
    }

}