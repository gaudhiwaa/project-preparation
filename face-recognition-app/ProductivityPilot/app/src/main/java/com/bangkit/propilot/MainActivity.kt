package com.bangkit.propilot

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.bangkit.propilot.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {
    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model:SsdMobilenetV11Metadata1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//      This file defines the user interface of the app
        setContentView(R.layout.activity_main)
//      Get camera permission
        get_permission()

//      Load file named labels.txt that contains the labels of the objects
        labels = FileUtil.loadLabels(this, "labels.txt")
//      Resizes the input images to 300x300 using bilinear interpolation
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
//      Load model that capable to detecting objects in images
        model = SsdMobilenetV11Metadata1.newInstance(this)
//      Creates a new thread for processing the camera video stream
        val handlerThread = HandlerThread("videoThread")
//      This starts the video processing thread
        handlerThread.start()
//      Creates a handler that can post tasks to the video processing thread
        handler = Handler(handlerThread.looper)

//      This finds the ImageView in the layout file
        imageView = findViewById(R.id.imageView)
//      This finds the TextureView in the layout file
        textureView = findViewById(R.id.textureView)

//      Sets a listener for the TextureView that will be called when the SurfaceTexture is ready.
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
//              Opens the camera and starts the video stream
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
//              This gets the current frame from the TextureView as a Bitmap
                bitmap = textureView.bitmap!!
//              This creates a TensorImage object from the Bitmap.
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

//              Processes the TensorImage using the pre-trained machine learning model.
                val outputs = model.process(image)
//              Gets the locations of the detected objects in the image as an array of floats.
                val locations = outputs.locationsAsTensorBuffer.floatArray
//              Gets the classes of the detected objects in the image as an array of floats.
                val classes = outputs.classesAsTensorBuffer.floatArray
//              Gets the scores of the detected objects in the image as an array of floats.
                val scores = outputs.scoresAsTensorBuffer.floatArray
//              Gets the number of detections in the image as an array of floats.
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray
//              Creates a mutable copy of the Bitmap.
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
//              This creates a Canvas object from the mutable Bitmap.
                val canvas = Canvas(mutable)

//              Mutable (which is the current camera frame being processed)
//              Drawing rectangles and text labels on the camera preview indicating the detected objects.
                val h = mutable.height
                val w = mutable.width
                paint.textSize = h/15f
                paint.strokeWidth = h/85f
                var x = 0
//              Represents the confidence score of the current object.
                scores.forEachIndexed { index, fl ->
                    x = index
//                  The index variable x is updated to the current object's index multiplied by 4.
//                  This is because the locations list contains four values (ymin, xmin, ymax, xmax) for each object.
                    x *= 4
//                  If statement checks if the confidence score is greater than 0.5.
//                  If so, the paint object is used to set the color and style for the rectangle and label.
                    if(fl > 0.5){
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE
//                      A rectangle is then drawn on the canvas using the drawRect method.
//                      The RectF object takes four arguments: the left, top, right, and bottom coordinates of the rectangle.
//                      These coordinates are calculated using the current object's location values and the bitmap's height and width.
                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
//                      Paint object is changed to FILL, and the label text is drawn on the canvas using the drawText method.
                        paint.style = Paint.Style.FILL
                        canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
                    }
                }

                imageView.setImageBitmap(mutable)

            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }

//  This code block is an override of the onDestroy() method of an Android activity,
//  which is called when the activity is being destroyed.
//  In this particular implementation, the model.close() method is called before the activity is destroyed.
    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

//  Open Camera
    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }

//  Camera permission
    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }
}