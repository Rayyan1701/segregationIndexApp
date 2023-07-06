package com.example.segregationindexmodel

import androidx.appcompat.app.AppCompatActivity

import android.content.Intent
import android.graphics.*

import android.os.Bundle
import android.os.FileUtils
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import com.example.segregationindexmodel.ml.Model1
import com.example.segregationindexmodel.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var btn: Button
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    lateinit var byteBuffer: ByteBuffer
    var colors = listOf<Int>(Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)

    lateinit var labels: List<String>
    lateinit var model: Model
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        labels = FileUtil.loadLabels(this, "labels.txt")
        model = Model.newInstance(this)

        paint.setColor(Color.BLUE)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5.0f
//        paint.textSize = paint.textSize*3

        Log.d("labels", labels.toString())


        val intent = Intent()
        intent.setAction(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")

        btn = findViewById(R.id.btn)
        imageView = findViewById(R.id.imaegView)

        btn.setOnClickListener {
            startActivityForResult(intent, 101)
        }






    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 101){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
//            uri?.let {
//                val inputStream = contentResolver.openInputStream(uri)
//                inputStream?.use {
//                    val byteArray = inputStream.readBytes()
//                    byteBuffer = ByteBuffer.wrap(byteArray)
//
//                    get_predictions()
//                }
//            }
            get_predictions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }



    fun get_predictions(){

//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 1, 3), DataType.UINT8)
//
//        val expectedSize = 3
//        if (byteBuffer.capacity() < expectedSize) {
//            val resizedByteBuffer = ByteBuffer.allocateDirect(expectedSize)
//            byteBuffer.rewind() // Rewind the original byteBuffer
//            resizedByteBuffer.put(byteBuffer) // Copy data from the original byteBuffer to the resizedByteBuffer
//            byteBuffer = resizedByteBuffer // Assign the resizedByteBuffer to byteBuffer
//        } else {
//            byteBuffer.rewind() // Rewind the byteBuffer before putting data
//        }
//
//
//
//        inputFeature0.loadBuffer(byteBuffer)
//
//
//// Runs model inference and gets result.
//        val outputs = model.process(inputFeature0)
//        val locations = outputs.outputFeature0AsTensorBuffer.floatArray
//        val classes = outputs.outputFeature1AsTensorBuffer.floatArray
//        val scores = outputs.outputFeature2AsTensorBuffer.floatArray
//        val numberOfDetections = outputs.outputFeature3AsTensorBuffer.floatArray



        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray



        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        var canvas = Canvas(mutable)
        var h = mutable.height
        var w = mutable.width


        paint.textSize = h/15f
        paint.strokeWidth = h/85f
        scores.forEachIndexed { index, fl ->
            if(fl > 0.5){
                var x = index
                x *= 4
                paint.setColor(colors.get(index))
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels[classes.get(index).toInt()] + " " + fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
            }
        }

        imageView.setImageBitmap(mutable)

    }



}

private fun Any.process(image: TensorImage?): Model1.Outputs {
    TODO("Not yet implemented")
}
