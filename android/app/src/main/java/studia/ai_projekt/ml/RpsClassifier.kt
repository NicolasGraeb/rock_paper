package studia.ai_projekt.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

data class ClassificationResult(
    val label: String,
    val confidence: Float,
    val probabilities: Map<String, Float>,
)

class RpsClassifier(
    context: Context,
    modelAssetPath: String = "model.onnx",
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val config = PreprocessingConfig.fromAssets(appContext)
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputBuffer = FloatArray(3 * config.imageSize * config.imageSize)

    val classNames: List<String> get() = config.classNames

    init {
        val modelBytes = appContext.assets.open(modelAssetPath).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, options)
    }

    @Synchronized
    fun classify(bitmap: Bitmap): ClassificationResult {
        val resized = Bitmap.createScaledBitmap(bitmap, config.imageSize, config.imageSize, true)
        fillInput(resized)
        if (resized !== bitmap) {
            resized.recycle()
        }

        val shape = longArrayOf(1, 3, config.imageSize.toLong(), config.imageSize.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuffer), shape).use { tensor ->
            session.run(Collections.singletonMap(config.inputName, tensor)).use { results ->
                val output = results[0].value as Array<FloatArray>
                val logits = output[0]
                val probs = softmax(logits)
                var bestIdx = 0
                for (i in 1 until probs.size) {
                    if (probs[i] > probs[bestIdx]) bestIdx = i
                }
                val probabilityMap = config.classNames.mapIndexed { index, name ->
                    name to probs[index]
                }.toMap()
                return ClassificationResult(
                    label = config.classNames[bestIdx],
                    confidence = probs[bestIdx],
                    probabilities = probabilityMap,
                )
            }
        }
    }

    private fun fillInput(bitmap: Bitmap) {
        val size = config.imageSize
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        val area = size * size
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            inputBuffer[i] = (r - config.mean[0]) / config.std[0]
            inputBuffer[area + i] = (g - config.mean[1]) / config.std[1]
            inputBuffer[2 * area + i] = (b - config.mean[2]) / config.std[2]
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        var sum = 0.0
        val exps = DoubleArray(logits.size) { index ->
            val value = exp((logits[index] - max).toDouble())
            sum += value
            value
        }
        return FloatArray(logits.size) { index -> (exps[index] / sum).toFloat() }
    }

    override fun close() {
        session.close()
    }
}
