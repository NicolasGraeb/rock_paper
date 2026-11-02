package studia.ai_projekt

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import studia.ai_projekt.ml.RpsClassifier
import kotlin.math.abs
import kotlin.math.max

// zdjęcia z HaGRID test (z export.py) przez RpsClassifier vs predykcje PyTorch
@RunWith(AndroidJUnit4::class)
class RpsClassifierParityTest {
    @Test
    fun predictionsMatchPyTorch() {
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val expected = JSONObject(testAssets.open("parity/expected.json").bufferedReader().use { it.readText() })
        val classNames = expected.getJSONArray("class_names").let { a -> List(a.length()) { a.getString(it) } }
        val images = expected.getJSONArray("images")

        RpsClassifier(appContext).use { classifier ->
            assertEquals(classNames, classifier.classNames)
            var agree = 0
            var correct = 0
            var maxDiff = 0f
            val times = mutableListOf<Long>()
            for (i in 0 until images.length()) {
                val item = images.getJSONObject(i)
                val bitmap = testAssets.open("parity/${item.getString("file")}").use {
                    BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply {
                        inPreferredConfig = Bitmap.Config.ARGB_8888
                    })!!
                }
                val start = System.nanoTime()
                val result = classifier.classify(bitmap)
                times += System.nanoTime() - start
                val ref = item.getJSONArray("probs").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
                val refLabel = classNames[ref.indices.maxBy { ref[it] }]
                if (result.label == refLabel) agree++
                if (result.label == item.getString("true")) correct++
                classNames.forEachIndexed { c, name -> maxDiff = max(maxDiff, abs(result.probabilities[name]!! - ref[c])) }
            }
            val n = images.length()
            val ms = times.sorted().map { it / 1e6 }
            val summary = "n=$n zgodne_z_pytorch=$agree poprawne=$correct max_roznica_p=%.4f mediana_ms=%.1f p95_ms=%.1f"
                .format(maxDiff, ms[n / 2], ms[(n * 95) / 100])
            Log.i("RpsParity", summary)
            println("RpsParity $summary")
            assertEquals("predykcje różne od PyTorch: $summary", n, agree)
            assertTrue("za duża różnica prawdopodobieństw: $summary", maxDiff < 0.05f)
        }
    }
}
