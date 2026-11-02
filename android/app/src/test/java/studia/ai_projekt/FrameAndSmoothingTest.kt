package studia.ai_projekt

import org.junit.Assert.assertEquals
import org.junit.Test
import studia.ai_projekt.camera.CropBox
import studia.ai_projekt.camera.frameCrop
import studia.ai_projekt.ml.ClassificationResult
import studia.ai_projekt.ml.PredictionSmoother

class FrameAndSmoothingTest {
    @Test
    fun cropMatchesCenteredFrameOnScreen() {
        val box = frameCrop(480, 640, 1080, 2400, fraction = 0.75f)
        // skala 2400 / 640 = 3.75, kwadrat 0.75 * 1080 = 810 px na ekranie -> 216 px w klatce
        assertEquals(CropBox(left = 132, top = 212, size = 216), box)
    }

    @Test
    fun cropNeverExceedsImage() {
        val box = frameCrop(480, 640, 480, 640, fraction = 1f)
        assertEquals(CropBox(left = 0, top = 80, size = 480), box)
    }

    @Test
    fun singleOutlierFrameDoesNotFlipLabel() {
        val classes = listOf("paper", "rock", "scissors")
        val smoother = PredictionSmoother(alpha = 0.3f)
        fun frame(p: Float, r: Float, s: Float) =
            ClassificationResult("", 0f, mapOf("paper" to p, "rock" to r, "scissors" to s))
        repeat(5) { smoother.update(frame(0.05f, 0.9f, 0.05f), classes) }
        val afterOutlier = smoother.update(frame(0.9f, 0.05f, 0.05f), classes)
        assertEquals("rock", afterOutlier.label)
    }
}
