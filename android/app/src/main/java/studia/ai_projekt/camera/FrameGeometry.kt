package studia.ai_projekt.camera

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

// bok kwadratu jako ułamek krótszego boku podglądu
const val FRAME_FRACTION = 1f

data class CropBox(val left: Int, val top: Int, val size: Int)

// PreviewView FILL_CENTER: środek widoku = środek klatki, skala = max(view / image)
fun frameCrop(imageWidth: Int, imageHeight: Int, viewWidth: Int, viewHeight: Int, fraction: Float = FRAME_FRACTION): CropBox {
    val scale = max(viewWidth.toFloat() / imageWidth, viewHeight.toFloat() / imageHeight)
    val sideOnScreen = min(viewWidth, viewHeight) * fraction
    val size = min((sideOnScreen / scale).roundToInt(), min(imageWidth, imageHeight))
    return CropBox(left = (imageWidth - size) / 2, top = (imageHeight - size) / 2, size = size)
}
