package studia.ai_projekt.ml

// średnia wykładnicza prawdopodobieństw z kolejnych klatek
class PredictionSmoother(private val alpha: Float = 0.3f) {
    private var smoothed: FloatArray? = null

    @Synchronized
    fun update(result: ClassificationResult, classNames: List<String>): ClassificationResult {
        val current = FloatArray(classNames.size) { result.probabilities[classNames[it]] ?: 0f }
        val previous = smoothed
        val next = if (previous == null) current else FloatArray(current.size) {
            alpha * current[it] + (1 - alpha) * previous[it]
        }
        smoothed = next
        val best = next.indices.maxBy { next[it] }
        return ClassificationResult(
            label = classNames[best],
            confidence = next[best],
            probabilities = classNames.indices.associate { classNames[it] to next[it] },
        )
    }
}
