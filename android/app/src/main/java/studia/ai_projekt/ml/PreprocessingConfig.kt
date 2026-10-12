package studia.ai_projekt.ml

import android.content.Context
import org.json.JSONObject

data class PreprocessingConfig(
    val imageSize: Int,
    val mean: FloatArray,
    val std: FloatArray,
    val classNames: List<String>,
    val inputName: String,
    val outputName: String,
) {
    companion object {
        fun fromAssets(context: Context, assetPath: String = "preprocessing.json"): PreprocessingConfig {
            val json = context.assets.open(assetPath).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val meanJson = root.getJSONArray("mean")
            val stdJson = root.getJSONArray("std")
            val classesJson = root.getJSONArray("class_names")
            val mean = FloatArray(meanJson.length()) { meanJson.getDouble(it).toFloat() }
            val std = FloatArray(stdJson.length()) { stdJson.getDouble(it).toFloat() }
            val classNames = List(classesJson.length()) { classesJson.getString(it) }
            return PreprocessingConfig(
                imageSize = root.getInt("image_size"),
                mean = mean,
                std = std,
                classNames = classNames,
                inputName = root.optString("input_name", "input"),
                outputName = root.optString("output_name", "logits"),
            )
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PreprocessingConfig) return false
        return imageSize == other.imageSize &&
            mean.contentEquals(other.mean) &&
            std.contentEquals(other.std) &&
            classNames == other.classNames &&
            inputName == other.inputName &&
            outputName == other.outputName
    }

    override fun hashCode(): Int {
        var result = imageSize
        result = 31 * result + mean.contentHashCode()
        result = 31 * result + std.contentHashCode()
        result = 31 * result + classNames.hashCode()
        result = 31 * result + inputName.hashCode()
        result = 31 * result + outputName.hashCode()
        return result
    }
}
