package com.eti.mobile

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Runs an ETI ONNX candidate on-device.
 *
 * Place a reviewed eti_model.onnx and its matching eti_model_metadata.json
 * in app/src/main/assets before enabling this class in the UI. Feature values
 * must be computed in the exact metadata order by the caller.
 */
class OnDeviceModelRunner(
    context: Context,
    modelAssetName: String = "eti_model.onnx",
    metadataAssetName: String = "eti_model_metadata.json"
) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val probabilityOutputName: String
    private val featureCount: Int
    private val classIds: List<Int>
    private val temperature: Double

    init {
        val metadataText = context.assets.open(metadataAssetName).bufferedReader().use { it.readText() }
        val metadata = JSONObject(metadataText)
        val featureNames = metadata.getJSONArray("feature_names")
        featureCount = featureNames.length()
        require(featureCount > 0) { "Model metadata has no feature names" }
        inputName = metadata.getString("input_name")
        probabilityOutputName = metadata.getString("probability_output_name")
        val classes = metadata.getJSONArray("class_ids")
        classIds = (0 until classes.length()).map { classes.getInt(it) }
        require(classIds.isNotEmpty()) { "Model metadata has no class IDs" }
        val calibration = metadata.optJSONObject("calibration")
        temperature = calibration?.optDouble("temperature", 1.0)
            ?.takeIf { it.isFinite() && it > 0.0 } ?: 1.0

        val modelFile = File(context.cacheDir, modelAssetName)
        context.assets.open(modelAssetName).use { source ->
            modelFile.outputStream().use { destination -> source.copyTo(destination) }
        }
        val options = OrtSession.SessionOptions()
        session = try {
            environment.createSession(modelFile.absolutePath, options)
        } finally {
            options.close()
        }
        require(session.inputNames.contains(inputName)) { "ONNX input name does not match metadata" }
        require(session.outputNames.contains(probabilityOutputName)) {
            "ONNX probability output name does not match metadata"
        }
    }

    /** Returns calibrated probabilities keyed by ETI class ID: 0=short, 1=flat, 2=long. */
    fun predict(featureValues: FloatArray): Map<Int, Float> {
        require(featureValues.size == featureCount) {
            "Expected ${featureCount} ordered feature values, got ${featureValues.size}"
        }
        require(featureValues.all { it.isFinite() }) { "Feature values must be finite" }

        val tensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(featureValues),
            longArrayOf(1L, featureCount.toLong())
        )
        tensor.use {
            session.run(mapOf(inputName to it), setOf(probabilityOutputName)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val rows = result.get(0).value as Array<FloatArray>
                require(rows.isNotEmpty() && rows[0].size == classIds.size) {
                    "ONNX probability output does not match the class mapping"
                }
                val raw = rows[0].map { it.toDouble() }
                require(raw.all { it.isFinite() && it >= 0.0 && it <= 1.0 }) {
                    "ONNX returned invalid probabilities"
                }
                val logits = raw.map { ln(max(it, 1e-12)) / temperature }
                val peak = logits.maxOrNull() ?: 0.0
                val weights = logits.map { exp(it - peak) }
                val total = weights.sum()
                require(total.isFinite() && total > 0.0) { "Probability normalization failed" }
                return classIds.indices.associate { index ->
                    classIds[index] to (weights[index] / total).toFloat()
                }
            }
        }
    }

    override fun close() {
        session.close()
    }
}
