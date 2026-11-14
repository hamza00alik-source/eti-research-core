package com.eti.mobile

import kotlin.math.ln
import kotlin.math.sqrt

data class OhlcvCandle(
    val timestampMs: Long,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Double
)

/** Causal spot feature builder kept in parity with eti.features.build_features. */
object MarketFeatureBuilder {
    val baseFeatureNames = listOf(
        "ret_1", "ret_3", "ret_6", "volatility_20", "sma_gap_10",
        "sma_gap_30", "rsi_14", "range_pct", "body_pct",
        "close_location", "volume_z20", "atr_pct"
    )

    fun buildLatest(candles: List<OhlcvCandle>, featureNames: List<String> = baseFeatureNames): FloatArray {
        require(candles.size >= 31) { "At least 31 completed candles are required" }
        require(featureNames.isNotEmpty() && featureNames.distinct().size == featureNames.size) {
            "Feature schema must be non-empty and contain unique names"
        }
        require(featureNames.all { baseFeatureNames.contains(it) }) {
            "This on-device builder supports the ETI base spot schema only"
        }
        validateCandles(candles)

        val last = candles.lastIndex
        val close = candles.map { it.close }
        val open = candles.map { it.open }
        val high = candles.map { it.high }
        val low = candles.map { it.low }
        val volume = candles.map { it.volume }

        fun pctChange(period: Int): Double = close[last] / close[last - period] - 1.0
        fun mean(values: List<Double>): Double = values.average()
        fun sampleStd(values: List<Double>): Double {
            require(values.size >= 2) { "At least two values are required for sample standard deviation" }
            val average = mean(values)
            val sumSquares = values.sumOf { (it - average) * (it - average) }
            return sqrt(sumSquares / (values.size - 1))
        }

        val logReturns = (last - 19..last).map { ln(close[it] / close[it - 1]) }
        val volatility20 = sampleStd(logReturns)
        val sma10 = mean(close.subList(last - 9, last + 1))
        val sma30 = mean(close.subList(last - 29, last + 1))

        val deltas = (last - 13..last).map { close[it] - close[it - 1] }
        val gains = deltas.map { maxOf(it, 0.0) }
        val losses = deltas.map { maxOf(-it, 0.0) }
        val averageGain = mean(gains)
        val averageLoss = mean(losses)
        val rsi = when {
            averageLoss == 0.0 && averageGain > 0.0 -> 100.0
            averageGain == 0.0 && averageLoss > 0.0 -> 0.0
            averageGain == 0.0 && averageLoss == 0.0 -> 50.0
            else -> 100.0 - 100.0 / (1.0 + averageGain / averageLoss)
        }

        val trueRanges = (last - 13..last).map { index ->
            maxOf(
                high[index] - low[index],
                kotlin.math.abs(high[index] - close[index - 1]),
                kotlin.math.abs(low[index] - close[index - 1])
            )
        }
        val atr = mean(trueRanges)
        val span = high[last] - low[last]
        require(span > 0.0) { "Latest candle range must be positive for close_location" }

        val volumes20 = volume.subList(last - 19, last + 1)
        val volumeStd = sampleStd(volumes20)
        require(volumeStd > 0.0) { "Latest 20 candle volumes have zero variance" }

        val values = mapOf(
            "ret_1" to pctChange(1),
            "ret_3" to pctChange(3),
            "ret_6" to pctChange(6),
            "volatility_20" to volatility20,
            "sma_gap_10" to (close[last] / sma10 - 1.0),
            "sma_gap_30" to (close[last] / sma30 - 1.0),
            "rsi_14" to rsi,
            "range_pct" to ((high[last] - low[last]) / close[last]),
            "body_pct" to ((close[last] - open[last]) / open[last]),
            "close_location" to ((close[last] - low[last]) / span),
            "volume_z20" to ((volume[last] - mean(volumes20)) / volumeStd),
            "atr_pct" to (atr / close[last])
        )
        return featureNames.map { name ->
            val value = values[name] ?: error("Unsupported feature: $name")
            require(value.isFinite()) { "Feature $name is not finite" }
            value.toFloat()
        }.toFloatArray()
    }

    private fun validateCandles(candles: List<OhlcvCandle>) {
        var previousTimestamp = Long.MIN_VALUE
        candles.forEach { candle ->
            require(candle.timestampMs > previousTimestamp) {
                "Candle timestamps must be unique and chronological"
            }
            previousTimestamp = candle.timestampMs
            require(
                candle.open.isFinite() && candle.high.isFinite() &&
                    candle.low.isFinite() && candle.close.isFinite() &&
                    candle.volume.isFinite()
            ) { "OHLCV values must be finite" }
            require(candle.open > 0.0 && candle.high > 0.0 && candle.low > 0.0 && candle.close > 0.0) {
                "OHLC prices must be positive"
            }
            require(candle.volume >= 0.0) { "Volume cannot be negative" }
            require(candle.high >= maxOf(candle.open, candle.close, candle.low)) {
                "Candle high is inconsistent"
            }
            require(candle.low <= minOf(candle.open, candle.close, candle.high)) {
                "Candle low is inconsistent"
            }
        }
    }
}
