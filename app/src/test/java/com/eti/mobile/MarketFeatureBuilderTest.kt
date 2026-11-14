package com.eti.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class MarketFeatureBuilderTest {
    private fun sampleCandles(): List<OhlcvCandle> = (0 until 40).map { index ->
        val i = index.toDouble()
        val open = 100.0 + i * 0.3 + sin(i * 0.2)
        val close = open * (1.0 + sin(i * 0.13) * 0.002 + 0.001)
        val high = maxOf(open, close) + 0.5 + 0.02 * cos(i * 0.17)
        val low = minOf(open, close) - 0.4 - 0.01 * sin(i * 0.11)
        val volume = 1000.0 + i * 3.0 + 20.0 * sin(i * 0.4)
        OhlcvCandle(index * 3_600_000L, open, high, low, close, volume)
    }

    @Test
    fun latestFeaturesMatchPythonReferenceVector() {
        val expected = floatArrayOf(
            0.0030175684f, 0.0100277284f, 0.0225153128f, 0.0010023164f,
            0.0167332097f, 0.0483936668f, 100.0f, 0.0089531626f,
            -0.0008734715f, 0.3877277977f, 1.1213243246f, 0.0090633792f
        )
        val actual = MarketFeatureBuilder.buildLatest(sampleCandles())
        assertEquals(expected.size, actual.size)
        expected.indices.forEach { index ->
            assertEquals("feature index $index", expected[index], actual[index], 1e-5f)
        }
    }

    @Test
    fun rejectsNonChronologicalCandles() {
        val candles = sampleCandles().toMutableList()
        candles[10] = candles[9]
        assertThrows(IllegalArgumentException::class.java) {
            MarketFeatureBuilder.buildLatest(candles)
        }
    }

    @Test
    fun rejectsDerivativeFeatureSchemasUntilUnitNormalizationIsSupported() {
        assertThrows(IllegalArgumentException::class.java) {
            MarketFeatureBuilder.buildLatest(sampleCandles(), MarketFeatureBuilder.baseFeatureNames + "funding_rate")
        }
    }
}
