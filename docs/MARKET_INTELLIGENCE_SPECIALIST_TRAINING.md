# Market-intelligence specialist training

```python
from eti.market_intelligence_training import train_market_intelligence_specialists

report = train_market_intelligence_specialists(
    ohlcv_frame,
    horizon=6,
    min_train=500,
    validation_size=100,
    test_size=100,
    step_size=100,
    max_model_fits=32,
)
```

This research-only pipeline evaluates two distinct roles:

- `regime_classifier`: predicts the causal market regime observed `horizon` bars ahead.
- `volatility_estimator`: estimates standard deviation of the next `horizon` log returns.

Features at time t use only the candle at t and prior candles. Targets are forward-looking by design. Expanding chronological folds reserve a purge gap at least as long as the target horizon before each outer test window. The test windows are never used to select a candidate. A majority-class classifier and median-volatility regressor provide naive baselines. Regime folds report support, predicted counts, recall and class coverage for every fixed regime label. If an outer test fold contains only one observed class, balanced accuracy is reported as `null` with `insufficient_test_class_coverage`; it is excluded from the mean rather than being presented as a meaningful balanced score. The report also counts folds with valid balanced-accuracy evidence.

The fit budget is checked before fitting. Each fold uses one model and one naive baseline for each role (four fit calls per fold). The report carries code/config/data fingerprints and an experiment identifier.

**Important limits:** this first implementation returns research reports, not serialized models. It does not write to the model registry, promote candidates, enable inference, or grant the Queen authority. A single instrument's folds are not independent replications. Before any promotion, run the same role-specific pipeline across multiple eligible instruments and regimes, review calibration/uncertainty, class coverage and baseline comparisons, and obtain independent approval. Live execution remains disabled.
