# Run ETI training and evaluation in Kaggle

Training is intentionally manual. Running the command below fits the baseline classifier from scratch inside purged nested walk-forward folds. It does not place orders and does not guarantee profitable performance.

## Kaggle notebook

1. Open the repository's canonical notebook: `notebooks/ETI_v0_1_Training.ipynb`.
2. Enable Internet in Kaggle settings to fetch public market candles.
3. Run all cells only when you are ready to start training.
4. The notebook evaluates BTC/USDT and ETH/USDT independently using Binance public spot OHLCV.
5. Download the generated JSON report before the Kaggle session ends.

The notebook clones the repository's `main` branch. Merge the reviewed architecture PR before starting it.

## Manual command

```bash
python -m eti.pipeline --exchange binance --symbols BTC/USDT,ETH/USDT --timeframe 1h --limit 3000 --evaluation nested --output artifacts/eti_nested_walk_forward.json
```

For a single symbol, use `--symbol BTC/USDT`. For a CSV, supply the required columns `timestamp,open,high,low,close,volume` using `--csv path/to/data.csv`; CSV and `--symbols` cannot be combined.

## What gets fitted

- `HistGradientBoostingClassifier`: a tabular model fitted from scratch in each fold.
- `DummyClassifier`: a naive majority-class comparator.
- Candidate selection uses each fold's validation segment. The outer test segment is reserved for reporting.
- Models are not exported or promoted automatically. No strategy is approved for live use.

## Data and compute notes

- Public Binance/MEXC adapters currently cover crypto spot only.
- Other labelled asset classes can be evaluated from local CSV data, but live FX/equity/commodity feeds are not connected.
- Kaggle sessions are temporary and compute limits apply. The notebook uses CPU by default; GPU is not required by the current scikit-learn model.
- Do not put exchange API keys in notebooks. Only public market data is used.


## One-click GitHub Actions run

You can also start the same CPU research job without opening Kaggle:

1. Open the repository's **Actions** tab.
2. Select **ETI manual research training**.
3. Choose Binance or MEXC, symbols, timeframe and candle limit.
4. Tap **Run workflow**.
5. When it completes, open the run's Artifacts section and download `eti-nested-research-report`.

This is a manual action only. CI never starts model fitting automatically. The workflow uses public spot candles, trains candidate models from scratch inside nested folds, and uploads evaluation metrics; it does not place orders.
