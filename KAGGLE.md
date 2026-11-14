# Run ETI training and evaluation in Kaggle

Training is intentionally manual. Running the command below fits the baseline classifier from scratch inside purged nested walk-forward folds. It does not place orders and does not guarantee profitable performance.

## Kaggle notebook

1. Open the repository's canonical notebook: `notebooks/ETI_v0_1_Training.ipynb`.
2. Enable Internet in Kaggle settings to fetch public market candles.
3. Run all cells only when you are ready to start training.
4. The budget-screening notebook evaluates BTC/USDT only using Binance public spot OHLCV.
5. Download the generated JSON report before the Kaggle session ends.

The notebook clones the repository's `main` branch. Merge the reviewed architecture PR before starting it.

## Manual command

```bash
python -m eti.pipeline --exchange binance --symbol BTC/USDT --timeframe 1h --limit 2000 --horizon 6 --min-train 400 --validation-size 100 --test-size 100 --step-size 200 --max-model-fits 32 --evaluation nested --output artifacts/eti_budget_screen.json
```

For a single symbol, use `--symbol BTC/USDT`. For a CSV, supply the required columns `timestamp,open,high,low,close,volume` using `--csv path/to/data.csv`; CSV and `--symbols` cannot be combined.

## What gets fitted

- `HistGradientBoostingClassifier`: a tabular model fitted from scratch in each fold.
- `DummyClassifier`: a naive majority-class comparator.
- Candidate selection uses each fold's validation segment. The outer test segment is reserved for reporting.
- Models are not exported or promoted automatically. No strategy is approved for live use.

## Budget-aware training policy

The user's hard planning ceilings are USD 10 per day and USD 100 per month. These are spending limits, not targets. This repository cannot read Kaggle billing or remaining quota, so it must never claim to calculate or enforce actual dollar spend. Check Kaggle's own usage/billing display before starting a run and stop if the run could exceed either ceiling.

The canonical first-stage screening run is CPU-only, one market (BTC/USDT), 1h candles, 2,000 candles, horizon 6, 400 minimum training rows, 100-row validation/test windows, and a 200-row step. It caps the estimated training workload at 32 model-fit calls before any model fitting begins. A fit-call count is only a compute proxy, not a dollar-cost estimate.

Do not run multi-symbol or hyperparameter sweeps in the screening stage. Do not rerun the same experiment just to regenerate a report. Save the JSON output and record the Kaggle run URL. A second-stage multi-market or larger-window experiment requires review of the first-stage results and an explicit budget check. No automatic retraining is enabled.

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
