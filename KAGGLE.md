# Run ETI in Kaggle

1. Create a new Kaggle Notebook (Python).
2. Enable Internet in notebook settings if you want to fetch public exchange candles.
3. In a notebook cell, run:

```python
!git clone https://github.com/hamza00alik-source/eti-research-core.git
%cd eti-research-core
!pip install -r requirements.txt
!python -m eti.pipeline --exchange binance --symbol BTC/USDT --timeframe 1h --limit 3000
```

For MEXC, replace `--exchange binance` with `--exchange mexc`.

To use a CSV instead, upload a file with columns `timestamp,open,high,low,close,volume`, then run:

```python
!python -m eti.pipeline --csv /kaggle/working/your_data.csv
```

Kaggle sessions are temporary. Download or save any artifacts you want to keep before the session ends. Do not put exchange API keys in a notebook; this milestone uses public data only.
