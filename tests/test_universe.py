from types import SimpleNamespace

import pandas as pd
import pytest

from eti.instruments import InstrumentSpec
from eti.universe import scan_market_universe


class FakeHub:
    def __init__(self):
        self.items = {
            "crypto": [
                InstrumentSpec("BTC/USDT", "binance", "crypto", "spot", "BTC", "USDT"),
                InstrumentSpec("ETH/USDT", "binance", "crypto", "spot", "ETH", "USDT"),
                InstrumentSpec("BTC/USD", "coinbaseexchange", "crypto", "spot", "BTC", "USD"),
            ],
            "equity": [
                InstrumentSpec("AAPL", "yahoo", "equity", "spot"),
                InstrumentSpec("MSFT", "yahoo", "equity", "spot"),
            ],
        }

    def discover(self, *, asset_class=None, market_type=None):
        items = self.items.get(asset_class, [])
        return {
            "instruments": items, "count": len(items),
            "sources_checked": sorted({item.venue for item in items}),
            "source_errors": [],
        }

    def fetch(self, instrument, timeframe, limit, *, min_rows=1):
        if instrument.symbol == "ETH/USDT":
            raise RuntimeError("provider temporarily unavailable")
        return SimpleNamespace(
            provenance={"venue": instrument.venue, "symbol": instrument.symbol,
                        "returned_rows": min_rows},
            quality={"rows": min_rows, "ohlcv_validated": True},
        )


def test_scan_market_universe_is_bounded_and_round_robins_venues():
    report = scan_market_universe(
        asset_classes=("crypto", "equity"), market_types=("spot",),
        timeframe="1h", limit=1000, min_rows=500,
        max_per_asset_class=2, max_per_venue=2, hub=FakeHub(),
    )
    crypto = [
        item["instrument"]["venue"] for item in report["candidates"]
        if item["instrument"]["asset_class"] == "crypto"
    ]
    assert crypto == ["binance", "coinbaseexchange"]
    assert report["summary"]["candidate_count"] == 4
    assert report["summary"]["data_ready_count"] == 3
    assert report["scan_only"] is True
    assert report["training_performed"] is False
    assert report["execution_enabled"] is False


def test_scan_records_provider_failures_without_hiding_other_candidates():
    report = scan_market_universe(
        asset_classes=("crypto",), market_types=("spot",),
        timeframe="1h", limit=1000, min_rows=500,
        max_per_asset_class=3, max_per_venue=2, hub=FakeHub(),
    )
    failed = [item for item in report["candidates"] if item["status"] == "unavailable"]
    assert len(failed) == 1
    assert failed[0]["error"]["message"] == "provider temporarily unavailable"
    assert report["summary"]["data_ready_count"] == 2


@pytest.mark.parametrize("kwargs", [
    {"asset_classes": ()},
    {"asset_classes": ("crypto", "unknown")},
    {"market_types": ()},
    {"limit": 10, "min_rows": 11},
    {"max_per_asset_class": 0},
])
def test_scan_rejects_invalid_universe_limits(kwargs):
    params = {
        "asset_classes": ("crypto",), "market_types": ("spot",),
        "timeframe": "1h", "limit": 100, "min_rows": 50,
        "hub": FakeHub(),
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        scan_market_universe(**params)


def test_pipeline_cli_writes_scan_report_without_training(monkeypatch, tmp_path):
    import json
    import sys
    import eti.pipeline as pipeline

    output = tmp_path / "market-universe.json"
    captured = {}
    def fake_scan(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "eti.market_universe.v1",
            "scan_only": True, "training_performed": False,
            "execution_enabled": False, "summary": {"data_ready_count": 1},
        }

    monkeypatch.setattr(pipeline, "scan_market_universe", fake_scan)
    monkeypatch.setattr(sys, "argv", [
        "eti.pipeline", "--scan-universe", "--asset-classes", "crypto,equity",
        "--market-types", "spot", "--limit", "1000",
        "--min-market-rows", "500", "--output", str(output),
    ])
    pipeline.main()
    report = json.loads(output.read_text())
    assert report["scan_only"] is True
    assert report["training_performed"] is False
    assert captured["asset_classes"] == ["crypto", "equity"]
    assert captured["market_types"] == ["spot"]
    assert captured["min_rows"] == 500
