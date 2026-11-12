import pytest

from eti.evaluation import walk_forward_folds
from eti.portfolio import assess_portfolio


def test_portfolio_enforces_gross_single_asset_and_position_count():
    ok = assess_portfolio(1000, {"BTC/USDT": 50}, "ETH/USDT", 50)
    assert ok.approved
    blocked = assess_portfolio(1000, {"BTC/USDT": 100, "ETH/USDT": 100,
                                     "SOL/USDT": 20, "XRP/USDT": 10,
                                     "ADA/USDT": 5}, "DOGE/USDT", 10)
    assert not blocked.approved
    assert "max_positions_reached" in blocked.reasons


def test_portfolio_rejects_correlated_concentration():
    result = assess_portfolio(1000, {}, "BTC/USDT", 100, correlated_group_exposure=100)
    assert not result.approved
    assert "correlated_group_limit" in result.reasons


def test_walk_forward_folds_are_ordered_and_purged():
    folds = walk_forward_folds(1200, min_train=400, validation_size=100,
                               test_size=100, step_size=100, purge_bars=8)
    assert folds
    for fold in folds:
        assert fold.train_end + 8 <= fold.validation_start
        assert fold.validation_end + 8 <= fold.test_start
        assert fold.train_start < fold.train_end < fold.validation_start
        assert fold.test_start < fold.test_end


def test_walk_forward_rejects_insufficient_history():
    with pytest.raises(ValueError):
        walk_forward_folds(100, min_train=500)


def test_portfolio_rejects_non_finite_and_invalid_position_values():
    import math
    bad_equity = assess_portfolio(math.nan, {}, "BTC/USDT", 10)
    bad_position = assess_portfolio(1000, {"BTC/USDT": math.inf}, "ETH/USDT", 10)
    assert not bad_equity.approved
    assert not bad_position.approved
    assert "invalid_existing_position" in bad_position.reasons
