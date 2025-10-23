import pandas as pd
from types import SimpleNamespace
from strategy.indicators import compute_indicators


def test_compute_indicators_columns():
    # Make a tiny DataFrame
    idx = pd.date_range("2025-01-01", periods=30, freq="5min", tz="Asia/Seoul")
    df = pd.DataFrame({
        "open":  [100 + i*0.1 for i in range(30)],
        "high":  [100 + i*0.1 + 0.5 for i in range(30)],
        "low":   [100 + i*0.1 - 0.5 for i in range(30)],
        "close": [100 + i*0.1 for i in range(30)],
        "volume":[10 for _ in range(30)],
    }, index=idx)
    cfg = SimpleNamespace(std_win=10, band_k_short=2.1, band_k_long=1.9)
    out = compute_indicators(df, cfg)
    for col in ["vwap", "std", "upper_short", "lower_long", "mom10"]:
        assert col in out.columns
