from typing import Any
import pandas as pd


def fetch_latest_ohlcv(ex: Any, symbol: str, timeframe: str, limit: int = 240) -> pd.DataFrame:
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=[
        "timestamp","open","high","low","close","volume"
    ])
    # UTC → KST로 변환해 인덱스를 저장
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    df.set_index("timestamp", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    # 1h VWAP on 5m bars
    w = int(60/5)
    pv = (df["close"] * df["volume"]).rolling(w).sum()
    vv = df["volume"].rolling(w).sum()
    vwap = pv / vv
    std = df["close"].rolling(cfg.std_win, min_periods=max(4, cfg.std_win//2)).std()

    # 비대칭 밴드
    upper_short = vwap + cfg.band_k_short * std
    lower_long  = vwap - cfg.band_k_long  * std

    out = df.copy()
    out["vwap"] = vwap
    out["std"]  = std
    out["upper_short"] = upper_short
    out["lower_long"]  = lower_long
    out["mom10"] = df["close"].pct_change(10)
    return out.dropna()
