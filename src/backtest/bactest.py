# backtest_monthly_grid_dual_asym.py
# - A 방식: LONG/SHORT 동시 운용 (각 1개 보유)
# - 진입 시 equity 기반 복리 반영 (eq *= 1 + ret*lev - fee)
# - TP/SL: 진입 순간 vwap/std로 '고정'
# - 비대칭 파라미터(롱/숏 분리) + 월별 그리드(멀티프로세스, 3^6 = 729 조합)
# - 결과: 월별 CSV 저장 (TP/SL 모두 taker 수수료 가정)

import os, re, time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import ccxt

# ===============================
# 공통 설정
# ===============================
DATA_DIR = "./data"
MONTHLY_DIR = "./monthly_dual_asym_grid"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MONTHLY_DIR).mkdir(parents=True, exist_ok=True)

CHART_TZ = "Asia/Seoul"

START_YM = "2024-01"
END_YM   = "2025-09"

SYMBOL    = "BTC/USDT"
TIMEFRAME = "5m"
FULL_START_DATE = "2024-01-01"
FULL_END_DATE   = "2025-10-01"

MAX_WORKERS: Optional[int] = None   # None → CPU-1
MIN_TRADES_FILTER = 10
TOPN_PRINT = 10
SAVE_TOPK_TRADES_PER_MONTH = 0  # 너무 무거우니 기본 0

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

# ===============================
# 유틸
# ===============================
def parse_tf_minutes(tf: str) -> int:
    m = re.match(r"(\d+)([mhdw])$", tf.strip().lower())
    if not m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    n, u = int(m.group(1)), m.group(2)
    return n if u=="m" else n*60 if u=="h" else n*60*24 if u=="d" else n*60*24*7

def month_iter(start_ym: str, end_ym: str) -> List[str]:
    p = pd.period_range(pd.Period(start_ym, freq="M"), pd.Period(end_ym, freq="M"), freq="M")
    return [str(x) for x in p]

def fetch_full_ohlcv(symbol: str, timeframe: str, start_date: str, end_date: str):
    tf_min = parse_tf_minutes(timeframe)
    fname = os.path.join(DATA_DIR, f"{symbol.replace('/', '')}_{start_date}_{end_date}_{timeframe}.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(CHART_TZ)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index(), fname, tf_min

    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T23:59:59Z")
    rows: List[List[float]] = []
    print(f"📡 {symbol} {timeframe} 다운로드… {start_date} → {end_date}")

    backoff = 1.5
    attempts = 0
    last_ts_seen = None
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as e:
            attempts += 1
            sleep_s = min(30, backoff ** attempts)
            print(f"⚠️ 오류: {e} → {sleep_s:.1f}s 후 재시도 ({attempts})")
            time.sleep(sleep_s)
            continue
        if not ohlcv:
            break
        rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts_seen is not None and last_ts <= last_ts_seen:
            break
        last_ts_seen = last_ts
        since = last_ts + 1
        time.sleep(0.15)
        if last_ts >= end_ts:
            break

    if not rows:
        raise RuntimeError("❌ 다운로드된 데이터가 없습니다.")

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(CHART_TZ)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(fname, index=True)
    print(f"✅ {len(df)} 봉 저장 → {fname}")
    return df[["open","high","low","close","volume"]], fname, tf_min

def save_month_slice(df_full: pd.DataFrame, ym: str, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, str]:
    ms = pd.Timestamp(f"{ym}-01 00:00:00", tz=CHART_TZ)
    me = (ms + pd.offsets.MonthBegin(1))
    dfm = df_full[(df_full.index >= ms) & (df_full.index < me)].copy()
    out = os.path.join(MONTHLY_DIR, f"{symbol.replace('/', '')}_{ym}_{timeframe}.csv")
    dfm.to_csv(out, index=True)
    return dfm, out

# ===============================
# 파라미터 (롱/숏 비대칭)
# ===============================
@dataclass
class BacktestParams:
    # 공통 인디케이터 창
    std_win: int = 12

    # 밴드(사이드별)
    band_k_long: float = 1.73
    band_k_short: float = 1.90

    # SL 배수(사이드별)
    stop_k_long: float = 3.0
    stop_k_short: float = 3.0

    # 진입 허용 최소 변동성(사이드별)
    std_min_long: float = 0.002
    std_min_short: float = 0.002

    # 모멘텀 필터(사이드별)
    momentum_threshold_long: float = -0.0045
    momentum_threshold_short: float = 0.0045

    # 타임스탑(사이드별)
    time_stop_min_long: int = 45
    time_stop_min_short: int = 45

    # 트리거/필터/자본
    trigger_mode: str = "intrabar"  # "intrabar" | "close"
    kst_exclude_start: int = 0
    kst_exclude_end: int = 6
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    account_init: float = 10_000.0
    leverage: float = 6.0
    max_daily_dd_pct: float = 0.05

    # 출력/저장
    print_trades: bool = False
    print_summary: bool = False
    trade_csv_path: str = ""
    period_label: str = ""

def coerce_param_types(p: BacktestParams) -> BacktestParams:
    for k in ("std_win","time_stop_min_long","time_stop_min_short","kst_exclude_start","kst_exclude_end"):
        setattr(p, k, int(round(float(getattr(p, k)))))
    for k in ("band_k_long","band_k_short","stop_k_long","stop_k_short",
              "std_min_long","std_min_short","momentum_threshold_long","momentum_threshold_short",
              "maker_fee","taker_fee","account_init","leverage","max_daily_dd_pct"):
        setattr(p, k, float(getattr(p, k)))
    if getattr(p, "trigger_mode", None) not in ("intrabar","close"):
        p.trigger_mode = "intrabar"
    return p

# ===============================
# 인디케이터 (TF 1h VWAP)
# ===============================
def compute_indicators(df: pd.DataFrame, p: BacktestParams, tf_min: int) -> pd.DataFrame:
    out = df.copy()
    w = int(max(1, round(60 / tf_min)))
    pv = (out["close"] * out["volume"]).rolling(w).sum()
    vv = out["volume"].rolling(w).sum()
    eps = 1e-12
    out["vwap"] = np.where(vv > eps, pv / vv, np.nan)

    std_win = int(round(float(p.std_win)))
    min_per = int(max(4, std_win // 2))
    out["std"] = out["close"].rolling(std_win, min_periods=min_per).std()

    # 밴드: 롱/숏 별도
    out["upper_short"] = out["vwap"] + float(p.band_k_short) * out["std"]
    out["lower_long"]  = out["vwap"] - float(p.band_k_long)  * out["std"]
    out["mom10"] = out["close"].pct_change(10)
    return out.dropna()

# ===============================
# 듀얼(A) — 비대칭 파라미터
# ===============================
def run_backtest_dual_asym(df: pd.DataFrame, p: BacktestParams, tf_min: int) -> Dict[str, Any]:
    p = coerce_param_types(p)
    base = compute_indicators(df, p, tf_min)
    if base.empty:
        return {"trades": 0, "winrate": 0.0, "pf": 0.0, "final_eq": p.account_init, "mdd": 0.0, "max_consec_sl": 0}

    long_pos = None   # dict(entry_px, entry_ts, tp_px, sl_px)
    short_pos = None

    eq = float(p.account_init)
    peak = eq
    max_dd = 0.0

    trades = wins = 0
    sum_gain = 0.0
    sum_loss_abs = 0.0
    consec_sl = 0
    max_consec_sl = 0

    trades_rows = []

    day_start_eq = eq
    last_day = base.index[0].date()
    bars_stop_long  = int(max(1, round(p.time_stop_min_long  / tf_min)))
    bars_stop_short = int(max(1, round(p.time_stop_min_short / tf_min)))

    def enter_long(ts, price, vwap, stdv):
        return {"entry_px": price, "entry_ts": ts, "tp_px": vwap, "sl_px": vwap - p.stop_k_long * stdv}

    def enter_short(ts, price, vwap, stdv):
        return {"entry_px": price, "entry_ts": ts, "tp_px": vwap, "sl_px": vwap + p.stop_k_short * stdv}

    def close_position(side: str, pos: dict, ts, exit_px, reason: str, stdv, mom10):
        nonlocal eq, trades, wins, sum_gain, sum_loss_abs, consec_sl, max_consec_sl
        entry_px = pos["entry_px"]
        pnl_raw = (exit_px - entry_px) / entry_px if side == "long" else (entry_px - exit_px) / entry_px
        pnl_lev = pnl_raw * p.leverage

        # ✅ TP/SL 모두 taker 수수료 가정
        eq *= (1 + pnl_lev - p.taker_fee)

        trades += 1
        if pnl_lev > 0:
            wins += 1
            sum_gain += pnl_lev
            consec_sl = 0
        else:
            sum_loss_abs += (-pnl_lev)
            if reason == "SL":
                consec_sl += 1
                max_consec_sl = max(max_consec_sl, consec_sl)
            else:
                consec_sl = 0

        hold_min = (ts - pos["entry_ts"]).total_seconds() / 60.0
        if p.trade_csv_path:
            trades_rows.append({
                "exit_ts": ts, "side": side,
                "entry_ts": pos["entry_ts"], "holding_min": round(hold_min,1),
                "entry_price": round(entry_px,2), "exit_price": round(exit_px,2),
                "tp_px": round(pos["tp_px"],2), "sl_px": round(pos["sl_px"],2),
                "pnl_pct": round(pnl_lev*100, 6),  # 참고: 여기의 pnl_pct는 수수료 제외된 순수 레버리지 수익률임
                "equity_after": round(eq,6),
                "exit_reason": reason, "std_val": round(stdv,6),
                "momentum": None if np.isnan(mom10) else round(mom10,6),
                "period": p.period_label,
            })

    for ts, r in base.iterrows():
        # 데일리 경계/가드
        if ts.date() != last_day:
            day_start_eq = eq
            consec_sl = 0
            last_day = ts.date()
        day_dd = (eq - day_start_eq) / max(1e-12, day_start_eq)
        if day_dd <= -p.max_daily_dd_pct:
            price = float(r["close"])
            stdv  = float(r["std"])
            mom10 = float(r["mom10"]) if not pd.isna(r["mom10"]) else np.nan
            if long_pos:  close_position("long", long_pos, ts, price, "DAILY_KILL", stdv, mom10);  long_pos = None
            if short_pos: close_position("short", short_pos, ts, price, "DAILY_KILL", stdv, mom10); short_pos = None
            break

        # 심야 제외
        if p.kst_exclude_start <= ts.hour < p.kst_exclude_end:
            continue

        price = float(r["close"])
        vwap  = float(r["vwap"])
        stdv  = float(r["std"])
        upper_short = float(r["upper_short"])
        lower_long  = float(r["lower_long"])
        mom10 = float(r["mom10"]) if not pd.isna(r["mom10"]) else np.nan
        bar_high = float(r["high"]); bar_low = float(r["low"])

        # ===== 진입(양방향 동시 허용) =====
        if not np.isnan(stdv):
            if long_pos is None and stdv >= p.std_min_long:
                go_long = (price < lower_long) and (np.isnan(mom10) or mom10 >= p.momentum_threshold_long)
                if go_long:
                    long_pos = enter_long(ts, price, vwap, stdv)
            if short_pos is None and stdv >= p.std_min_short:
                go_short = (price > upper_short) and (np.isnan(mom10) or mom10 <= p.momentum_threshold_short)
                if go_short:
                    short_pos = enter_short(ts, price, vwap, stdv)

        # ===== 종료: LONG =====
        if long_pos is not None:
            exit_hit = None; exit_px = None
            if p.trigger_mode == "intrabar":
                if bar_low <= long_pos["sl_px"]:   exit_hit, exit_px = "SL", long_pos["sl_px"]
                elif bar_high >= long_pos["tp_px"]:exit_hit, exit_px = "TP", long_pos["tp_px"]
            else:
                if price <= long_pos["sl_px"]:     exit_hit, exit_px = "SL", price
                elif price >= long_pos["tp_px"]:   exit_hit, exit_px = "TP", price
            if exit_hit is None and p.time_stop_min_long:
                bars_held = int((ts - long_pos["entry_ts"]).total_seconds() // (60 * tf_min))
                if bars_held >= bars_stop_long:    exit_hit, exit_px = "TIME", price
            if exit_hit is not None:
                close_position("long", long_pos, ts, exit_px, exit_hit, stdv, mom10)
                long_pos = None

        # ===== 종료: SHORT =====
        if short_pos is not None:
            exit_hit = None; exit_px = None
            if p.trigger_mode == "intrabar":
                if bar_high >= short_pos["sl_px"]: exit_hit, exit_px = "SL", short_pos["sl_px"]
                elif bar_low <= short_pos["tp_px"]:exit_hit, exit_px = "TP", short_pos["tp_px"]
            else:
                if price >= short_pos["sl_px"]:    exit_hit, exit_px = "SL", price
                elif price <= short_pos["tp_px"]:  exit_hit, exit_px = "TP", price
            if exit_hit is None and p.time_stop_min_short:
                bars_held = int((ts - short_pos["entry_ts"]).total_seconds() // (60 * tf_min))
                if bars_held >= bars_stop_short:   exit_hit, exit_px = "TIME", price
            if exit_hit is not None:
                close_position("short", short_pos, ts, exit_px, exit_hit, stdv, mom10)
                short_pos = None

        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq)/peak if peak>0 else 0.0)

    winrate = (wins / trades) if trades > 0 else 0.0
    pf = (sum_gain / sum_loss_abs) if sum_loss_abs > 0 else (float("inf") if trades>0 else 0.0)
    if p.trade_csv_path and trades_rows:
        pd.DataFrame(trades_rows).to_csv(p.trade_csv_path, index=False)
    return {"trades": trades, "winrate": winrate, "pf": pf, "final_eq": eq, "mdd": max_dd, "max_consec_sl": max_consec_sl}

# ===============================
# 그리드/워커 (비대칭 축 구성)
# ===============================
RAW_DF_PATH: Optional[str] = None
GLOBAL_TF_MIN: Optional[int] = None

def worker_init(df_path: str, tf_min: int):
    global RAW_DF_PATH, GLOBAL_TF_MIN
    RAW_DF_PATH = df_path
    GLOBAL_TF_MIN = tf_min

def load_df_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(CHART_TZ)
    df.set_index("timestamp", inplace=True)
    return df[["open","high","low","close","volume"]].sort_index()

def grid_product(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for vals in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))

def _safe_eval_combo(combo: Dict[str, Any], base_params: BacktestParams) -> Dict[str, Any]:
    try:
        assert RAW_DF_PATH and GLOBAL_TF_MIN
        df = load_df_from_path(RAW_DF_PATH)
        p = replace(base_params)
        p.print_trades = False
        p.print_summary = False
        p.trade_csv_path = ""  # 그리드 중엔 거래 로그 저장 안 함

        # pair 축을 분해해 주입
        for k, v in combo.items():
            if k == "stop_k_pair":
                p.stop_k_long, p.stop_k_short = float(v[0]), float(v[1])
            elif k == "std_min_pair":
                p.std_min_long, p.std_min_short = float(v[0]), float(v[1])
            elif k == "time_stop_pair":
                p.time_stop_min_long, p.time_stop_min_short = int(round(float(v[0]))), int(round(float(v[1])))
            elif k == "momentum_pair":
                p.momentum_threshold_long, p.momentum_threshold_short = float(v[0]), float(v[1])
            else:
                setattr(p, k, v)

        p = coerce_param_types(p)
        res = run_backtest_dual_asym(df, p, GLOBAL_TF_MIN)

        out = {**combo, **res}
        # pair들을 펼쳐 컬럼으로 담기
        if "stop_k_pair" in combo:
            out["stop_k_long"], out["stop_k_short"] = combo["stop_k_pair"]
        if "std_min_pair" in combo:
            out["std_min_long"], out["std_min_short"] = combo["std_min_pair"]
        if "time_stop_pair" in combo:
            out["time_stop_min_long"], out["time_stop_min_short"] = combo["time_stop_pair"]
        if "momentum_pair" in combo:
            out["momentum_threshold_long"], out["momentum_threshold_short"] = combo["momentum_pair"]
        out["ok"] = True
        return out
    except Exception as e:
        out = {**combo, "ok": False, "error": str(e)}
        return out

def run_grid_multiproc_for_month(
    df_csv_path: str,
    tf_min: int,
    base_params: BacktestParams,
    grid: Dict[str, List[Any]],
    out_csv: str,
    min_trades_filter: int = 10,
    topn_print: int = 10,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    combos = list(grid_product(grid))
    max_workers = workers or max(1, os.cpu_count()-1)
    results: List[Dict[str, Any]] = []
    err_cnt = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, initargs=(df_csv_path, tf_min)) as ex:
        futs = [ex.submit(_safe_eval_combo, c, base_params) for c in combos]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if not r.get("ok", False):
                err_cnt += 1
            results.append(r)
            if i % 20 == 0:
                print(f"… dual-asym {Path(df_csv_path).stem} progress: {i}/{len(combos)} (errors={err_cnt})")

    df_res = pd.DataFrame(results)
    ok_df = df_res[df_res["ok"] == True].drop(columns=["ok"])
    err_df = df_res[df_res["ok"] == False].drop(columns=["ok"]) if "ok" in df_res else pd.DataFrame()

    if not ok_df.empty:
        ok_df = ok_df.assign(pf_sort=np.where(np.isinf(ok_df["pf"]), 1e9, ok_df["pf"]))
        ok_df = ok_df.sort_values(["pf_sort","winrate","final_eq"], ascending=[False, False, False]).drop(columns=["pf_sort"])
    ok_df.to_csv(out_csv, index=False)
    print(f"📄 saved: {out_csv} (ok={len(ok_df)}, err={len(err_df)})")

    show = ok_df
    if "trades" in show.columns:
        show = show[show["trades"] >= min_trades_filter]
    print(f"\n===== Top {topn_print} (dual-asym, trades≥{min_trades_filter}) for {Path(df_csv_path).stem} =====")
    if show.empty:
        print("(no results)")
    else:
        cols = [c for c in [
            "band_k_long","band_k_short","std_win",
            "stop_k_long","stop_k_short",
            "std_min_long","std_min_short",
            "time_stop_min_long","time_stop_min_short",
            "momentum_threshold_long","momentum_threshold_short",
            "pf","winrate","final_eq","mdd","trades","max_consec_sl"
        ] if c in show.columns]
        print(show.head(topn_print)[cols].to_string(index=False))

    return ok_df

# ===============================
# 실행
# ===============================
if __name__ == "__main__":
    # 1) 전체 데이터 1회 다운로드
    df_full, full_csv_path, tf_min = fetch_full_ohlcv(SYMBOL, TIMEFRAME, FULL_START_DATE, FULL_END_DATE)

    # 2) 공통 베이스 파라미터 (주의: 결과 계산은 TP/SL 모두 taker 수수료로 반영)
    base = BacktestParams(
        std_win=12,
        band_k_long=1.73, band_k_short=1.90,
        stop_k_long=3.0, stop_k_short=3.0,
        std_min_long=0.002, std_min_short=0.002,
        momentum_threshold_long=-0.0045, momentum_threshold_short=0.0045,
        time_stop_min_long=45, time_stop_min_short=45,
        trigger_mode="intrabar",
        kst_exclude_start=0, kst_exclude_end=6,
        maker_fee=0.0, taker_fee=0.0005,
        account_init=10_000.0, leverage=6.0,
        max_daily_dd_pct=0.05,
        print_trades=False, print_summary=False,
    )
    base = coerce_param_types(base)

    # 3) 비대칭 듀얼 그리드 (3^6 = 729 조합)
    grid_dual_asym = {
        "band_k_long":  [1.60, 1.73, 1.90],        # 롱 밴드
        "band_k_short": [1.73, 1.90, 2.10],        # 숏 밴드 (1.9 근방 확장)
        "std_win":      [10, 12, 16],              # 공통 STD 윈도우

        # 아래 3개는 (롱, 숏) 페어로 묶어서 '한 축'으로 카운트 → 3^6 유지
        "stop_k_pair":  [(2.5,3.0), (3.0,3.0), (3.0,3.5)],
        "std_min_pair": [(0.0015,0.0015), (0.0020,0.0020), (0.0020,0.0025)],
        "time_stop_pair": [(30,30), (45,30), (60,45)],

        # 모멘텀도 페어(참고: 개별 축으로 빼면 조합 수 급증)
        "momentum_pair": [(-0.006,0.006), (-0.0045,0.0045), (-0.003,0.003)],
    }

    months = month_iter(START_YM, END_YM)
    print(f"\n🗓 Months: {len(months)} ({months[0]} → {months[-1]})")
    print(f"Grid size (dual-asym): {len(list(grid_product(grid_dual_asym)))}")

    for ym in months:
        print(f"\n====================  {ym}  ====================")
        df_m, monthly_csv = save_month_slice(df_full, ym, SYMBOL, TIMEFRAME)
        if df_m.empty or len(df_m) < 50:
            print(f"(skip {ym}) 데이터가 부족합니다.")
            continue

        out_dual = os.path.join(MONTHLY_DIR, f"grid_dual_asym_results_{ym}.csv")
        ok_dual = run_grid_multiproc_for_month(
            df_csv_path=monthly_csv, tf_min=tf_min, base_params=base,
            grid=grid_dual_asym, out_csv=out_dual,
            min_trades_filter=MIN_TRADES_FILTER, topn_print=TOPN_PRINT, workers=MAX_WORKERS
        )

        # (선택) Top-K 거래 로그 저장 — 비교용. 월 내 초기값은 고정 10k 사용.
        if SAVE_TOPK_TRADES_PER_MONTH > 0 and not ok_dual.empty:
            show = ok_dual
            if "trades" in show.columns:
                show = show[show["trades"] >= MIN_TRADES_FILTER]
            show = show.head(SAVE_TOPK_TRADES_PER_MONTH)
            i = 1
            for _, row in show.iterrows():
                p = replace(base)
                for k in ["band_k_long","band_k_short","std_win",
                          "stop_k_long","stop_k_short",
                          "std_min_long","std_min_short",
                          "time_stop_min_long","time_stop_min_short",
                          "momentum_threshold_long","momentum_threshold_short"]:
                    if k in row and not pd.isna(row[k]):
                        setattr(p, k, float(row[k]) if k not in ("std_win","time_stop_min_long","time_stop_min_short") else int(round(float(row[k]))))
                p.trade_csv_path = os.path.join(MONTHLY_DIR, f"trades_dual_asym_top{i}_{ym}.csv")
                p.period_label = f"dual_asym_top{i}_{ym}"
                p.print_summary = True
                p.account_init = 10_000.0
                p = coerce_param_types(p)
                print(f"💾 saving trades for dual-asym top{i} — {ym}")
                run_backtest_dual_asym(df_m, p, tf_min)
                i += 1

    print("\n✅ Monthly dual-asym grid runs finished.")
