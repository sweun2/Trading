
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# ----------------------
# Technical Indicators
# ----------------------

def stochastic_kd(close, high, low, k_period=14, d_period=3, smooth_k=1):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100.0
    if smooth_k > 1:
        k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    else:
        k = raw_k
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def adx(high, low, close, period=14):
    # Wilder's DMI/ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di) ) * 100
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val, plus_di, minus_di

# ----------------------
# Strategy & Backtest
# ----------------------

@dataclass
class Trade:
    entry_time: str
    direction: str   # 'long' or 'short'
    entry_price: float
    size_frac: float # fraction of equity allocated (1.0 means full equity notional; leverage applied separately)
    add_1: float
    add_2: float
    exit_time: str
    exit_price: float
    pnl_pct_equity: float

def backtest(df,
             k_period=14, d_period=3,
             adx_period=14, adx_threshold=25.0,
             leverage=10.0,
             equity_start=1.0,
             tp_equity_pct=0.05,    # +5% equity take-profit per-position
             dd_stop_equity_pct=0.20,  # -20% monthly circuit breaker
             use_htf_confirm=False,
             htf_minutes=60):
    """
    df must include datetime index (tz-naive ok) and columns: open, high, low, close, volume
    """
    df = df.copy()
    k, d = stochastic_kd(df['close'], df['high'], df['low'], k_period=k_period, d_period=d_period, smooth_k=1)
    df['k'] = k
    df['d'] = d
    df['k_prev'] = df['k'].shift(1)
    df['d_prev'] = df['d'].shift(1)

    adx_val, pdi, mdi = adx(df['high'], df['low'], df['close'], period=adx_period)
    df['adx'] = adx_val

    # Optional HTF confirm: 1h stoch extreme alignment
    if use_htf_confirm:
        htf = df.resample(f'{htf_minutes}T').last()
        hk, hd = stochastic_kd(htf['close'], htf['high'], htf['low'], k_period=k_period, d_period=d_period, smooth_k=1)
        htf['hk'] = hk; htf['hd'] = hd
        df['hk'] = htf['hk'].reindex(df.index).ffill()
        df['hd'] = htf['hd'].reindex(df.index).ffill()
    else:
        df['hk'] = np.nan; df['hd'] = np.nan

    equity = equity_start
    high_equity = equity_start
    circuit_breaker = equity_start * (1 - dd_stop_equity_pct)

    trades = []
    open_pos = None  # track ongoing position with scaling

    # Helper to compute position PnL vs entry VWAP with leverage
    def position_pnl_pct(direction, entry_price, exit_price, size_frac):
        if direction == 'short':
            ret = (entry_price - exit_price) / entry_price
        else:
            ret = (exit_price - entry_price) / entry_price
        # leverage applies to notional; equity impact scales with size_frac * leverage
        return ret * leverage * size_frac

    # State for scaling
    scale_state = {'stage': 0, 'last_peak_k': None, 'last_trough_k': None, 'entry_price_vwap': None, 'size_frac': 0.0, 'direction': None, 'entry_time': None, 'add_1': np.nan, 'add_2': np.nan}

    for ts, row in df.iterrows():
        price = row['close']
        k = row['k']; kp = row['k_prev']
        dval = row['d']
        adx_ok = row['adx'] >= adx_threshold

        # Skip until indicators valid
        if np.isnan(k) or np.isnan(kp) or np.isnan(dval) or np.isnan(row['adx']):
            continue

        # Circuit breaker
        if equity <= circuit_breaker:
            # close any open position at current price, record, then break
            if open_pos is not None:
                pnl_pct = position_pnl_pct(scale_state['direction'], scale_state['entry_price_vwap'], price, scale_state['size_frac'])
                equity += pnl_pct * equity_start  # equity change in absolute (relative to initial equity)
                trades.append(Trade(entry_time=str(scale_state['entry_time']),
                                    direction=scale_state['direction'],
                                    entry_price=scale_state['entry_price_vwap'],
                                    size_frac=scale_state['size_frac'],
                                    add_1=scale_state['add_1'], add_2=scale_state['add_2'],
                                    exit_time=str(ts), exit_price=price,
                                    pnl_pct_equity=pnl_pct))
                open_pos=None
            break

        # Generate signals
        # SHORT scaling: extreme overbought then turning down
        short_first = (kp >= 90.0) and (k < kp) and adx_ok
        # additional: new higher extreme (k_prev > last_peak_k) then turn down again
        short_add = False
        if scale_state['stage']>=1 and scale_state['direction']=='short':
            if kp >= 90.0 and (scale_state['last_peak_k'] is None or kp > scale_state['last_peak_k']) and (k < kp):
                short_add = True

        # LONG scaling: extreme oversold then turning up
        long_first = (kp <= 10.0) and (k > kp) and adx_ok
        long_add = False
        if scale_state['stage']>=1 and scale_state['direction']=='long':
            if kp <= 10.0 and (scale_state['last_trough_k'] is None or kp < scale_state['last_trough_k']) and (k > kp):
                long_add = True

        # Optional HTF confirm (if enabled)
        if use_htf_confirm:
            if short_first or short_add:
                if not (row['hk']>=80):
                    short_first = False; short_add = False
            if long_first or long_add:
                if not (row['hk']<=20):
                    long_first = False; long_add = False

        # Entry/scale logic (max 3 increments of 1/3 each)
        if open_pos is None:
            if short_first:
                scale_state = {'stage': 1, 'last_peak_k': kp, 'last_trough_k': None,
                               'entry_price_vwap': price, 'size_frac': 1/3, 'direction':'short', 'entry_time': ts, 'add_1': np.nan, 'add_2': np.nan}
                open_pos = True
            elif long_first:
                scale_state = {'stage': 1, 'last_peak_k': None, 'last_trough_k': kp,
                               'entry_price_vwap': price, 'size_frac': 1/3, 'direction':'long', 'entry_time': ts, 'add_1': np.nan, 'add_2': np.nan}
                open_pos = True
        else:
            # scaling
            if scale_state['stage']==1 and short_add and scale_state['direction']=='short':
                # VWAP update
                scale_state['entry_price_vwap'] = (scale_state['entry_price_vwap']*(1/3) + price*(1/3*1)) / (2/3)
                scale_state['size_frac'] = 2/3
                scale_state['stage']=2
                scale_state['last_peak_k'] = kp
                scale_state['add_1'] = float(price)
            elif scale_state['stage']==2 and short_add and scale_state['direction']=='short':
                scale_state['entry_price_vwap'] = (scale_state['entry_price_vwap']*(2/3) + price*(1/3)) / 1.0
                scale_state['size_frac'] = 1.0
                scale_state['stage']=3
                scale_state['last_peak_k'] = kp
                scale_state['add_2'] = float(price)
            elif scale_state['stage']==1 and long_add and scale_state['direction']=='long':
                scale_state['entry_price_vwap'] = (scale_state['entry_price_vwap']*(1/3) + price*(1/3)) / (2/3)
                scale_state['size_frac'] = 2/3
                scale_state['stage']=2
                scale_state['last_trough_k'] = kp
                scale_state['add_1'] = float(price)
            elif scale_state['stage']==2 and long_add and scale_state['direction']=='long':
                scale_state['entry_price_vwap'] = (scale_state['entry_price_vwap']*(2/3) + price*(1/3)) / 1.0
                scale_state['size_frac'] = 1.0
                scale_state['stage']=3
                scale_state['last_trough_k'] = kp
                scale_state['add_2'] = float(price)

            # Exit rules
            exit_signal = False
            if scale_state['direction']=='short':
                # condition exit: slow%D <= 20
                if dval <= 20.0:
                    exit_signal = True
            else:
                # long: slow%D >= 80
                if dval >= 80.0:
                    exit_signal = True

            # Take-profit on equity per position (+5% of starting equity)
            if not exit_signal and scale_state['entry_price_vwap'] is not None:
                pnl_pct = ( (scale_state['entry_price_vwap'] - price)/scale_state['entry_price_vwap'] if scale_state['direction']=='short'
                           else (price - scale_state['entry_price_vwap'])/scale_state['entry_price_vwap'] )
                pnl_equity_pct = pnl_pct * leverage * scale_state['size_frac']
                if pnl_equity_pct >= tp_equity_pct:
                    exit_signal = True

            if exit_signal:
                pnl_pct = position_pnl_pct(scale_state['direction'], scale_state['entry_price_vwap'], price, scale_state['size_frac'])
                equity += pnl_pct  # equity measured relative to start=1.0
                trades.append(Trade(entry_time=str(scale_state['entry_time']),
                                    direction=scale_state['direction'],
                                    entry_price=scale_state['entry_price_vwap'],
                                    size_frac=scale_state['size_frac'],
                                    add_1=scale_state['add_1'], add_2=scale_state['add_2'],
                                    exit_time=str(ts), exit_price=price,
                                    pnl_pct_equity=pnl_pct))
                open_pos=None
                scale_state={'stage':0,'last_peak_k':None,'last_trough_k':None,'entry_price_vwap':None,'size_frac':0.0,'direction':None,'entry_time':None,'add_1':float('nan'),'add_2':float('nan')}

    # Close any open at last price
    if open_pos is not None:
        price = df['close'].iloc[-1]
        pnl_pct = position_pnl_pct(scale_state['direction'], scale_state['entry_price_vwap'], price, scale_state['size_frac'])
        trades.append(Trade(entry_time=str(scale_state['entry_time']),
                            direction=scale_state['direction'],
                            entry_price=scale_state['entry_price_vwap'],
                            size_frac=scale_state['size_frac'],
                            add_1=scale_state['add_1'], add_2=scale_state['add_2'],
                            exit_time=str(df.index[-1]), exit_price=price,
                            pnl_pct_equity=pnl_pct))

    # Summary
    if len(trades)==0:
        summary = {
            'trades': 0, 'wins': 0, 'losses': 0, 'winrate': 0.0,
            'avg_win_%': 0.0, 'avg_loss_%': 0.0, 'expectancy_%': 0.0,
            'sum_pnl_%': 0.0
        }
        return trades, summary

    pnl_list = [t.pnl_pct_equity for t in trades]
    wins = [p for p in pnl_list if p>0]
    losses = [p for p in pnl_list if p<=0]

    winrate = len(wins)/len(trades)
    avg_win = np.mean(wins)*100 if wins else 0.0
    avg_loss = np.mean(losses)*100 if losses else 0.0
    expectancy = (winrate*(avg_win/100)) - ((1-winrate)*(abs(avg_loss)/100))
    summary = {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'winrate': round(winrate*100,2),
        'avg_win_%': round(avg_win,3),
        'avg_loss_%': round(avg_loss,3),
        'expectancy_%': round(expectancy*100,3),
        'sum_pnl_%': round(sum(pnl_list)*100,3)
    }
    return trades, summary

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # expected columns: time, open, high, low, close, volume
    # time in ISO or epoch ms
    if 'time' not in df.columns:
        raise ValueError("CSV must contain 'time' column")
    # parse time
    try:
        df['time'] = pd.to_datetime(df['time'], utc=True)
    except:
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.set_index('time').sort_index()
    # standardize columns
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # limit to September 2025 UTC
    df = df.loc['2025-09-01':'2025-09-30 23:59:59']
    return df

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to 15m BTC/USDC CSV for 2025-09 with columns time,open,high,low,close,volume")
    p.add_argument("--adx", type=float, default=25.0, help="ADX threshold")
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--tp", type=float, default=0.05, help="Take profit per position relative to starting equity (e.g., 0.05=+5%)")
    p.add_argument("--ddstop", type=float, default=0.20, help="Monthly circuit breaker drawdown from starting equity (0.20= -20%)")
    p.add_argument("--htf", action="store_true", help="Enable 1h stochastic confirmation")
    args = p.parse_args()

    df = load_data(args.csv)
    trades, summary = backtest(df, adx_threshold=args.adx, leverage=args.leverage, tp_equity_pct=args.tp, dd_stop_equity_pct=args.ddstop, use_htf_confirm=args.htf)

    # Save trades
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    trades_df.to_csv("btc15m_trades_sep2025.csv", index=False)

    # Save summary
    pd.DataFrame([summary]).to_csv("btc15m_report_sep2025.csv", index=False)

    print("Summary:", summary)
    print("Saved trades -> btc15m_trades_sep2025.csv")
    print("Saved report -> btc15m_report_sep2025.csv")

if __name__ == "__main__":
    main()
