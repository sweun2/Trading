from __future__ import annotations
import time, json, math
from pathlib import Path
from typing import Tuple, Optional
from ccxt.base.errors import InvalidNonce  # ← 추가

from src.utils.time_utils import now_kst

LOCK_STATE = Path("logs/runtime_guard_state.json")

def sync_time(ex, logger):
    """Best-effort: adjust for time difference with the exchange."""
    try:
        ex.options = getattr(ex, 'options', {}) or {}
        ex.options['adjustForTimeDifference'] = True
        ex.options['recvWindow'] = 60000
        ex.load_time_difference()
        logger.info(f"[TIME] synced. offset(ms)={getattr(ex, 'timeDifference', 0)}")
    except Exception as e:
        logger.warning(f"[TIME] sync failed: {e}")


def _save_lock_until(ts_iso: str) -> None:
    try:
        LOCK_STATE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_STATE.write_text(json.dumps({"locked_until": ts_iso}, ensure_ascii=False, indent=2))
    except Exception: pass

def read_lock_until() -> Optional[str]:
    try:
        data = json.loads(LOCK_STATE.read_text())
        return data.get("locked_until")
    except Exception:
        return None

def startup_preflight(ex, cfg, logger) -> None:
    """Cancel open orders at startup and optionally abort if state isn't flat."""
    try:
        if cfg.STARTUP_CANCEL_OPEN_ORDERS:
            try:
                ex.cancel_all_orders(cfg.SYMBOL)
            except Exception:
                # fallback: manual
                for o in ex.fetch_open_orders(symbol=cfg.SYMBOL):
                    try:
                        ex.cancel_order(o['id'], cfg.SYMBOL)
                    except Exception as e:
                        logger.warning(f"cancel failed: {e}")
        if cfg.REQUIRE_STATE_RECOVERY:
            oo = ex.fetch_open_orders(symbol=cfg.SYMBOL)
            if len(oo) > 0:
                raise RuntimeError(f"Open orders remain after startup cancel: {len(oo)}")
    except InvalidNonce as e:
        # Binance -1021: local clock ahead of server
        if 'ahead of the server' in str(e) or '-1021' in str(e):
            logger.warning("[TIME] -1021 detected, syncing time and retrying startup_preflight...")
            sync_time(ex, logger)
            time.sleep(0.2)
            # retry once
            if cfg.STARTUP_CANCEL_OPEN_ORDERS:
                try:
                    ex.cancel_all_orders(cfg.SYMBOL)
                except Exception:
                    for o in ex.fetch_open_orders(symbol=cfg.SYMBOL):
                        try:
                            ex.cancel_order(o['id'], cfg.SYMBOL)
                        except Exception as e2:
                            logger.warning(f"cancel failed: {e2}")
            if cfg.REQUIRE_STATE_RECOVERY:
                oo = ex.fetch_open_orders(symbol=cfg.SYMBOL)
                if len(oo) > 0:
                    raise RuntimeError(f"Open orders remain after startup cancel: {len(oo)}")
        else:
            logger.error(f"[STARTUP GUARD] {e}")
            raise
    except Exception as e:
        logger.error(f"[STARTUP GUARD] {e}")
        raise

def price_protect_ok(ex, symbol: str, cfg, logger) -> Tuple[bool,str]:
    """Check mark-last deviation and spread/depth against thresholds."""
    try:
        ticker = ex.fetch_ticker(symbol)
        last = float(ticker.get('last') or ticker.get('close') or 0.0)
        mark = last
        try:
            # best-effort: ccxt sometimes exposes mark via 'info'
            info = ticker.get('info') or {}
            m = info.get('markPrice') or info.get('mark_price')
            if m: mark = float(m)
        except Exception:
            pass
        if cfg.TRIGGER_PRICE_SOURCE == 'mark':
            ref = mark
        else:
            ref = last
        if last > 0 and ref > 0:
            dev = abs(last - ref) / ref
            if dev > cfg.PRICE_PROTECT_MAX_DEVIATION_PCT:
                return False, f"price deviation {dev:.4%} > {cfg.PRICE_PROTECT_MAX_DEVIATION_PCT:.2%}"
        ob = ex.fetch_order_book(symbol, limit=5)
        bids = ob.get('bids') or []
        asks = ob.get('asks') or []
        if not bids or not asks:
            return False, "empty orderbook"
        best_bid, bid_sz = bids[0][0], bids[0][1]
        best_ask, ask_sz = asks[0][0], asks[0][1]
        spread = (best_ask - best_bid) / max(1e-12, (best_ask + best_bid)/2)
        if spread > cfg.MAX_SPREAD_PCT:
            return False, f"spread {spread:.4%} > {cfg.MAX_SPREAD_PCT:.2%}"
        # Notional depth check at top level
        mid = (best_ask + best_bid)/2
        if mid * min(bid_sz, ask_sz) < cfg.MIN_BBO_SIZE_NOTIONAL:
            return False, f"min BBO notional {mid*min(bid_sz, ask_sz):.2f} < {cfg.MIN_BBO_SIZE_NOTIONAL}"
        return True, "ok"
    except Exception as e:
        logger.warning(f"[GUARD] price_protect check error: {e}")
        return False, f"error: {e}"

def set_daily_lock(hours: int) -> None:
    if hours and hours > 0:
        until = now_kst() + __import__('datetime').timedelta(hours=hours)
        _save_lock_until(until.isoformat())

def locked_now() -> bool:
    ts = read_lock_until()
    if not ts: return False
    try:
        t = __import__('datetime').datetime.fromisoformat(ts)
        return now_kst() < t
    except Exception:
        return False
