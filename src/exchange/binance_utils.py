from typing import Optional, Tuple
import ccxt


def make_exchange(cfg) -> ccxt.binance:
    params = {
        "apiKey": cfg.API_KEY,
        "secret": cfg.API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    ex = ccxt.binance(params)
    if cfg.TESTNET:
        ex.set_sandbox_mode(True)
        ex.urls["api"]["fapi"] = "https://testnet.binancefuture.com"
    return ex


def safe_set_margin_and_leverage(ex, symbol: str, leverage: int = 20, margin_mode: str = "cross") -> None:
    # margin mode
    try:
        ex.set_margin_mode(margin_mode, symbol)
        print(f"[OK] margin mode -> {margin_mode} for {symbol}")
    except Exception as e:  # noqa: BLE001
        print("[WARN] set_margin_mode failed:", e)
        try:
            market_id = ex.market(symbol)["id"]
            mt = "CROSSED" if margin_mode.lower().startswith("cross") else "ISOLATED"
            fn = getattr(ex, "fapiPrivatePostMarginType", None) or getattr(ex, "fapiPrivate_post_margintype", None)
            if fn:
                fn({"symbol": market_id, "marginType": mt})
                print(f"[OK] fallback marginType -> {mt} for {symbol}")
        except Exception as e2:  # noqa: BLE001
            print("[FAIL] fallback marginType failed:", e2)
    # leverage
    try:
        ex.set_leverage(leverage, symbol)
        print(f"[OK] leverage -> {leverage}x for {symbol}")
    except Exception as e:  # noqa: BLE001
        print("[WARN] set_leverage failed:", e)
        try:
            market_id = ex.market(symbol)["id"]
            fn = getattr(ex, "fapiPrivatePostLeverage", None) or getattr(ex, "fapiPrivate_post_leverage", None)
            if fn:
                fn({"symbol": market_id, "leverage": leverage})
                print(f"[OK] fallback leverage -> {leverage}x for {symbol}")
        except Exception as e2:  # noqa: BLE001
            print("[FAIL] fallback leverage failed:", e2)


def safe_set_position_mode_hedge(ex, enable: bool = True) -> None:
    # True → Hedge Mode (LONG/SHORT 동시)
    try:
        ex.set_position_mode(enable)
        print(f"[OK] position mode → hedge={enable}")
    except Exception as e:  # noqa: BLE001
        print("[WARN] set_position_mode failed:", e)
        try:
            fn = getattr(ex, "fapiPrivatePostPositionsideDual", None) or getattr(ex, "fapiPrivate_post_positionside_dual", None)
            if fn:
                fn({"dualSidePosition": "true" if enable else "false"})
                print(f"[OK] fallback positionside-dual → {enable}")
        except Exception as e2:  # noqa: BLE001
            print("[FAIL] fallback positionside-dual failed:", e2)


def get_filters(ex, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    m = ex.market(symbol)
    info = m.get("info", {})
    filters = info.get("filters", [])
    tickSize = stepSize = minNotional = None
    minQty = None
    for f in filters:
        if f.get("filterType") == "PRICE_FILTER":
            tickSize = float(f.get("tickSize", "0.0"))
        elif f.get("filterType") == "LOT_SIZE":
            stepSize = float(f.get("stepSize", "0.0"))
            minQty = float(f.get("minQty", "0.0"))
        elif f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
            minNotional = float(f.get("notional", f.get("minNotional", "0.0")))
    return tickSize, stepSize, minQty, minNotional


def get_mark_price(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    info = t.get("info", {})
    mp = info.get("markPrice")
    return float(mp) if mp is not None else float(t.get("last"))


def get_equity_USDC(ex) -> float:
    bal = ex.fetch_balance()
    info = bal.get("info", {})
    v = bal.get("USDC", {}).get("total")
    if v is not None:
        return float(v)
    for k in ["totalWalletBalance", "totalMarginBalance", "availableBalance"]:
        if k in info:
            try:
                return float(info[k])
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError("Equity(USDC) 조회 실패")
