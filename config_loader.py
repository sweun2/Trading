from dataclasses import dataclass
import configparser
from pathlib import Path

_cfg = configparser.ConfigParser()
for candidate in ["./config.ini", "./config.skeleton.ini", str(Path(__file__).resolve().parent / "config.ini")]:
    _cfg.read(candidate)

def _get(section, key, fallback=None):
    if _cfg.has_option(section, key):
        return _cfg.get(section, key)
    return fallback
def _getint(section, key, fallback=None):
    try:
        return _cfg.getint(section, key, fallback=fallback)
    except Exception:
        v = _get(section, key, None)
        return int(v) if v is not None and str(v).strip() != '' else fallback
def _getfloat(section, key, fallback=None):
    try:
        return _cfg.getfloat(section, key, fallback=fallback)
    except Exception:
        v = _get(section, key, None)
        return float(v) if v is not None and str(v).strip() != '' else fallback
def _getbool(section, key, fallback=None):
    try:
        return _cfg.getboolean(section, key, fallback=fallback)
    except Exception:
        v = _get(section, key, None)
        if v is None: return fallback
        s = str(v).strip().lower()
        if s in ('1','true','yes','y','on'): return True
        if s in ('0','false','no','n','off'): return False
        return fallback

@dataclass
class _CFG:
    # API
    API_KEY: str = _get('API','API_KEY','')
    API_SECRET: str = _get('API','API_SECRET','')
    TESTNET: bool = _getbool('API','TESTNET', False)

    # Market/Data
    SYMBOL: str = _get('MARKET','SYMBOL','BTC/USDC')
    TIMEFRAME: str = _get('MARKET','TIMEFRAME','5m')
    LOOKBACK: int = _getint('MARKET','LOOKBACK',240)

    # Indicators/Strategy params
    std_win: int = _getint('INDICATORS','std_win',10)
    band_k_long: float = _getfloat('STRATEGY_LONG','band_k',1.90)
    stop_k_long: float = _getfloat('STRATEGY_LONG','stop_k',3.0)
    std_min_long: float = _getfloat('STRATEGY_LONG','std_min',0.0015)
    momentum_threshold_long: float = _getfloat('STRATEGY_LONG','momentum_threshold',-0.003)
    time_stop_min_long: int = _getint('STRATEGY_LONG','time_stop_min',30)

    band_k_short: float = _getfloat('STRATEGY_SHORT','band_k',2.10)
    stop_k_short: float = _getfloat('STRATEGY_SHORT','stop_k',3.5)
    std_min_short: float = _getfloat('STRATEGY_SHORT','std_min',0.0015)
    momentum_threshold_short: float = _getfloat('STRATEGY_SHORT','momentum_threshold',0.003)
    time_stop_min_short: int = _getint('STRATEGY_SHORT','time_stop_min',30)
    ALLOW_SHORTS: bool = _getbool('STRATEGY_SHORT','allow_shorts', True)

    # Risk/Leverage
    LEVERAGE_SET: int = _getint('RISK','leverage_set',20)
    TARGET_LEVERAGE: float = _getfloat('RISK','target_leverage',6.0)
    MAX_DAILY_DD_PCT: float = _getfloat('RISK','max_daily_dd_pct',0.05)
    MAX_CONSEC_SL: int = _getint('RISK','max_consec_sl',3)

    # Execution
    ENTRY_MODE: str = _get('EXECUTION','entry_mode','market')
    TP_REPRICE_EPS: float = _getfloat('EXECUTION','tp_reprice_eps',0.0005)
    SL_REPRICE_EPS: float = _getfloat('EXECUTION','sl_reprice_eps',0.0005)
    TP_LIMIT_OFFSET_PCT: float = _getfloat('EXECUTION','tp_limit_offset_pct',0.0001)
    QTY_EPS: float = _getfloat('EXECUTION','qty_eps',1e-12)
    PRICE_EPS: float = _getfloat('EXECUTION','price_eps',1e-8)
    SLEEP_AFTER_ENTRY_CHECK_MS: int = _getint('EXECUTION','sleep_after_entry_check_ms',250)
    SLEEP_AFTER_CANCEL_MS: int = _getint('EXECUTION','sleep_after_cancel_ms',250)
    MAIN_LOOP_SLEEP_MS: int = _getint('EXECUTION','main_loop_sleep_ms',500)
    REDUCE_ONLY: bool = _getbool('EXECUTION','reduce_only', True)

    # Session filter
    KST_FILTER_START: int = _getint('SESSION_FILTER','kst_filter_start',0)
    KST_FILTER_END: int = _getint('SESSION_FILTER','kst_filter_end',6)

    # Exchange modes
    POSITION_MODE: str = _get('EXCHANGE','position_mode','hedge')  # hedge|oneway
    MARGIN_MODE: str = _get('EXCHANGE','margin_mode','cross')      # cross|isolated

    # Logging
    LOGFILE: str = _get('LOG','logfile','logs/live.log')
    TRADES_CSV: str = _get('LOG','trades_csv','logs/trades_live.csv')
    LOGLEVEL: str = _get('LOG','loglevel','INFO')

    # Guards (critical safety)
    REQUIRE_STATE_RECOVERY: bool = _getbool('GUARDS','REQUIRE_STATE_RECOVERY', True)
    STARTUP_CANCEL_OPEN_ORDERS: bool = _getbool('GUARDS','STARTUP_CANCEL_OPEN_ORDERS', True)
    TRIGGER_PRICE_SOURCE: str = _get('GUARDS','TRIGGER_PRICE_SOURCE','mark')
    PRICE_PROTECT_MAX_DEVIATION_PCT: float = _getfloat('GUARDS','PRICE_PROTECT_MAX_DEVIATION_PCT',0.002)
    LIMIT_PRICE_BAND_PCT: float = _getfloat('GUARDS','LIMIT_PRICE_BAND_PCT',0.001)
    MARKET_METADATA_REFRESH_SEC: int = _getint('GUARDS','MARKET_METADATA_REFRESH_SEC',300)
    MIN_BBO_SIZE_NOTIONAL: float = _getfloat('GUARDS','MIN_BBO_SIZE_NOTIONAL',1000.0)
    MAX_SPREAD_PCT: float = _getfloat('GUARDS','MAX_SPREAD_PCT',0.001)
    RESERVED_MARGIN_PCT: float = _getfloat('GUARDS','RESERVED_MARGIN_PCT',0.2)
    HALT_ON_METADATA_CHANGE: bool = _getbool('GUARDS','HALT_ON_METADATA_CHANGE', True)
    HALT_ON_BRACKET_CHANGE: bool = _getbool('GUARDS','HALT_ON_BRACKET_CHANGE', True)
    API_RETRY_MAX: int = _getint('GUARDS','API_RETRY_MAX',5)
    API_BACKOFF_BASE_MS: int = _getint('GUARDS','API_BACKOFF_BASE_MS',200)
    CIRCUIT_BREAKER_COOLDOWN_SEC: int = _getint('GUARDS','CIRCUIT_BREAKER_COOLDOWN_SEC',60)
    NTP_MAX_DRIFT_MS: int = _getint('GUARDS','NTP_MAX_DRIFT_MS',150)
    DAILY_LOCK_HOURS_AFTER_DD: int = _getint('GUARDS','DAILY_LOCK_HOURS_AFTER_DD',8)
    COOLDOWN_MIN_AFTER_K_LOSSES: int = _getint('GUARDS','COOLDOWN_MIN_AFTER_K_LOSSES',30)
    BLOCK_AROUND_FUNDING_MIN: int = _getint('GUARDS','BLOCK_AROUND_FUNDING_MIN',10)
    FUNDING_BLOCK_THRESHOLD: float = _getfloat('GUARDS','FUNDING_BLOCK_THRESHOLD',0.0005)

CFG = _CFG()
