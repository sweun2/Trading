from dataclasses import dataclass
import configparser


config = configparser.ConfigParser()
config.read('./config.ini')


@dataclass
class Config:
    API_KEY: str = config['API']['API_KEY'].strip()
    API_SECRET: str = config["API"]["API_SECRET"].strip()
    TESTNET: bool = False

    SYMBOL: str = "BTC/USDC"   # 실계정 USDT면 "BTC/USDT"
    TIMEFRAME: str = "5m"
    LOOKBACK: int = 240

    # ===== 전략 파라미터 (비대칭 고정) =====
    std_win: int = 10
    # 롱
    band_k_long: float = 1.90
    stop_k_long: float = 3.0
    std_min_long: float = 0.0015
    momentum_threshold_long: float = -0.003
    time_stop_min_long: int = 30
    # 숏
    band_k_short: float = 2.10
    stop_k_short: float = 3.5
    std_min_short: float = 0.0015
    momentum_threshold_short: float = 0.003
    time_stop_min_short: int = 30
    ALLOW_SHORTS: bool = True

    # ===== 레버리지/리스크 =====
    LEVERAGE_SET: int = 20           # 거래소 설정 레버리지
    TARGET_LEVERAGE: float = 6.0      # 실제 오픈 노출(진입시 Equity * 6x)
    MAX_DAILY_DD_PCT: float = 0.05
    MAX_CONSEC_SL: int = 3
    ENTRY_MODE: str = "market"        # "market" | "limit-postonly"

    # 재배치 민감도
    TP_REPRICE_EPS: float = 0.0005
    SL_REPRICE_EPS: float = 0.0005

    # 세션 필터: KST 0~6시 제외
    KST_FILTER_START: int = 0
    KST_FILTER_END: int = 6

    # TP 리밋 가격 오프셋 (0.05% = 0.0001)
    TP_LIMIT_OFFSET_PCT: float = 0.0001
