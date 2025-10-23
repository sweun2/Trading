from config.settings import Config
from utils.logger import setup_logger
from exchange.binance_utils import (
    make_exchange,
    safe_set_position_mode_hedge,
    safe_set_margin_and_leverage,
)
from storage.trade_logger import TradeCSV
from strategy.trader import Trader


def main() -> None:
    logger = setup_logger("live", logfile="logs/live.log")
    cfg = Config()

    ex = make_exchange(cfg)
    ex.load_markets()

    # Hedge mode + Cross + leverage
    safe_set_position_mode_hedge(ex, enable=True)
    safe_set_margin_and_leverage(ex, cfg.SYMBOL, leverage=cfg.LEVERAGE_SET, margin_mode="cross")

    trader = Trader(ex, cfg, logger=logger, tlog=TradeCSV("logs/trades_live.csv"))
    trader.run()


if __name__ == "__main__":
    main()
