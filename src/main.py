from config_loader import CFG
from src.utils.logger import setup_logger
from src.exchange.binance_utils import (
    make_exchange,
    safe_set_position_mode_hedge,
    safe_set_margin_and_leverage,
)
from src.storage.trade_logger import TradeCSV
from src.strategy.trader import Trader
from src.strategy.guards import startup_preflight, locked_now
import sys, os, time, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main() -> None:
    logger = setup_logger("live", logfile="logs/live.log")
    cfg = CFG

    ex = make_exchange(cfg)
    try:
        ex.options = getattr(ex, 'options', {}) or {}
        ex.options['adjustForTimeDifference'] = True
        ex.options['recvWindow'] = 60000  # increase tolerance
        try:
            ex.load_time_difference()
            logging.getLogger(__name__).info(
                f"[TIME] exchange time offset(ms) = {getattr(ex, 'timeDifference', 0)}"
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"[TIME] load_time_difference failed: {e}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[TIME] options setup failed: {e}")

    ex.load_markets()

    # Startup safety
    startup_preflight(ex, cfg, logger)

    # Hedge/Cross/Leverage from config
    safe_set_position_mode_hedge(ex, enable=(cfg.POSITION_MODE == 'hedge'))
    safe_set_margin_and_leverage(ex, cfg.SYMBOL, leverage=cfg.LEVERAGE_SET, margin_mode=cfg.MARGIN_MODE)

    if locked_now():
        logger.warning("[LOCK] Daily DD lock active. Exiting.")
        return
    trader = Trader(ex, cfg, logger=logger, tlog=TradeCSV("logs/trades_live.csv"))
    trader.run()


if __name__ == "__main__":
    main()
