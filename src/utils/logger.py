from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "live", logfile: str = "logs/live.log", level: int = logging.INFO) -> logging.Logger:
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
