import time
from datetime import datetime, timedelta, timezone


def now_kst() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=9)


def align_to_next_close(tf: str = "5m") -> None:
    unit = int(tf.replace("m",""))
    now = datetime.now(timezone.utc)
    minute = (now.minute // unit + 1) * unit
    next_dt = now.replace(second=1, microsecond=0)
    next_dt = next_dt.replace(minute=0) + timedelta(hours=1) if minute >= 60 else next_dt.replace(minute=minute)
    sleep_s = (next_dt - now).total_seconds()
    if sleep_s > 0:
        time.sleep(sleep_s)
