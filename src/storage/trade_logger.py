from pathlib import Path
import csv
from typing import Optional


class TradeCSV:
    def __init__(self, path: str = "logs/trades_live.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","event","side","entry_px","exit_px","qty","pnl_pct","equity","reason"])

    def write(
        self,
        ts: str,
        event: str,
        side: Optional[str],
        entry_px: Optional[float],
        exit_px: Optional[float],
        qty: Optional[float],
        pnl_pct: Optional[float],
        equity: Optional[float],
        reason: Optional[str],
    ) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ts, event, side or "",
                f"{entry_px:.2f}" if entry_px is not None else "",
                f"{exit_px:.2f}" if exit_px is not None else "",
                f"{qty:.6f}" if qty is not None else "",
                f"{pnl_pct:.6f}" if pnl_pct is not None else "",
                f"{equity:.2f}" if equity is not None else "",
                reason or ""
            ])
