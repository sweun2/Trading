from types import SimpleNamespace
from storage.trade_logger import TradeCSV
from strategy.trader import Trader


class FakeEx:
    def market(self, symbol):
        return {
            "id": "BTCUSDC",
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            },
        }


def test_trader_init_minimal(tmp_path):
    ex = FakeEx()
    cfg = SimpleNamespace(SYMBOL="BTC/USDC")
    log_file = tmp_path / "test_trades.csv"
    t = Trader(ex, cfg, tlog=TradeCSV(str(log_file)))
    assert t.stepSize == 0.001
    assert t.minQty == 0.001
