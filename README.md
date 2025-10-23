# Trading Bot Skeleton

Refactor of your single-file `live.py` into a modular package without changing internal logic.
Includes type hints and minimal test scaffolding.

## Run

```bash
pip install -r requirements.txt
# Put your secrets in config.ini (see config.sample.ini)
python src/main.py
```

## Tests

```bash
pytest -q
```

## Layout

```
src/
  main.py
  config/settings.py
  utils/{logger.py,time_utils.py,math_utils.py}
  exchange/binance_utils.py
  strategy/{indicators.py,trader.py}
  storage/trade_logger.py
tests/
  conftest.py
  test_utils.py
  test_indicators.py
  test_trader.py
```
