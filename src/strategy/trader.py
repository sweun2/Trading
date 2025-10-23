from __future__ import annotations

import time
import logging
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.math_utils import round_step
from utils.time_utils import now_kst, align_to_next_close
from exchange.binance_utils import get_filters, get_equity_USDC, get_mark_price
from strategy.indicators import fetch_latest_ohlcv, compute_indicators
from storage.trade_logger import TradeCSV


class Trader:
    def __init__(self, ex: Any, cfg: Any, logger: Optional[logging.Logger] = None, tlog: Optional[TradeCSV] = None) -> None:
        self.ex = ex
        self.cfg = cfg

        # 포지션 두 슬롯
        self.long_pos: Optional[Dict[str, Any]] = None    # {"entry_px":..., "qty":..., "entry_time":...}
        self.short_pos: Optional[Dict[str, Any]] = None

        # 엔트리(포스트온리) 추적
        self.entry_id_long: Optional[str] = None
        self.entry_id_short: Optional[str] = None

        # 각 사이드별 TP/SL ID & 가격
        self.tp_id_long: Optional[str] = None;  self.sl_id_long: Optional[str] = None
        self.tp_id_short: Optional[str] = None; self.sl_id_short: Optional[str] = None
        self.last_tp_px_long: Optional[float] = None;  self.last_sl_px_long: Optional[float] = None
        self.last_tp_px_short: Optional[float] = None; self.last_sl_px_short: Optional[float] = None

        # 마지막 봉 타임스탬프(KST)
        self.last_candle_ts = None

        # 세션 상태
        self.day_start_eq: Optional[float] = None
        self.consec_sl: int = 0
        self.halt: bool = False  # 연속 SL 가드 발동시 신규 진입 차단

        self.tickSize, self.stepSize, self.minQty, self.minNotional = get_filters(ex, cfg.SYMBOL)

        self.logger = logger or logging.getLogger("live")
        self.tlog = tlog or TradeCSV("logs/trades_live.csv")

    # ========== 공통 보조 ==========
    def cancel_if_exists(self, oid: Optional[str]) -> None:
        if not oid:
            return
        try:
            self.ex.cancel_order(oid, self.cfg.SYMBOL)
        except Exception:
            pass

    def cancel_side_tp_sl(self, side: str) -> None:
        if side == "long":
            self.cancel_if_exists(self.tp_id_long); self.tp_id_long = None
            self.cancel_if_exists(self.sl_id_long); self.sl_id_long = None
            self.last_tp_px_long = None; self.last_sl_px_long = None
        else:
            self.cancel_if_exists(self.tp_id_short); self.tp_id_short = None
            self.cancel_if_exists(self.sl_id_short); self.sl_id_short = None
            self.last_tp_px_short = None; self.last_sl_px_short = None

    def cancel_entry_order(self, side: str) -> None:
        if side == "long":
            if self.entry_id_long:
                self.cancel_if_exists(self.entry_id_long)
                self.entry_id_long = None
        else:
            if self.entry_id_short:
                self.cancel_if_exists(self.entry_id_short)
                self.entry_id_short = None

    # ========== 주문 ==========
    def _position_side_str(self, side: str) -> str:
        return "LONG" if side == "long" else "SHORT"

    def place_entry(self, side: str, qty: float, px: Optional[float] = None) -> Dict[str, Any]:
        ord_side = "buy" if side == "long" else "sell"
        # 엔트리에는 reduceOnly 금지 (Hedge)
        params = {"positionSide": self._position_side_str(side)}
        if self.cfg.ENTRY_MODE == "limit-postonly":
            params.update({"postOnly": True})  # ccxt가 GTX로 매핑
            assert px is not None, "limit-postonly 모드엔 가격 필요"
            return self.ex.create_order(self.cfg.SYMBOL, "limit", ord_side, qty, px, params)
        else:
            return self.ex.create_order(self.cfg.SYMBOL, "market", ord_side, qty, None, params)

    def place_tp(self, side: str, stop_px: float, qty: float) -> Dict[str, Any]:
        """
        TAKE_PROFIT (limit) + stopPrice
        - 트리거: stopPrice(= vwap 등)
        - 체결용 limit: stop 대비 0.05% 유리한 방향 (Config.TP_LIMIT_OFFSET_PCT)
          * LONG(SELL): stop * (1 + off)
          * SHORT(BUY): stop * (1 - off)
        """
        ord_side = "sell" if side == "long" else "buy"

        # 트리거(stop) 라운딩 (BUY일 때 floor, SELL일 때 ceil)
        stop_px = round_step(stop_px, self.tickSize, "floor" if ord_side == "buy" else "ceil")

        off = float(self.cfg.TP_LIMIT_OFFSET_PCT)
        if side == "long":  # SELL
            limit_px = stop_px * (1 + off)
            limit_px = round_step(limit_px, self.tickSize, "ceil")
        else:               # SHORT -> BUY
            limit_px = stop_px * (1 - off)
            limit_px = round_step(limit_px, self.tickSize, "floor")

        params = {
            "stopPrice": stop_px,
            "workingType": "MARK_PRICE",
            "positionSide": self._position_side_str(side),
            # Hedge 모드: reduceOnly/postOnly 미전송
            # timeInForce 기본 GTC
        }
        return self.ex.create_order(self.cfg.SYMBOL, "take_profit", ord_side, qty, limit_px, params)

    def place_sl(self, side: str, stop_px: float, qty: float) -> Dict[str, Any]:
        # STOP_MARKET (시장가 청산), Hedge 모드 → reduceOnly 미전송
        ord_side = "sell" if side == "long" else "buy"
        stop_px = round_step(stop_px, self.tickSize, "floor" if ord_side=="buy" else "ceil")
        params = {
            "stopPrice": stop_px,
            "workingType": "MARK_PRICE",
            "positionSide": self._position_side_str(side),
        }
        return self.ex.create_order(self.cfg.SYMBOL, "stop_market", ord_side, qty, None, params)

    # ========== 사이징 ==========
    def compute_qty(self, entry_px: float, equity_USDC: float) -> float:
        notional = equity_USDC * float(self.cfg.TARGET_LEVERAGE)
        qty = notional / max(1e-12, float(entry_px))
        if self.stepSize:
            qty = round_step(qty, self.stepSize, "floor")
        if self.minQty and qty < self.minQty:
            qty = self.minQty
        if self.minNotional and qty * entry_px < self.minNotional:
            qty = round_step(self.minNotional / entry_px, self.stepSize or 1e-8, "ceil")
        return max(qty, 0.0)

    # ========== 시그널 ==========
    def eligible_time_kst(self, kst_dt) -> bool:
        h = kst_dt.hour
        return not (self.cfg.KST_FILTER_START <= h < self.cfg.KST_FILTER_END)

    def evaluate_signals(self, ind: pd.DataFrame) -> List[Dict[str, Any]]:
        if self.halt:
            return []

        if ind.empty:
            return []

        close_kst = ind.index[-1]
        if self.last_candle_ts and close_kst <= self.last_candle_ts:
            return []
        self.last_candle_ts = close_kst

        if not self.eligible_time_kst(close_kst):
            return []

        row = ind.iloc[-1]
        price = float(row["close"])
        vwap  = float(row["vwap"])
        std   = float(row["std"])
        upper_short = float(row["upper_short"])
        lower_long  = float(row["lower_long"])
        mom10 = float(row["mom10"])

        sigs: List[Dict[str, Any]] = []
        # 롱
        if std >= self.cfg.std_min_long and (np.isnan(mom10) or mom10 >= self.cfg.momentum_threshold_long):
            if price < lower_long and self.long_pos is None:
                sigs.append({"slot": "long", "price": price, "vwap": vwap, "std": std})
        # 숏
        if self.cfg.ALLOW_SHORTS and std >= self.cfg.std_min_short and (np.isnan(mom10) or mom10 <= self.cfg.momentum_threshold_short):
            if price > upper_short and self.short_pos is None:
                sigs.append({"slot": "short", "price": price, "vwap": vwap, "std": std})
        return sigs

    # ========== 재배치 ==========
    def maybe_reprice_for_side(self, side: str, vwap: float, new_sl: float) -> bool:
        def need_reprice(old: Optional[float], new: float, eps: float) -> bool:
            if old is None:
                return True
            return abs(new - old) / max(1e-12, old) >= eps

        if side == "long":
            assert self.long_pos is not None
            qty = self.long_pos["qty"]; changed = False
            # TP (vwap이 바뀌면 stopPrice/limit 같이 갱신)
            if need_reprice(self.last_tp_px_long, vwap, self.cfg.TP_REPRICE_EPS):
                self.cancel_if_exists(self.tp_id_long); self.tp_id_long = None
                try:
                    tp = self.place_tp("long", vwap, qty)
                    self.tp_id_long = tp["id"]; self.last_tp_px_long = vwap; changed = True
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[LONG] TP 재설정 실패: {e}")
            # SL (넓히지 않음)
            target_sl = new_sl if (self.last_sl_px_long is None or new_sl > self.last_sl_px_long) else self.last_sl_px_long
            if need_reprice(self.last_sl_px_long, target_sl, self.cfg.SL_REPRICE_EPS):
                self.cancel_if_exists(self.sl_id_long); self.sl_id_long = None
                try:
                    sl = self.place_sl("long", target_sl, qty)
                    self.sl_id_long = sl["id"]; self.last_sl_px_long = target_sl; changed = True
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[LONG] SL 재설정 실패: {e}")
            return changed

        else:
            assert self.short_pos is not None
            qty = self.short_pos["qty"]; changed = False
            # TP
            if need_reprice(self.last_tp_px_short, vwap, self.cfg.TP_REPRICE_EPS):
                self.cancel_if_exists(self.tp_id_short); self.tp_id_short = None
                try:
                    tp = self.place_tp("short", vwap, qty)
                    self.tp_id_short = tp["id"]; self.last_tp_px_short = vwap; changed = True
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SHORT] TP 재설정 실패: {e}")
            # SL (넓히지 않음)
            target_sl = new_sl if (self.last_sl_px_short is None or new_sl < self.last_sl_px_short) else self.last_sl_px_short
            if need_reprice(self.last_sl_px_short, target_sl, self.cfg.SL_REPRICE_EPS):
                self.cancel_if_exists(self.sl_id_short); self.sl_id_short = None
                try:
                    sl = self.place_sl("short", target_sl, qty)
                    self.sl_id_short = sl["id"]; self.last_sl_px_short = target_sl; changed = True
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SHORT] SL 재설정 실패: {e}")
            return changed

    # ========== 체결 감지 ==========
    def _detect_and_finalize_exit(self, side: str, ind_row: pd.Series) -> None:
        try:
            oo = self.ex.fetch_open_orders(self.cfg.SYMBOL)
            open_ids = {o["id"] for o in oo}

            if side == "long":
                tp_missing = (self.tp_id_long is not None) and (self.tp_id_long not in open_ids)
                sl_missing = (self.sl_id_long is not None) and (self.sl_id_long not in open_ids)
                if not (tp_missing or sl_missing):
                    return
            else:
                tp_missing = (self.tp_id_short is not None) and (self.tp_id_short not in open_ids)
                sl_missing = (self.sl_id_short is not None) and (self.sl_id_short not in open_ids)
                if not (tp_missing or sl_missing):
                    return

            size_long = size_short = 0.0
            mid = self.ex.market(self.cfg.SYMBOL)["id"]
            try:
                poss = self.ex.fetch_positions([self.cfg.SYMBOL])
                for p in poss:
                    if p.get("symbol") != mid:
                        continue
                    ps = (p.get("positionSide") or p.get("side") or "").upper()
                    sz = abs(float(p.get("contracts", 0) or 0.0))
                    if ps == "LONG":
                        size_long = sz
                    elif ps == "SHORT":
                        size_short = sz
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"포지션 조회 실패(무시): {e}")

            if side == "long" and size_long < 1e-8 and self.long_pos is not None:
                entry_px = self.long_pos["entry_px"]
                exit_px = float(ind_row["close"])
                pnl = (exit_px - entry_px) / entry_px
                reason = "SL" if sl_missing and not tp_missing else "TP"
                cur_eq = get_equity_USDC(self.ex)
                self.tlog.write(now_kst().isoformat(), f"EXIT_{reason}", "long",
                                entry_px, exit_px, self.long_pos["qty"], pnl, cur_eq, "filled")
                self.logger.info(f"[EXIT LONG] {reason} entry≈{entry_px} exit≈{exit_px:.2f} pnl%={pnl*100:.4f}")
                if reason == "SL":
                    self.consec_sl += 1
                else:
                    self.consec_sl = 0
                if self.consec_sl >= self.cfg.MAX_CONSEC_SL:
                    self._trigger_consec_halt()
                self.cancel_side_tp_sl("long")
                self.long_pos = None

            if side == "short" and size_short < 1e-8 and self.short_pos is not None:
                entry_px = self.short_pos["entry_px"]
                exit_px = float(ind_row["close"])
                pnl = (entry_px - exit_px) / entry_px
                reason = "SL" if sl_missing and not tp_missing else "TP"
                cur_eq = get_equity_USDC(self.ex)
                self.tlog.write(now_kst().isoformat(), f"EXIT_{reason}", "short",
                                entry_px, exit_px, self.short_pos["qty"], pnl, cur_eq, "filled")
                self.logger.info(f"[EXIT SHORT] {reason} entry≈{entry_px} exit≈{exit_px:.2f} pnl%={pnl*100:.4f}")
                if reason == "SL":
                    self.consec_sl += 1
                else:
                    self.consec_sl = 0
                if self.consec_sl >= self.cfg.MAX_CONSEC_SL:
                    self._trigger_consec_halt()
                self.cancel_side_tp_sl("short")
                self.short_pos = None

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"체결 확인 실패: {e}")

    # ========== 동기화 ==========
    def sync_positions(self, ind_row: pd.Series) -> None:
        """거래소에 포지션이 존재하지만 내부 상태가 비어 있을 때 상태를 세팅하고 보호 주문을 부착한다."""
        try:
            mid = self.ex.market(self.cfg.SYMBOL)["id"]
            poss = self.ex.fetch_positions([self.cfg.SYMBOL])
            size_long = size_short = 0.0
            entry_long = entry_short = None
            for p in poss:
                if p.get("symbol") != mid:
                    continue
                ps = (p.get("positionSide") or p.get("side") or "").upper()
                sz = abs(float(p.get("contracts", 0) or 0.0))
                ep = float(p.get("entryPrice") or 0.0)
                if ps == "LONG":
                    size_long = sz; entry_long = ep
                elif ps == "SHORT":
                    size_short = sz; entry_short = ep

            # LONG 동기화
            if size_long > 1e-8 and self.long_pos is None:
                self.long_pos = {"entry_px": entry_long or float(ind_row["close"]), "qty": size_long, "entry_time": now_kst()}
                self.entry_id_long = None
                vwap = float(ind_row["vwap"]); std = float(ind_row["std"])
                sl_px = max(1e-8, vwap - self.cfg.stop_k_long * std)
                try:
                    tp = self.place_tp("long", vwap, size_long)
                    self.tp_id_long = tp["id"]; self.last_tp_px_long = vwap
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SYNC LONG] TP 발주 실패: {e}")
                try:
                    sl = self.place_sl("long", sl_px, size_long)
                    self.sl_id_long = sl["id"]; self.last_sl_px_long = sl_px
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SYNC LONG] SL 발주 실패: {e}")
                self.logger.info("[SYNC LONG] 거래소 포지션 감지 → 상태 초기화 및 보호주문 부착")

            # SHORT 동기화
            if size_short > 1e-8 and self.short_pos is None:
                self.short_pos = {"entry_px": entry_short or float(ind_row["close"]), "qty": size_short, "entry_time": now_kst()}
                self.entry_id_short = None
                vwap = float(ind_row["vwap"]); std = float(ind_row["std"])
                sl_px = max(1e-8, vwap + self.cfg.stop_k_short * std)
                try:
                    tp = self.place_tp("short", vwap, size_short)
                    self.tp_id_short = tp["id"]; self.last_tp_px_short = vwap
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SYNC SHORT] TP 발주 실패: {e}")
                try:
                    sl = self.place_sl("short", sl_px, size_short)
                    self.sl_id_short = sl["id"]; self.last_sl_px_short = sl_px
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"[SYNC SHORT] SL 발주 실패: {e}")
                self.logger.info("[SYNC SHORT] 거래소 포지션 감지 → 상태 초기화 및 보호주문 부착")

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"[SYNC] 실패: {e}")

    # ========== 하트비트 ==========
    def heartbeat(self, ind_row: pd.Series, cur_eq: float, day_dd: float) -> None:
        try:
            lp = f"L:{(self.long_pos or {}).get('qty')}" if self.long_pos else "L:0"
            sp = f"S:{(self.short_pos or {}).get('qty')}" if self.short_pos else "S:0"
            line = (
                f"HB | {lp} {sp} close={float(ind_row['close']):.2f} "
                f"vwap={float(ind_row['vwap']):.2f} std={float(ind_row['std']):.4f} "
                f"bands=[L{float(ind_row['lower_long']):.2f} / U{float(ind_row['upper_short']):.2f}] "
                f"mom10={float(ind_row['mom10']):.4f} "
                f"tpL={self.last_tp_px_long} slL={self.last_sl_px_long} "
                f"tpS={self.last_tp_px_short} slS={self.last_sl_px_short} "
                f"eq={cur_eq:.2f} dd={day_dd:.2%} halt={self.halt}"
            )
            self.logger.info(line)
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"heartbeat encode error: {e}")

    # ========== 컨트롤 ==========
    def _trigger_consec_halt(self) -> None:
        msg = f"[SESSION HALT] consecutive SL = {self.consec_sl} (>= {self.cfg.MAX_CONSEC_SL})"
        self.halt = True
        self.logger.warning(msg)
        self.tlog.write(now_kst().isoformat(), "STOP_CONSEC_SL", None, None, None, None, None, get_equity_USDC(self.ex), msg)

    def flatten_all(self) -> None:
        """모든 사이드 포지션 강제 청산 및 보호주문/엔트리오더 취소"""
        for side in ("long","short"):
            pos = self.long_pos if side=="long" else self.short_pos
            try:
                self.cancel_entry_order(side)
                self.cancel_side_tp_sl(side)
                if pos:
                    ord_side = "sell" if side=="long" else "buy"
                    # Hedge 모드: reduceOnly 없이 positionSide로 정확히 닫음
                    self.ex.create_order(self.cfg.SYMBOL, "market", ord_side, pos["qty"], None,
                                         {"positionSide": self._position_side_str(side)})
                    self.logger.warning(f"[FLATTEN {side.upper()}] reduce market sent")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"[FLATTEN {side}] 실패: {e}")
        self.long_pos = None
        self.short_pos = None
        self.tp_id_long = self.sl_id_long = None
        self.tp_id_short = self.sl_id_short = None
        self.entry_id_long = self.entry_id_short = None
        self.last_tp_px_long = self.last_sl_px_long = None
        self.last_tp_px_short = self.last_sl_px_short = None

    # ========== 메인 루프 ==========
    def run(self) -> None:
        self.logger.info(f"▶ starting trader... testnet={self.cfg.TESTNET}")
        self.day_start_eq = get_equity_USDC(self.ex)
        last_day = now_kst().date()

        while True:
            try:
                align_to_next_close(self.cfg.TIMEFRAME)

                # 날짜 변경 시 리셋
                kst_today = now_kst().date()
                if kst_today != last_day:
                    self.day_start_eq = get_equity_USDC(self.ex)
                    self.consec_sl = 0
                    self.halt = False  # 새로운 날에는 가드 해제
                    last_day = kst_today
                    self.logger.info("[DAY ROLLOVER] reset day_start_eq, consec_sl, halt")

                # DD 가드
                cur_eq = get_equity_USDC(self.ex)
                day_dd = (cur_eq - self.day_start_eq) / max(1e-8, self.day_start_eq)
                if day_dd <= -self.cfg.MAX_DAILY_DD_PCT:
                    msg = f"[KILL] Daily DD {day_dd:.2%} <= -{self.cfg.MAX_DAILY_DD_PCT:.2%}. Flatten and stop."
                    self.logger.warning(msg)
                    self.flatten_all()
                    self.tlog.write(now_kst().isoformat(), "KILL_DD", None, None, None, None, None, cur_eq, msg)
                    break

                # 데이터 & 인디케이터
                df = fetch_latest_ohlcv(self.ex, self.cfg.SYMBOL, self.cfg.TIMEFRAME, self.cfg.LOOKBACK)
                ind = compute_indicators(df, self.cfg)
                if ind.empty:
                    self.logger.info("indicators empty, skip tick")
                    continue

                # 거래소 실제 포지션과 내부 상태 동기화 (체결 누락 대비)
                self.sync_positions(ind.iloc[-1])

                # 진입 시그널(양쪽 가능)
                signals = self.evaluate_signals(ind)
                for sig in signals:
                    slot = sig["slot"]  # "long" | "short"
                    # post-only 모드에서 동일 사이드의 기존 엔트리 오더가 열려 있으면 중복 엔트리 금지
                    if self.cfg.ENTRY_MODE == "limit-postonly":
                        open_ids = {o["id"] for o in self.ex.fetch_open_orders(self.cfg.SYMBOL)}
                        if slot == "long" and self.entry_id_long and self.entry_id_long in open_ids:
                            self.logger.info("[ENTRY LONG] pending post-only exists, skip")
                            continue
                        if slot == "short" and self.entry_id_short and self.entry_id_short in open_ids:
                            self.logger.info("[ENTRY SHORT] pending post-only exists, skip")
                            continue

                    mark_px = get_mark_price(self.ex, self.cfg.SYMBOL)
                    vwap = sig["vwap"]; std = sig["std"]

                    # SL px (사이드별)
                    sl_px = (vwap - self.cfg.stop_k_long * std) if slot == "long" else (vwap + self.cfg.stop_k_short * std)
                    sl_px = max(1e-8, sl_px)

                    # 사이즈: Equity * TARGET_LEVERAGE
                    qty = self.compute_qty(mark_px, cur_eq)
                    if qty <= 0:
                        self.logger.info(f"[ENTRY-{slot}] size 0, skip")
                        continue

                    if self.cfg.ENTRY_MODE == "limit-postonly":
                        ob = self.ex.fetch_order_book(self.cfg.SYMBOL)
                        if slot == "long":
                            best_bid = ob["bids"][0][0] if ob["bids"] else mark_px
                            entry_px = round_step(best_bid - (self.tickSize or 0.0), self.tickSize or 1e-8, "floor")
                        else:
                            best_ask = ob["asks"][0][0] if ob["asks"] else mark_px
                            entry_px = round_step(best_ask + (self.tickSize or 0.0), self.tickSize or 1e-8, "ceil")
                        self.logger.info(f"[ENTRY {slot.upper()}] post-only px≈{entry_px:.2f} qty={qty}")
                        od = self.place_entry(slot, qty, px=entry_px)
                        if slot == "long":
                            self.entry_id_long = od.get("id")
                        else:
                            self.entry_id_short = od.get("id")

                        # 즉시 체결 확인 → 체결 시에만 상태 세팅 및 보호주문 부착
                        time.sleep(0.25)
                        mid = self.ex.market(self.cfg.SYMBOL)["id"]
                        try:
                            poss = self.ex.fetch_positions([self.cfg.SYMBOL])
                            for p in poss:
                                if p.get("symbol") != mid:
                                    continue
                                ps = (p.get("positionSide") or p.get("side") or "").upper()
                                sz = abs(float(p.get("contracts", 0) or 0.0))
                                if slot == "long" and ps == "LONG" and sz > 1e-12 and self.long_pos is None:
                                    avg_entry = float(p.get("entryPrice") or mark_px)
                                    self.long_pos = {"entry_px": avg_entry, "qty": sz, "entry_time": now_kst()}
                                    self.entry_id_long = None
                                    try:
                                        tp = self.place_tp("long", vwap, sz)
                                        self.tp_id_long = tp["id"]; self.last_tp_px_long = vwap
                                    except Exception as e:  # noqa: BLE001
                                        self.logger.warning(f"[LONG] TP 발주 실패: {e}")
                                    try:
                                        sl = self.place_sl("long", sl_px, sz)
                                        self.sl_id_long = sl["id"]; self.last_sl_px_long = sl_px
                                    except Exception as e:  # noqa: BLE001
                                        self.logger.warning(f"[LONG] SL 발주 실패: {e}")
                                    self.tlog.write(now_kst().isoformat(), "ENTRY", "long", avg_entry, None, sz, None, cur_eq, "")
                                    self.logger.info("[ENTRY LONG] filled immediately")
                                if slot == "short" and ps == "SHORT" and sz > 1e-12 and self.short_pos is None:
                                    avg_entry = float(p.get("entryPrice") or mark_px)
                                    self.short_pos = {"entry_px": avg_entry, "qty": sz, "entry_time": now_kst()}
                                    self.entry_id_short = None
                                    try:
                                        tp = self.place_tp("short", vwap, sz)
                                        self.tp_id_short = tp["id"]; self.last_tp_px_short = vwap
                                    except Exception as e:  # noqa: BLE001
                                        self.logger.warning(f"[SHORT] TP 발주 실패: {e}")
                                    try:
                                        sl = self.place_sl("short", sl_px, sz)
                                        self.sl_id_short = sl["id"]; self.last_sl_px_short = sl_px
                                    except Exception as e:  # noqa: BLE001
                                        self.logger.warning(f"[SHORT] SL 발주 실패: {e}")
                                    self.tlog.write(now_kst().isoformat(), "ENTRY", "short", avg_entry, None, sz, None, cur_eq, "")
                                    self.logger.info("[ENTRY SHORT] filled immediately")
                        except Exception as e:  # noqa: BLE001
                            self.logger.warning(f"평단 조회 실패: {e}")

                    else:
                        # 시장가 즉시 진입
                        self.logger.info(f"[ENTRY {slot.upper()}] market qty={qty}")
                        _ = self.place_entry(slot, qty)
                        time.sleep(0.25)
                        avg_entry = mark_px
                        try:
                            poss = self.ex.fetch_positions([self.cfg.SYMBOL])
                            mid = self.ex.market(self.cfg.SYMBOL)["id"]
                            for p in poss:
                                if p.get("symbol") != mid:
                                    continue
                                ps = (p.get("positionSide") or p.get("side") or "").upper()
                                sz = abs(float(p.get("contracts", 0) or 0.0))
                                if slot == "long" and ps == "LONG" and sz > 1e-12:
                                    avg_entry = float(p.get("entryPrice") or mark_px); break
                                if slot == "short" and ps == "SHORT" and sz > 1e-12:
                                    avg_entry = float(p.get("entryPrice") or mark_px); break
                        except Exception as e:  # noqa: BLE001
                            self.logger.warning(f"평단 조회 실패: {e}")

                        if slot == "long":
                            self.long_pos = {"entry_px": avg_entry, "qty": qty, "entry_time": now_kst()}
                            try:
                                tp = self.place_tp("long", vwap, qty)
                                self.tp_id_long = tp["id"]; self.last_tp_px_long = vwap
                            except Exception as e:  # noqa: BLE001
                                self.logger.warning(f"[LONG] TP 발주 실패: {e}")
                            try:
                                sl = self.place_sl("long", sl_px, qty)
                                self.sl_id_long = sl["id"]; self.last_sl_px_long = sl_px
                            except Exception as e:  # noqa: BLE001
                                self.logger.warning(f"[LONG] SL 발주 실패: {e}")
                            self.tlog.write(now_kst().isoformat(), "ENTRY", "long", avg_entry, None, qty, None, cur_eq, "")
                        else:
                            self.short_pos = {"entry_px": avg_entry, "qty": qty, "entry_time": now_kst()}
                            try:
                                tp = self.place_tp("short", vwap, qty)
                                self.tp_id_short = tp["id"]; self.last_tp_px_short = vwap
                            except Exception as e:  # noqa: BLE001
                                self.logger.warning(f"[SHORT] TP 발주 실패: {e}")
                            try:
                                sl = self.place_sl("short", sl_px, qty)
                                self.sl_id_short = sl["id"]; self.last_sl_px_short = sl_px
                            except Exception as e:  # noqa: BLE001
                                self.logger.warning(f"[SHORT] SL 발주 실패: {e}")
                            self.tlog.write(now_kst().isoformat(), "ENTRY", "short", avg_entry, None, qty, None, cur_eq, "")

                # 타임 스탑 & 재배치 & 체결감지 — LONG
                if self.long_pos is not None:
                    hold_min = (now_kst() - self.long_pos["entry_time"]).total_seconds() / 60.0
                    if hold_min >= self.cfg.time_stop_min_long:
                        self.logger.info(f"[TIME-STOP LONG] hold={hold_min:.1f}m → market close")
                        entry_px = self.long_pos["entry_px"]
                        exit_px = float(ind.iloc[-1]["close"])
                        pnl = (exit_px - entry_px) / entry_px
                        self.cancel_side_tp_sl("long")
                        self.ex.create_order(self.cfg.SYMBOL, "market", "sell", self.long_pos["qty"], None,
                                             {"positionSide": "LONG"})
                        cur_eq = get_equity_USDC(self.ex)
                        self.tlog.write(now_kst().isoformat(), "EXIT_TIME", "long", entry_px, exit_px, self.long_pos["qty"], pnl, cur_eq, "time_stop")
                        self.long_pos = None
                        self.consec_sl = 0
                    else:
                        row = ind.iloc[-1]
                        vwap = float(row["vwap"]); std = float(row["std"])
                        new_sl = max(1e-8, vwap - self.cfg.stop_k_long * std)
                        self.maybe_reprice_for_side("long", vwap, new_sl)
                        self._detect_and_finalize_exit("long", ind.iloc[-1])

                # 타임 스탑 & 재배치 & 체결감지 — SHORT
                if self.short_pos is not None:
                    hold_min = (now_kst() - self.short_pos["entry_time"]).total_seconds() / 60.0
                    if hold_min >= self.cfg.time_stop_min_short:
                        self.logger.info(f"[TIME-STOP SHORT] hold={hold_min:.1f}m → market close")
                        entry_px = self.short_pos["entry_px"]
                        exit_px = float(ind.iloc[-1]["close"])
                        pnl = (entry_px - exit_px) / entry_px
                        self.cancel_side_tp_sl("short")
                        self.ex.create_order(self.cfg.SYMBOL, "market", "buy", self.short_pos["qty"], None,
                                             {"positionSide": "SHORT"})
                        cur_eq = get_equity_USDC(self.ex)
                        self.tlog.write(now_kst().isoformat(), "EXIT_TIME", "short", entry_px, exit_px, self.short_pos["qty"], pnl, cur_eq, "time_stop")
                        self.short_pos = None
                        self.consec_sl = 0
                    else:
                        row = ind.iloc[-1]
                        vwap = float(row["vwap"]); std = float(row["std"])
                        new_sl = max(1e-8, vwap + self.cfg.stop_k_short * std)
                        self.maybe_reprice_for_side("short", vwap, new_sl)
                        self._detect_and_finalize_exit("short", ind.iloc[-1])

                # 하트비트
                self.heartbeat(ind.iloc[-1], cur_eq, day_dd)

            except Exception as e:  # noqa: BLE001
                self.logger.error(f"ERROR: {e}")
                traceback.print_exc()
                time.sleep(0.5)
                continue
