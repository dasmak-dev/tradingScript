# -*- coding: utf-8 -*-
"""
=====================================================================
 BOT DE TRADING INTRADÍA (SCALPING) — Streamlit + CCXT/yfinance + Plotly
=====================================================================

Modos de funcionamiento:
  1) Backtest      -> reproduce velas históricas recientes barra a barra (instantáneo).
  2) Paper Trading -> tiempo real con dinero SIMULADO (comisiones y slippage incluidos).
  3) Real          -> tiempo real con órdenes REALES en un exchange vía CCXT
                      (cripto). Soporta modo testnet/sandbox del exchange.

Ejecución:
  Interfaz:        streamlit run app.py
  Sin interfaz:    python app.py --headless --mode paper --symbol BTC/USDT
                   (ver `python app.py --help`)

AVISO: Software educativo. El trading intradía conlleva un riesgo elevado de
pérdida. Ningún resultado de backtest o paper trading garantiza resultados reales.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

# Las librerías de datos se importan de forma opcional para que el script
# arranque aunque falte una de ellas (se avisa en la interfaz).
try:
    import ccxt  # Criptomonedas: datos + ejecución real
except ImportError:  # pragma: no cover
    ccxt = None
try:
    import yfinance as yf  # Acciones / ETFs: solo datos (sin ejecución)
except ImportError:  # pragma: no cover
    yf = None


# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
@dataclass
class BotConfig:
    # --- Datos de mercado ---
    source: str = "ccxt"              # "ccxt" (cripto) | "yfinance" (acciones/ETF)
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"             # "1m" | "5m" | "15m"

    # --- Capital y gestión de riesgo ---
    initial_capital: float = 500.0    # Presupuesto máximo que el bot puede usar
    risk_per_trade_pct: float = 1.0   # % del capital arriesgado por operación (hasta el SL)
    max_position_pct: float = 95.0    # % máximo del efectivo invertido en una posición
    max_daily_loss_pct: float = 3.0   # STOP DIARIO: pérdida máxima del día (% del capital al inicio del día)
    daily_target_pct: float = 2.0     # META DIARIA: beneficio del día que detiene el bot hasta mañana
    global_stop_equity: float = 450.0  # STOP GLOBAL: si el capital cae por debajo, el bot se apaga
    global_target_equity: float = 600.0  # META GLOBAL: si el capital lo alcanza, el bot se apaga
    max_trades_per_day: int = 20
    fee_pct: float = 0.10             # Comisión por lado (%) usada en la simulación
    slippage_pct: float = 0.02        # Deslizamiento simulado (%)
    min_notional: float = 10.0        # Tamaño mínimo de orden (en moneda de cotización)

    # --- Estrategia (EMA + RSI + Bollinger + ATR) ---
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_entry_min: float = 45.0       # El cruce EMA solo vale si hay impulso (RSI > min)...
    rsi_entry_max: float = 65.0       # ...y no hay sobrecompra (RSI < max)
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stop_atr_mult: float = 1.5        # Stop Loss = entrada - 1.5 x ATR
    reward_risk: float = 2.0          # Take Profit = entrada + 2 x distancia del stop
    cooldown_bars: int = 2            # Velas de espera tras cerrar una operación

    # --- Sesión intradía (hora local de la zona indicada) ---
    session_tz: str = "UTC"
    session_start: str = "00:05"      # No se abre nada antes de esta hora
    no_entry_after: str = "23:30"     # No se abren posiciones nuevas a partir de aquí
    flatten_at: str = "23:50"         # Cierre OBLIGATORIO de cualquier posición abierta

    def quote_ccy(self) -> str:
        if self.source == "ccxt" and "/" in self.symbol:
            return self.symbol.split("/")[1].split(":")[0]
        return "USD"

    def base_ccy(self) -> str:
        if "/" in self.symbol:
            return self.symbol.split("/")[0]
        return self.symbol

    @classmethod
    def from_dict(cls, d: dict) -> "BotConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Valores por defecto de sesión según el tipo de mercado
SESSION_PRESETS = {
    # Cripto cotiza 24/7: definimos un "día de trading" en UTC y cerramos todo antes de medianoche
    "ccxt": dict(session_tz="UTC", session_start="00:05", no_entry_after="23:30", flatten_at="23:50"),
    # Bolsa EE.UU. (NYSE/NASDAQ): 09:30-16:00 hora de Nueva York
    "yfinance": dict(session_tz="America/New_York", session_start="09:35",
                     no_entry_after="15:30", flatten_at="15:55"),
}

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


def _parse_hhmm(s: str) -> dtime:
    h, m = s.strip().split(":")
    return dtime(int(h), int(m))


# =====================================================================
# 2. DATOS DE MERCADO
# =====================================================================
def make_public_exchange(exchange_id: str):
    """Instancia CCXT sin claves (solo datos públicos)."""
    if ccxt is None:
        raise RuntimeError("Falta la librería ccxt (pip install ccxt)")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_ohlcv_ccxt(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Descarga velas con paginación (los exchanges limitan ~1000 velas por petición)."""
    tf_ms = TF_SECONDS[timeframe] * 1000
    since = exchange.milliseconds() - limit * tf_ms
    rows: list = []
    while len(rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=min(1000, limit - len(rows)))
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 2 or since > exchange.milliseconds():
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float).sort_index().tail(limit)


def fetch_ohlcv_yf(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """yfinance: 1m permite ~7 días de histórico; 5m/15m hasta 60 días."""
    if yf is None:
        raise RuntimeError("Falta la librería yfinance (pip install yfinance)")
    period = "7d" if timeframe == "1m" else "60d"
    df = yf.Ticker(symbol).history(period=period, interval=timeframe, prepost=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance no devolvió datos para {symbol}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC"))
    df.index.name = "ts"
    return df.astype(float).tail(limit)


def fetch_data(cfg: BotConfig, limit: int, exchange=None) -> pd.DataFrame:
    if cfg.source == "ccxt":
        return fetch_ohlcv_ccxt(exchange or make_public_exchange(cfg.exchange_id), cfg.symbol, cfg.timeframe, limit)
    return fetch_ohlcv_yf(cfg.symbol, cfg.timeframe, limit)


def split_closed_bars(df: pd.DataFrame, timeframe: str, now: Optional[pd.Timestamp] = None):
    """Separa velas CERRADAS (las únicas que generan señales) de la vela en curso."""
    now = now or pd.Timestamp.now(tz="UTC")
    closed_mask = df.index + pd.Timedelta(seconds=TF_SECONDS[timeframe]) <= now
    return df[closed_mask], df[~closed_mask]


# =====================================================================
# 3. INDICADORES TÉCNICOS
# =====================================================================
def compute_indicators(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    """Todos los indicadores son causales: el valor en la vela i solo usa datos <= i."""
    out = df.copy()
    c = out["close"]

    # Medias móviles exponenciales (tendencia de corto plazo)
    out["ema_fast"] = c.ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = c.ewm(span=cfg.ema_slow, adjust=False).mean()

    # RSI de Wilder (impulso / sobrecompra-sobreventa)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    a = 1.0 / cfg.rsi_period
    avg_gain = gain.ewm(alpha=a, adjust=False, min_periods=cfg.rsi_period).mean()
    avg_loss = loss.ewm(alpha=a, adjust=False, min_periods=cfg.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).where(avg_loss != 0, 100.0)

    # Bandas de Bollinger (volatilidad / extremos)
    mid = c.rolling(cfg.bb_period).mean()
    std = c.rolling(cfg.bb_period).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_up"] = mid + cfg.bb_std * std
    out["bb_low"] = mid - cfg.bb_std * std

    # ATR (volatilidad absoluta -> tamaño del Stop Loss)
    prev_c = c.shift(1)
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev_c).abs(), (out["low"] - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    return out


# =====================================================================
# 4. ESTRATEGIA — reglas de entrada y salida con su explicación textual
# =====================================================================
def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/d"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6g}"


def entry_signal(prev: pd.Series, row: pd.Series, cfg: BotConfig) -> Optional[str]:
    """Devuelve el MOTIVO de compra (texto) o None si no hay señal. Solo largos (spot)."""
    needed = ["ema_fast", "ema_slow", "rsi", "bb_low", "bb_mid", "atr"]
    if any(pd.isna(row[k]) or pd.isna(prev[k]) for k in needed):
        return None

    # Regla A — MOMENTUM: cruce alcista de EMAs con impulso confirmado por RSI
    cross_up = prev["ema_fast"] <= prev["ema_slow"] and row["ema_fast"] > row["ema_slow"]
    if cross_up and cfg.rsi_entry_min < row["rsi"] < cfg.rsi_entry_max:
        return (f"Comprando porque la EMA {cfg.ema_fast} ({_fmt(row['ema_fast'])}) cruza al alza la "
                f"EMA {cfg.ema_slow} ({_fmt(row['ema_slow'])}) y el RSI={row['rsi']:.1f} confirma impulso "
                f"sin sobrecompra ({cfg.rsi_entry_min:.0f} < RSI < {cfg.rsi_entry_max:.0f}).")

    # Regla B — REBOTE: el precio estaba fuera de la Bollinger inferior con RSI en sobreventa y vuelve a entrar
    rebound = prev["close"] < prev["bb_low"] and row["close"] > row["bb_low"]
    if rebound and min(prev["rsi"], row["rsi"]) < cfg.rsi_oversold:
        return (f"Comprando porque el precio rebota en la Banda de Bollinger inferior "
                f"(cierre previo {_fmt(prev['close'])} < banda {_fmt(prev['bb_low'])}, ahora {_fmt(row['close'])} "
                f"vuelve dentro) con RSI={min(prev['rsi'], row['rsi']):.1f} < {cfg.rsi_oversold:.0f} (sobreventa).")
    return None


def exit_signal(prev: pd.Series, row: pd.Series, cfg: BotConfig) -> Optional[str]:
    """Motivo de venta técnica (además del SL/TP, que se gestiona en el motor)."""
    if pd.isna(row["ema_fast"]) or pd.isna(prev["ema_fast"]):
        return None
    cross_down = prev["ema_fast"] >= prev["ema_slow"] and row["ema_fast"] < row["ema_slow"]
    if cross_down:
        return (f"Vendiendo porque la EMA {cfg.ema_fast} ({_fmt(row['ema_fast'])}) cruza a la baja la "
                f"EMA {cfg.ema_slow} ({_fmt(row['ema_slow'])}): se pierde el impulso alcista.")
    if row["rsi"] > cfg.rsi_overbought and row["close"] >= row["bb_up"]:
        return (f"Vendiendo porque RSI={row['rsi']:.1f} > {cfg.rsi_overbought:.0f} y el precio "
                f"({_fmt(row['close'])}) toca la Banda de Bollinger superior ({_fmt(row['bb_up'])}): sobrecompra, "
                f"se asegura beneficio.")
    return None


# =====================================================================
# 5. BROKERS — Paper (simulado) y Real (CCXT)
# =====================================================================
@dataclass
class Fill:
    qty: float
    price: float
    fee: float
    order_id: str = ""


class PaperBroker:
    """Ejecución simulada: precio de referencia + slippage + comisión."""
    is_live = False
    name = "Paper"

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def buy(self, qty: float, ref_price: float) -> Fill:
        price = ref_price * (1 + self.cfg.slippage_pct / 100)
        return Fill(qty, price, qty * price * self.cfg.fee_pct / 100, "paper")

    def sell(self, qty: float, ref_price: float) -> Fill:
        price = ref_price * (1 - self.cfg.slippage_pct / 100)
        return Fill(qty, price, qty * price * self.cfg.fee_pct / 100, "paper")


class LiveCCXTBroker:
    """
    Ejecución REAL en un exchange de cripto mediante CCXT (órdenes a mercado, spot, sin apalancamiento).
    El bot NUNCA usa más saldo que el presupuesto configurado (initial_capital), aunque la cuenta tenga más.
    """
    is_live = True

    def __init__(self, cfg: BotConfig, api_key: str, secret: str, password: str = "", sandbox: bool = False):
        if ccxt is None:
            raise RuntimeError("Falta la librería ccxt (pip install ccxt)")
        params = {"apiKey": api_key, "secret": secret, "enableRateLimit": True,
                  "options": {"defaultType": "spot"}}
        if password:
            params["password"] = password
        self.cfg = cfg
        self.ex = getattr(ccxt, cfg.exchange_id)(params)
        if sandbox:
            self.ex.set_sandbox_mode(True)  # Testnet: órdenes reales contra dinero ficticio del exchange
        self.ex.load_markets()
        if cfg.symbol not in self.ex.markets:
            raise RuntimeError(f"El símbolo {cfg.symbol} no existe en {cfg.exchange_id}")
        self.market = self.ex.market(cfg.symbol)
        self.name = f"REAL {cfg.exchange_id}{' (testnet)' if sandbox else ''}"

    # --- utilidades ---
    def free_balance(self, ccy: str) -> float:
        bal = self.ex.fetch_balance()
        return float((bal.get("free") or {}).get(ccy) or 0.0)

    def _check_limits(self, qty: float, price: float):
        lim = self.market.get("limits") or {}
        min_amt = (lim.get("amount") or {}).get("min")
        min_cost = (lim.get("cost") or {}).get("min")
        if min_amt and qty < min_amt:
            raise RuntimeError(f"Cantidad {qty} menor que el mínimo del exchange ({min_amt})")
        if min_cost and qty * price < min_cost:
            raise RuntimeError(f"Importe {qty * price:.2f} menor que el mínimo del exchange ({min_cost})")

    def _wait_order(self, order: dict) -> dict:
        for _ in range(10):
            if order.get("status") in ("closed", "canceled") and order.get("filled"):
                return order
            time.sleep(0.5)
            try:
                order = self.ex.fetch_order(order["id"], self.cfg.symbol)
            except Exception:
                break
        return order

    def _fee_in_quote(self, order: dict, price: float, qty: float) -> float:
        fee = order.get("fee") or {}
        cost, ccy = fee.get("cost"), fee.get("currency")
        if cost is not None:
            if ccy == self.cfg.quote_ccy():
                return float(cost)
            if ccy == self.cfg.base_ccy():
                return float(cost) * price
        return qty * price * self.cfg.fee_pct / 100  # estimación si el exchange no informa

    # --- órdenes ---
    def buy(self, qty: float, ref_price: float) -> Fill:
        qty = float(self.ex.amount_to_precision(self.cfg.symbol, qty))
        self._check_limits(qty, ref_price)
        free = self.free_balance(self.cfg.quote_ccy())
        if free < qty * ref_price * (1 + self.cfg.fee_pct / 100):
            raise RuntimeError(f"Saldo insuficiente en el exchange: {free:.2f} {self.cfg.quote_ccy()}")
        order = self._wait_order(self.ex.create_order(self.cfg.symbol, "market", "buy", qty))
        filled = float(order.get("filled") or qty)
        price = float(order.get("average") or order.get("price") or ref_price)
        return Fill(filled, price, self._fee_in_quote(order, price, filled), str(order.get("id")))

    def sell(self, qty: float, ref_price: float) -> Fill:
        # Se vende lo que realmente hay (la comisión de compra pudo descontarse en la moneda base)
        qty = min(qty, self.free_balance(self.cfg.base_ccy()))
        qty = float(self.ex.amount_to_precision(self.cfg.symbol, qty))
        if qty <= 0:
            raise RuntimeError("No hay saldo de la moneda base para vender")
        order = self._wait_order(self.ex.create_order(self.cfg.symbol, "market", "sell", qty))
        filled = float(order.get("filled") or qty)
        price = float(order.get("average") or order.get("price") or ref_price)
        return Fill(filled, price, self._fee_in_quote(order, price, filled), str(order.get("id")))


# =====================================================================
# 6. MOTOR DEL BOT — estrategia + GESTIÓN DE RIESGO
# =====================================================================
@dataclass
class Position:
    qty: float
    entry_price: float
    entry_time: str
    stop: float
    target: float
    entry_fee: float
    reason: str


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    fees: float
    entry_reason: str
    exit_reason: str


STATUS_ACTIVE = "🟢 ACTIVO"
STATUS_PAUSED = "⏸️ PAUSADO (solo gestiona la posición abierta)"
STATUS_STOPPED = "⏹️ DETENIDO"
STATUS_DAILY_TARGET = "🎯 META DIARIA ALCANZADA (reanuda mañana)"
STATUS_DAILY_STOP = "🛑 STOP DIARIO ALCANZADO (reanuda mañana)"
STATUS_GLOBAL_TARGET = "🏁 META GLOBAL ALCANZADA (bot apagado)"
STATUS_GLOBAL_STOP = "⛔ STOP GLOBAL ALCANZADO (bot apagado)"


class TradingBot:
    def __init__(self, cfg: BotConfig, broker):
        self.cfg = cfg
        self.broker = broker
        self.cash: float = cfg.initial_capital
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.log: list[dict] = []
        self.markers: list[dict] = []
        self.equity_curve: list[dict] = []
        self.status: str = STATUS_ACTIVE
        self.paused = False
        self.halted_today = False
        self.halted_global = False
        self.day: Optional[str] = None
        self.day_start_equity: float = cfg.initial_capital
        self.trades_today = 0
        self.bar_count = 0
        self.cooldown_until = 0
        self.last_bar_time: Optional[pd.Timestamp] = None
        self.last_price: Optional[float] = None
        self._t_start = _parse_hhmm(cfg.session_start)
        self._t_no_entry = _parse_hhmm(cfg.no_entry_after)
        self._t_flatten = _parse_hhmm(cfg.flatten_at)

    # ---------- utilidades ----------
    def equity(self, price: Optional[float] = None) -> float:
        price = price if price is not None else self.last_price
        if self.position and price:
            return self.cash + self.position.qty * price
        return self.cash

    def floating_pnl(self, price: Optional[float] = None) -> float:
        price = price if price is not None else self.last_price
        if not self.position or not price:
            return 0.0
        p = self.position
        return p.qty * price - p.qty * p.entry_price - p.entry_fee

    def _log(self, ts, kind: str, msg: str):
        self.log.append({"hora": _ts_str(ts, self.cfg.session_tz), "tipo": kind, "mensaje": msg})

    # ---------- arranque en tiempo real ----------
    def prime(self, closed_df: pd.DataFrame):
        """Tiempo real: las velas pasadas solo sirven para calentar indicadores, no se operan."""
        if len(closed_df):
            self.last_bar_time = closed_df.index[-1]
            self.last_price = float(closed_df["close"].iloc[-1])
            self._roll_day(closed_df.index[-1], self.last_price)
            self._log(closed_df.index[-1], "INFO",
                      f"Bot iniciado ({self.broker.name}). Capital asignado: {self.cfg.initial_capital:.2f} "
                      f"{self.cfg.quote_ccy()}. Esperando la siguiente vela cerrada de {self.cfg.timeframe}.")

    # ---------- bucle principal: procesa SOLO velas cerradas nuevas ----------
    def step(self, ind_df: pd.DataFrame):
        if len(ind_df) < 2:
            return
        start = 1
        if self.last_bar_time is not None:
            new_idx = np.where(ind_df.index > self.last_bar_time)[0]
            if len(new_idx) == 0:
                return
            start = max(1, int(new_idx[0]))
        for i in range(start, len(ind_df)):
            self._on_bar(ind_df.index[i], ind_df.iloc[i - 1], ind_df.iloc[i])
            self.last_bar_time = ind_df.index[i]

    # ---------- comprobación entre velas con el precio actual (tiempo real) ----------
    def check_price(self, price: float, now: pd.Timestamp):
        """Stop Loss / Take Profit / límites de riesgo con el precio en vivo (sin esperar al cierre de vela)."""
        self.last_price = price
        if self.position:
            if price <= self.position.stop:
                self._close(now, price, f"Stop Loss alcanzado en vivo: precio {_fmt(price)} <= SL {_fmt(self.position.stop)}.")
            elif price >= self.position.target:
                self._close(now, price, f"Take Profit alcanzado en vivo: precio {_fmt(price)} >= TP {_fmt(self.position.target)}.")
        local = now.tz_convert(self.cfg.session_tz)
        if self.position and local.time() >= self._t_flatten:
            self._close(now, price, f"Cierre intradía obligatorio: son las {local:%H:%M} (límite {self.cfg.flatten_at}).")
        self._check_risk_limits(now, price)

    # ---------- lógica por vela ----------
    def _roll_day(self, ts: pd.Timestamp, price: float):
        day = ts.tz_convert(self.cfg.session_tz).strftime("%Y-%m-%d")
        if day == self.day:
            return
        if self.position:  # No debería ocurrir (se cierra antes), pero por seguridad
            self._close(ts, price, "Cierre forzado: cambio de día con la posición abierta (intradía estricto).")
        self.day = day
        self.day_start_equity = self.equity(price)
        self.trades_today = 0
        if self.halted_today and not self.halted_global:
            self._log(ts, "RIESGO", "Nuevo día de trading: se reinician los límites diarios y el bot se reactiva.")
        self.halted_today = False
        if not self.halted_global:
            self.status = STATUS_PAUSED if self.paused else STATUS_ACTIVE

    def _on_bar(self, ts: pd.Timestamp, prev: pd.Series, row: pd.Series):
        self.bar_count += 1
        close = float(row["close"])
        self._roll_day(ts, float(row["open"]))
        local_t = ts.tz_convert(self.cfg.session_tz).time()

        # (1) Stop Loss / Take Profit dentro de la vela (usa máximos y mínimos)
        if self.position:
            p = self.position
            if row["low"] <= p.stop:  # Si en la misma vela se tocan SL y TP, se asume SL (conservador)
                px = min(p.stop, float(row["open"]))
                self._close(ts, px, f"Stop Loss tocado: mínimo de la vela {_fmt(row['low'])} <= SL {_fmt(p.stop)}. "
                                    f"Se limita la pérdida según el plan.")
            elif row["high"] >= p.target:
                px = max(p.target, float(row["open"]))
                self._close(ts, px, f"Take Profit tocado: máximo de la vela {_fmt(row['high'])} >= TP {_fmt(p.target)} "
                                    f"({self.cfg.reward_risk:.1f}R).")

        # (2) Cierre intradía obligatorio: nunca se mantiene una posición tras la hora límite
        if self.position and local_t >= self._t_flatten:
            self._close(ts, close, f"Cierre intradía obligatorio: vela de las {local_t:%H:%M} >= hora límite {self.cfg.flatten_at}. "
                                   f"No se dejan posiciones abiertas por la noche.")

        # (3) Salida técnica
        if self.position:
            reason = exit_signal(prev, row, self.cfg)
            if reason:
                self._close(ts, close, reason)

        self.last_price = close
        # (4) Límites de riesgo diarios y globales
        self._check_risk_limits(ts, close)
        self.equity_curve.append({"ts": ts, "equity": self.equity(close)})

        # (5) Entrada
        if self.position or self.halted_today or self.halted_global:
            return
        reason = entry_signal(prev, row, self.cfg)
        if not reason:
            return
        if self.paused:
            self._log(ts, "OMITIDA", "Señal de compra ignorada: el bot está en pausa. " + reason)
            return
        if not (self._t_start <= local_t < self._t_no_entry):
            self._log(ts, "OMITIDA", f"Señal de compra ignorada: fuera del horario de entradas "
                                     f"({self.cfg.session_start}-{self.cfg.no_entry_after}).")
            return
        if self.trades_today >= self.cfg.max_trades_per_day:
            self._log(ts, "OMITIDA", f"Señal de compra ignorada: máximo de {self.cfg.max_trades_per_day} operaciones diarias.")
            return
        if self.bar_count < self.cooldown_until:
            self._log(ts, "OMITIDA", "Señal de compra ignorada: periodo de enfriamiento tras la última operación.")
            return
        self._open(ts, close, float(row["atr"]), reason)

    # ---------- GESTIÓN DE RIESGO: tamaño de posición ----------
    def _open(self, ts, price: float, atr: float, reason: str):
        cfg = self.cfg
        # Distancia del stop basada en volatilidad (ATR), con un mínimo del 0.1% para evitar stops ridículos
        stop_dist = max(atr * cfg.stop_atr_mult, price * 0.001)
        equity = self.equity(price)

        # Riesgo monetario = % del capital... pero nunca más de lo que queda hasta el STOP DIARIO
        # ni de lo que queda hasta el STOP GLOBAL.
        risk_amount = equity * cfg.risk_per_trade_pct / 100
        daily_room = self.day_start_equity * cfg.max_daily_loss_pct / 100 + (equity - self.day_start_equity)
        global_room = equity - cfg.global_stop_equity
        risk_amount = min(risk_amount, daily_room, global_room)
        if risk_amount <= 0:
            self._log(ts, "OMITIDA", "Señal de compra ignorada: no queda margen de pérdida dentro de los límites de riesgo.")
            return

        qty_by_risk = risk_amount / stop_dist
        max_notional = self.cash * cfg.max_position_pct / 100 / (1 + cfg.fee_pct / 100)
        qty = min(qty_by_risk, max_notional / price)
        if qty * price < cfg.min_notional:
            self._log(ts, "OMITIDA", f"Señal de compra ignorada: el tamaño calculado ({qty * price:.2f} "
                                     f"{cfg.quote_ccy()}) es menor que el mínimo ({cfg.min_notional}).")
            return
        try:
            fill = self.broker.buy(qty, price)
        except Exception as e:
            self._log(ts, "ERROR", f"Fallo al enviar la orden de compra: {e}")
            return

        stop = fill.price - stop_dist
        target = fill.price + stop_dist * cfg.reward_risk
        self.cash -= fill.qty * fill.price + fill.fee
        self.position = Position(fill.qty, fill.price, str(ts), stop, target, fill.fee, reason)
        self.trades_today += 1
        sizing = (f" | Tamaño: {_fmt(fill.qty)} {cfg.base_ccy()} ≈ {fill.qty * fill.price:.2f} {cfg.quote_ccy()}; "
                  f"riesgo máx. {fill.qty * stop_dist:.2f} {cfg.quote_ccy()}; SL {_fmt(stop)} ({cfg.stop_atr_mult}×ATR), "
                  f"TP {_fmt(target)} ({cfg.reward_risk}R).")
        self._log(ts, "COMPRA", reason + sizing)
        self.markers.append({"ts": ts, "price": fill.price, "side": "buy", "text": reason + sizing})

    def _close(self, ts, price: float, reason: str):
        p = self.position
        if not p:
            return
        try:
            fill = self.broker.sell(p.qty, price)
        except Exception as e:
            self._log(ts, "ERROR", f"Fallo al enviar la orden de venta ({e}). Se reintentará en la próxima actualización.")
            return
        proceeds = fill.qty * fill.price - fill.fee
        self.cash += proceeds
        pnl = proceeds - (p.qty * p.entry_price + p.entry_fee)
        pnl_pct = pnl / (p.qty * p.entry_price) * 100
        self.trades.append(Trade(p.entry_time, str(ts), p.entry_price, fill.price, p.qty, pnl, pnl_pct,
                                 p.entry_fee + fill.fee, p.reason, reason))
        self.position = None
        self.cooldown_until = self.bar_count + self.cfg.cooldown_bars
        result = "GANADORA ✅" if pnl > 0 else "PERDEDORA ❌"
        msg = f"{reason} | Resultado: {pnl:+.2f} {self.cfg.quote_ccy()} ({pnl_pct:+.2f}%) — {result}"
        self._log(ts, "VENTA", msg)
        self.markers.append({"ts": ts, "price": fill.price, "side": "sell", "text": msg})

    # ---------- GESTIÓN DE RIESGO: límites diarios y globales ----------
    def _check_risk_limits(self, ts, price: float):
        # Si un límite ya detuvo el bot pero el cierre falló (p.ej. error de red), se reintenta
        if (self.halted_today or self.halted_global) and self.position:
            self._close(ts, price, "Reintento de cierre: hay un límite de riesgo activo.")
        if self.halted_global:
            return
        cfg = self.cfg
        eq = self.equity(price)
        day_pnl = eq - self.day_start_equity
        max_loss = self.day_start_equity * cfg.max_daily_loss_pct / 100
        target = self.day_start_equity * cfg.daily_target_pct / 100

        if eq <= cfg.global_stop_equity:
            self._close(ts, price, f"STOP GLOBAL: capital {eq:.2f} <= mínimo permitido {cfg.global_stop_equity:.2f}.")
            self.halted_global, self.status = True, STATUS_GLOBAL_STOP
            self._log(ts, "RIESGO", "Bot apagado por STOP GLOBAL. Revisa la estrategia antes de reanudar.")
        elif eq >= cfg.global_target_equity:
            self._close(ts, price, f"META GLOBAL: capital {eq:.2f} >= objetivo {cfg.global_target_equity:.2f}.")
            self.halted_global, self.status = True, STATUS_GLOBAL_TARGET
            self._log(ts, "RIESGO", "¡Meta global alcanzada! El bot se apaga para proteger el beneficio.")
        elif not self.halted_today and day_pnl <= -max_loss:
            self._close(ts, price, f"STOP DIARIO: P/L del día {day_pnl:+.2f} <= -{max_loss:.2f} ({cfg.max_daily_loss_pct}%).")
            self.halted_today, self.status = True, STATUS_DAILY_STOP
            self._log(ts, "RIESGO", "Pérdida diaria máxima alcanzada: no se opera más hoy.")
        elif not self.halted_today and day_pnl >= target:
            self._close(ts, price, f"META DIARIA: P/L del día {day_pnl:+.2f} >= {target:.2f} ({cfg.daily_target_pct}%).")
            self.halted_today, self.status = True, STATUS_DAILY_TARGET
            self._log(ts, "RIESGO", "Meta diaria alcanzada: el bot descansa hasta mañana.")

    # ---------- control manual ----------
    def set_paused(self, paused: bool, ts=None):
        self.paused = paused
        if not (self.halted_today or self.halted_global):
            self.status = STATUS_PAUSED if paused else STATUS_ACTIVE
        self._log(ts or pd.Timestamp.now(tz="UTC"), "INFO", "Bot pausado." if paused else "Bot reanudado.")

    def stop(self, ts=None):
        ts = ts or pd.Timestamp.now(tz="UTC")
        if self.position and self.last_price:
            self._close(ts, self.last_price, "Cierre manual: el usuario ha detenido el bot.")
        if not self.halted_global:
            self.status = STATUS_STOPPED
        self._log(ts, "INFO", "Bot detenido.")

    # ---------- estadísticas ----------
    def stats(self) -> dict:
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        realized = sum(t.pnl for t in self.trades)
        eq = self.equity()
        curve = pd.Series([e["equity"] for e in self.equity_curve]) if self.equity_curve else pd.Series([eq])
        dd = ((curve - curve.cummax()) / curve.cummax() * 100).min()
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        return {
            "equity": eq, "cash": self.cash, "floating": self.floating_pnl(),
            "realized": realized, "total": eq - self.cfg.initial_capital,
            "day_pnl": eq - self.day_start_equity,
            "n": len(self.trades), "wins": len(wins), "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
            "max_dd": float(dd) if not math.isnan(dd) else 0.0,
            "fees": sum(t.fees for t in self.trades),
        }

    # ---------- persistencia (para recuperar el estado si se cae la app) ----------
    def to_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg), "cash": self.cash,
            "position": asdict(self.position) if self.position else None,
            "trades": [asdict(t) for t in self.trades], "log": self.log,
            "markers": [{**m, "ts": str(m["ts"])} for m in self.markers],
            "equity_curve": [{"ts": str(e["ts"]), "equity": e["equity"]} for e in self.equity_curve],
            "status": self.status, "paused": self.paused, "halted_today": self.halted_today,
            "halted_global": self.halted_global, "day": self.day, "day_start_equity": self.day_start_equity,
            "trades_today": self.trades_today,
            "last_bar_time": str(self.last_bar_time) if self.last_bar_time is not None else None,
            "last_price": self.last_price, "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_dict(self, d: dict):
        self.cash = d["cash"]
        self.position = Position(**d["position"]) if d.get("position") else None
        self.trades = [Trade(**t) for t in d.get("trades", [])]
        self.log = d.get("log", [])
        self.markers = [{**m, "ts": pd.Timestamp(m["ts"])} for m in d.get("markers", [])]
        self.equity_curve = [{"ts": pd.Timestamp(e["ts"]), "equity": e["equity"]} for e in d.get("equity_curve", [])]
        for k in ("status", "paused", "halted_today", "halted_global", "day", "day_start_equity",
                  "trades_today", "last_price"):
            setattr(self, k, d.get(k, getattr(self, k)))
        # last_bar_time no se restaura: tras reanudar se vuelve a "calentar" con prime()


def state_path(mode: str, cfg: BotConfig) -> str:
    safe = cfg.symbol.replace("/", "-").replace(":", "-")
    return f"bot_state_{mode}_{safe}.json"


def save_state(bot: TradingBot, path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bot.to_dict(), f, ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _ts_str(ts, tz: str) -> str:
    try:
        return pd.Timestamp(ts).tz_convert(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def run_backtest(cfg: BotConfig, df: pd.DataFrame) -> TradingBot:
    """Reproduce las velas en orden, como si llegasen en tiempo real (sin mirar al futuro)."""
    bot = TradingBot(cfg, PaperBroker(cfg))
    ind = compute_indicators(df, cfg)
    bot._log(ind.index[0], "INFO", f"Backtest iniciado sobre {len(ind)} velas de {cfg.timeframe} ({cfg.symbol}).")
    bot.step(ind)
    if bot.position:  # El backtest termina sin posiciones abiertas
        bot._close(ind.index[-1], float(ind["close"].iloc[-1]), "Fin del backtest: cierre de la posición abierta.")
        bot.equity_curve.append({"ts": ind.index[-1], "equity": bot.equity()})
    return bot


# =====================================================================
# 7. INTERFAZ STREAMLIT
# =====================================================================
def build_chart(ind: pd.DataFrame, bot: Optional[TradingBot], cfg: BotConfig, max_bars: int = 300):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = ind.tail(max_bars).copy()
    d.index = d.index.tz_convert(cfg.session_tz)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26],
                        vertical_spacing=0.03, subplot_titles=(None, "RSI"))
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Precio", increasing_line_color="#26A69A", decreasing_line_color="#EF5350"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_up"], name="Bollinger sup.", line=dict(color="#9AA0A6", width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_low"], name="Bollinger inf.", line=dict(color="#9AA0A6", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(154,160,166,0.08)"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema_fast"], name=f"EMA {cfg.ema_fast}", line=dict(color="#F2A93B", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema_slow"], name=f"EMA {cfg.ema_slow}", line=dict(color="#4C8DF6", width=2)), 1, 1)

    if bot and bot.markers:
        t0 = ind.index[-len(d)]
        for side, sym, color, label, yshift in (("buy", "triangle-up", "#00C853", "COMPRA", "low"),
                                                ("sell", "triangle-down", "#FF1744", "VENTA", "high")):
            ms = [m for m in bot.markers if m["side"] == side and pd.Timestamp(m["ts"]) >= t0]
            if not ms:
                continue
            xs = [pd.Timestamp(m["ts"]).tz_convert(cfg.session_tz) for m in ms]
            hover = ["<b>" + label + "</b> @ " + _fmt(m["price"]) + "<br>" +
                     "<br>".join(textwrap.wrap(m["text"], 70)) for m in ms]
            fig.add_trace(go.Scatter(x=xs, y=[m["price"] for m in ms], mode="markers+text", name=label,
                                     text=["B" if side == "buy" else "S"] * len(ms),
                                     textposition="bottom center" if side == "buy" else "top center",
                                     marker=dict(symbol=sym, size=14, color=color, line=dict(width=1, color="white")),
                                     hovertext=hover, hoverinfo="text"), 1, 1)
        if bot.position:
            fig.add_hline(y=bot.position.stop, line=dict(color="#FF1744", width=1, dash="dash"),
                          annotation_text="SL", row=1, col=1)
            fig.add_hline(y=bot.position.target, line=dict(color="#00C853", width=1, dash="dash"),
                          annotation_text="TP", row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI", line=dict(color="#B07CFF", width=2), showlegend=False), 2, 1)
    fig.add_hline(y=cfg.rsi_overbought, line=dict(color="#9AA0A6", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=cfg.rsi_oversold, line=dict(color="#9AA0A6", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def build_equity_chart(bot: TradingBot, cfg: BotConfig):
    import plotly.graph_objects as go
    if not bot.equity_curve:
        return None
    e = pd.DataFrame(bot.equity_curve)
    e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.tz_convert(cfg.session_tz)
    fig = go.Figure(go.Scatter(x=e["ts"], y=e["equity"], name="Capital", line=dict(color="#4C8DF6", width=2)))
    fig.add_hline(y=cfg.initial_capital, line=dict(color="#9AA0A6", width=1, dash="dot"), annotation_text="Inicial")
    fig.add_hline(y=cfg.global_stop_equity, line=dict(color="#FF1744", width=1, dash="dash"), annotation_text="Stop global")
    fig.add_hline(y=cfg.global_target_equity, line=dict(color="#00C853", width=1, dash="dash"), annotation_text="Meta global")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def _get_secret(name: str) -> str:
    """Busca claves en .streamlit/secrets.toml o variables de entorno (nunca en el código)."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def sidebar_config():
    import streamlit as st
    sb = st.sidebar
    sb.title("⚙️ Configuración")
    mode = sb.radio("Modo de operación", ["Backtest", "Paper Trading", "Real"],
                    help="Backtest: histórico reciente. Paper: tiempo real simulado. Real: órdenes reales en el exchange.")

    sb.subheader("📈 Activo")
    source_label = sb.selectbox("Mercado", ["Cripto (CCXT)", "Acciones/ETF (yfinance)"],
                                disabled=(mode == "Real"),
                                help="El modo Real solo está disponible para cripto vía CCXT.")
    source = "ccxt" if source_label.startswith("Cripto") or mode == "Real" else "yfinance"
    preset = SESSION_PRESETS[source]
    if source == "ccxt":
        exchange_id = sb.selectbox("Exchange", ["binance", "kraken", "bitget", "bybit", "okx", "kucoin", "coinbase"])
        symbol = sb.text_input("Par", "BTC/USDT" if exchange_id != "kraken" else "BTC/EUR")
    else:
        exchange_id = "binance"
        symbol = sb.text_input("Ticker", "SPY", help="Ej: SPY, QQQ, AAPL, TSLA, SAN.MC (mercado EE.UU. por defecto)")
    timeframe = sb.selectbox("Temporalidad", ["1m", "5m", "15m"], index=0)

    cfg0 = BotConfig()
    sb.subheader("💰 Capital y riesgo")
    initial = sb.number_input("Capital inicial", 20.0, 1e7, 500.0, 50.0)
    c1, c2 = sb.columns(2)
    max_loss = c1.number_input("Stop diario (%)", 0.1, 50.0, cfg0.max_daily_loss_pct, 0.5)
    target = c2.number_input("Meta diaria (%)", 0.1, 100.0, cfg0.daily_target_pct, 0.5)
    g_stop = c1.number_input("Stop global (capital)", 0.0, 1e7, round(initial * 0.9, 2), 10.0)
    g_target = c2.number_input("Meta global (capital)", 0.0, 1e8, round(initial * 1.2, 2), 10.0)
    risk = c1.number_input("Riesgo/operación (%)", 0.1, 10.0, cfg0.risk_per_trade_pct, 0.1)
    max_trades = c2.number_input("Máx. operaciones/día", 1, 500, cfg0.max_trades_per_day)
    fee = c1.number_input("Comisión (%/lado)", 0.0, 2.0, cfg0.fee_pct, 0.01)
    slip = c2.number_input("Slippage simulado (%)", 0.0, 2.0, cfg0.slippage_pct, 0.01)

    with sb.expander("🧠 Estrategia"):
        ema_f = st.number_input("EMA rápida", 2, 100, cfg0.ema_fast)
        ema_s = st.number_input("EMA lenta", 3, 300, cfg0.ema_slow)
        rsi_p = st.number_input("Periodo RSI", 2, 50, cfg0.rsi_period)
        rsi_os, rsi_ob = st.slider("RSI sobreventa / sobrecompra", 5, 95, (int(cfg0.rsi_oversold), int(cfg0.rsi_overbought)))
        rsi_lo, rsi_hi = st.slider("Rango RSI para entrar en cruce EMA", 5, 95, (int(cfg0.rsi_entry_min), int(cfg0.rsi_entry_max)))
        bb_p = st.number_input("Periodo Bollinger", 5, 100, cfg0.bb_period)
        bb_k = st.number_input("Desviaciones Bollinger", 0.5, 4.0, cfg0.bb_std, 0.1)
        atr_m = st.number_input("Stop = ATR ×", 0.3, 10.0, cfg0.stop_atr_mult, 0.1)
        rr = st.number_input("Take Profit (R:R)", 0.3, 10.0, cfg0.reward_risk, 0.1)

    with sb.expander("🕒 Sesión intradía"):
        tz = st.text_input("Zona horaria", preset["session_tz"])
        s_start = st.text_input("Inicio entradas (HH:MM)", preset["session_start"])
        s_noent = st.text_input("Sin entradas desde (HH:MM)", preset["no_entry_after"])
        s_flat = st.text_input("Cierre obligatorio (HH:MM)", preset["flatten_at"])

    cfg = BotConfig(source=source, exchange_id=exchange_id, symbol=symbol.strip(), timeframe=timeframe,
                    initial_capital=initial, risk_per_trade_pct=risk, max_daily_loss_pct=max_loss,
                    daily_target_pct=target, global_stop_equity=g_stop, global_target_equity=g_target,
                    max_trades_per_day=int(max_trades), fee_pct=fee, slippage_pct=slip,
                    ema_fast=int(ema_f), ema_slow=int(ema_s), rsi_period=int(rsi_p),
                    rsi_oversold=float(rsi_os), rsi_overbought=float(rsi_ob),
                    rsi_entry_min=float(rsi_lo), rsi_entry_max=float(rsi_hi),
                    bb_period=int(bb_p), bb_std=float(bb_k), stop_atr_mult=float(atr_m), reward_risk=float(rr),
                    session_tz=tz, session_start=s_start, no_entry_after=s_noent, flatten_at=s_flat)

    extra = {"mode": mode}
    if mode == "Backtest":
        extra["bars"] = sb.slider("Velas a reproducir", 200, 3000, 1000, 100)
    else:
        extra["refresh"] = sb.slider("Refresco (segundos)", 5, 120, 15)
    if mode == "Real":
        sb.subheader("🔐 Conexión real")
        sb.caption("Recomendado: guarda las claves en `.streamlit/secrets.toml` (BOT_API_KEY, BOT_API_SECRET, "
                   "BOT_API_PASSWORD). Usa claves SIN permiso de retirada.")
        extra["api_key"] = sb.text_input("API Key", _get_secret("BOT_API_KEY"), type="password")
        extra["secret"] = sb.text_input("API Secret", _get_secret("BOT_API_SECRET"), type="password")
        extra["password"] = sb.text_input("Passphrase (OKX/KuCoin/Bitget)", _get_secret("BOT_API_PASSWORD"), type="password")
        extra["sandbox"] = sb.checkbox("Usar TESTNET del exchange (dinero ficticio)", True)
        extra["ack"] = sb.checkbox("Entiendo que puedo perder dinero real")
        extra["confirm"] = sb.text_input("Escribe CONFIRMO para habilitar órdenes reales", "")
    return cfg, extra


def validate_cfg(cfg: BotConfig) -> list[str]:
    errs = []
    if cfg.ema_fast >= cfg.ema_slow:
        errs.append("La EMA rápida debe ser menor que la lenta.")
    if cfg.global_stop_equity >= cfg.initial_capital:
        errs.append("El stop global debe ser inferior al capital inicial.")
    if cfg.global_target_equity <= cfg.initial_capital:
        errs.append("La meta global debe ser superior al capital inicial.")
    try:
        _ = (_parse_hhmm(cfg.session_start), _parse_hhmm(cfg.no_entry_after), _parse_hhmm(cfg.flatten_at),
             pd.Timestamp.now(tz=cfg.session_tz))
    except Exception:
        errs.append("Revisa la zona horaria y las horas de sesión (formato HH:MM).")
    if cfg.source == "ccxt" and ccxt is None:
        errs.append("Falta ccxt: pip install ccxt")
    if cfg.source == "yfinance" and yf is None:
        errs.append("Falta yfinance: pip install yfinance")
    return errs


def render_dashboard(bot: Optional[TradingBot], ind: Optional[pd.DataFrame], cfg: BotConfig, mode: str):
    import streamlit as st
    ccy = cfg.quote_ccy()
    if bot is None:
        st.info("Configura el bot en el panel lateral y pulsa **▶️ Iniciar**.")
        if ind is not None:
            st.plotly_chart(build_chart(ind, None, cfg), use_container_width=True)
        return

    s = bot.stats()
    status = bot.status
    color = "green" if "ACTIVO" in status or "META" in status else ("orange" if "PAUSADO" in status else "red")
    st.markdown(f"### Estado: :{color}[{status}]  ·  {bot.broker.name}  ·  {cfg.symbol} {cfg.timeframe}")

    m = st.columns(6)
    m[0].metric("Capital actual", f"{s['equity']:.2f} {ccy}", f"{s['total']:+.2f} ({s['total'] / cfg.initial_capital * 100:+.2f}%)")
    m[1].metric("P/L flotante", f"{s['floating']:+.2f} {ccy}")
    m[2].metric("P/L del día", f"{s['day_pnl']:+.2f} {ccy}",
                f"Stop -{bot.day_start_equity * cfg.max_daily_loss_pct / 100:.2f} / Meta +{bot.day_start_equity * cfg.daily_target_pct / 100:.2f}",
                delta_color="off")
    m[3].metric("Ganadoras / Perdedoras", f"{s['wins']} / {s['losses']}", f"Acierto {s['win_rate']:.0f}%", delta_color="off")
    pf = "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    m[4].metric("Profit factor", pf, f"Máx. DD {s['max_dd']:.2f}%", delta_color="off")
    m[5].metric("Efectivo libre", f"{s['cash']:.2f} {ccy}", f"Comisiones {s['fees']:.2f}", delta_color="off")

    if bot.position:
        p = bot.position
        st.info(f"📌 Posición abierta: {_fmt(p.qty)} {cfg.base_ccy()} @ {_fmt(p.entry_price)} · SL {_fmt(p.stop)} · "
                f"TP {_fmt(p.target)} · Precio actual {_fmt(bot.last_price)}")

    if ind is not None:
        st.plotly_chart(build_chart(ind, bot, cfg), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("🧾 Log de decisiones")
        if bot.log:
            log_df = pd.DataFrame(bot.log[::-1])
            st.dataframe(log_df, use_container_width=True, height=360, hide_index=True,
                         column_config={"mensaje": st.column_config.TextColumn(width="large")})
            st.download_button("Descargar log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                               "log_decisiones.csv", "text/csv")
    with right:
        st.subheader("📊 Curva de capital")
        fig = build_equity_chart(bot, cfg)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("📋 Operaciones")
        if bot.trades:
            tdf = pd.DataFrame([asdict(t) for t in bot.trades])[
                ["entry_time", "exit_time", "entry_price", "exit_price", "qty", "pnl", "pnl_pct", "fees", "exit_reason"]]
            tdf = tdf.rename(columns={"entry_time": "entrada", "exit_time": "salida", "entry_price": "px entrada",
                                      "exit_price": "px salida", "qty": "cantidad", "pnl": f"P/L {ccy}",
                                      "pnl_pct": "P/L %", "fees": "comisiones", "exit_reason": "motivo salida"})
            st.dataframe(tdf.iloc[::-1].round(4), use_container_width=True, height=220, hide_index=True)
            st.download_button("Descargar operaciones (CSV)", tdf.to_csv(index=False).encode("utf-8"),
                               "operaciones.csv", "text/csv")
        else:
            st.caption("Aún no hay operaciones cerradas.")


def streamlit_main():
    import streamlit as st
    st.set_page_config(page_title="Bot de Trading Intradía", page_icon="📈", layout="wide")
    st.title("📈 Bot de Trading Intradía — Scalping EMA + RSI + Bollinger")
    st.caption("Software educativo. El trading conlleva riesgo de pérdida total del capital asignado.")

    cfg, extra = sidebar_config()
    mode = extra["mode"]
    ss = st.session_state
    if "bot" not in ss:
        ss["bot"] = None
    if "running" not in ss:
        ss["running"] = False
    if "ind" not in ss:
        ss["ind"] = None
    if "data_ex" not in ss:
        ss["data_ex"] = None

    errs = validate_cfg(cfg)
    for e in errs:
        st.sidebar.error(e)

    # ---------------- Botones de control ----------------
    b1, b2, b3, b4 = st.sidebar.columns(4)
    start = b1.button("▶️", help="Iniciar (aplica la configuración actual)", disabled=bool(errs), use_container_width=True)
    pause = b2.button("⏸️", help="Pausar / reanudar (la posición abierta se sigue gestionando)", use_container_width=True)
    stop = b3.button("⏹️", help="Detener y cerrar la posición", use_container_width=True)
    reset = b4.button("🔄", help="Reiniciar todo", use_container_width=True)
    load_saved = False
    if mode != "Backtest" and os.path.exists(state_path(mode, cfg)):
        load_saved = st.sidebar.checkbox("Recuperar estado guardado al iniciar", False,
                                         help=f"Archivo {state_path(mode, cfg)}")

    if reset:
        ss.bot, ss.running, ss.ind = None, False, None
        st.rerun()

    if start:
        try:
            if mode == "Backtest":
                with st.spinner("Descargando datos y ejecutando backtest..."):
                    df = fetch_data(cfg, extra["bars"])
                    ss.bot = run_backtest(cfg, df)
                    ss.ind = compute_indicators(df, cfg)
                    ss.running = False
            else:
                if mode == "Real":
                    if not (extra["api_key"] and extra["secret"]):
                        raise RuntimeError("Introduce la API Key y el Secret.")
                    if not extra["ack"] or extra["confirm"].strip().upper() != "CONFIRMO":
                        raise RuntimeError("Para operar con dinero real marca la casilla de riesgo y escribe CONFIRMO.")
                    broker = LiveCCXTBroker(cfg, extra["api_key"], extra["secret"], extra["password"], extra["sandbox"])
                    free = broker.free_balance(cfg.quote_ccy())
                    if free < cfg.initial_capital:
                        st.warning(f"Tu saldo libre ({free:.2f} {cfg.quote_ccy()}) es menor que el capital asignado. "
                                   f"Las compras pueden fallar por saldo insuficiente.")
                    data_ex = broker.ex  # mismos datos que el exchange donde se ejecuta
                else:
                    broker = PaperBroker(cfg)
                    data_ex = make_public_exchange(cfg.exchange_id) if cfg.source == "ccxt" else None
                bot = TradingBot(cfg, broker)
                df = fetch_data(cfg, 400, data_ex)
                closed, _ = split_closed_bars(df, cfg.timeframe)
                if load_saved:
                    with open(state_path(mode, cfg), encoding="utf-8") as f:
                        bot.load_dict(json.load(f))
                    bot._log(pd.Timestamp.now(tz="UTC"), "INFO", "Estado anterior recuperado desde disco.")
                bot.prime(closed)
                ss.bot, ss.data_ex, ss.ind = bot, data_ex, compute_indicators(df, cfg)
                ss.running, ss.mode = True, mode
        except Exception as e:
            st.error(f"No se pudo iniciar: {e}")

    bot: Optional[TradingBot] = ss.bot
    if pause and bot:
        bot.set_paused(not bot.paused)
    if stop and bot:
        bot.stop()
        ss.running = False
        if mode != "Backtest":
            save_state(bot, state_path(ss.get("mode", mode), bot.cfg))

    # ---------------- Ciclo en tiempo real (Paper / Real) ----------------
    if ss.running and bot:
        try:
            df = fetch_data(bot.cfg, 400, ss.data_ex)
            closed, live = split_closed_bars(df, bot.cfg.timeframe)
            ind = compute_indicators(closed, bot.cfg)
            bot.step(ind)  # señales solo con velas cerradas
            current = float(live["close"].iloc[-1]) if len(live) else float(closed["close"].iloc[-1])
            bot.check_price(current, pd.Timestamp.now(tz="UTC"))  # SL/TP en vivo
            ss.ind = compute_indicators(df, bot.cfg)
            save_state(bot, state_path(ss.get("mode", mode), bot.cfg))
            if bot.halted_global:
                ss.running = False
        except Exception as e:
            st.warning(f"Error de datos/conexión (se reintentará): {e}")

    if bot and bot.cfg != cfg and mode != "Backtest" and ss.running:
        st.sidebar.caption("ℹ️ Has cambiado la configuración: pulsa ▶️ para aplicarla (reinicia el bot).")

    render_dashboard(bot, ss.ind, bot.cfg if bot else cfg, mode)

    if ss.running:
        st.caption(f"🔁 Actualizando cada {extra.get('refresh', 15)} s · Última: {datetime.now():%H:%M:%S}. "
                   f"Mantén esta pestaña abierta (o usa el modo --headless).")
        time.sleep(extra.get("refresh", 15))
        st.rerun()


# =====================================================================
# 8. MODO SIN INTERFAZ (para dejarlo corriendo en un servidor / VPS)
# =====================================================================
def headless_main(argv=None):
    ap = argparse.ArgumentParser(description="Bot de trading intradía sin interfaz")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--config", help="JSON con cualquier campo de BotConfig")
    ap.add_argument("--symbol")
    ap.add_argument("--exchange")
    ap.add_argument("--timeframe")
    ap.add_argument("--capital", type=float)
    ap.add_argument("--refresh", type=int, default=15)
    ap.add_argument("--sandbox", action="store_true", help="Usar testnet del exchange")
    ap.add_argument("--resume", action="store_true", help="Recuperar estado guardado")
    ap.add_argument("--i-understand-real-money", action="store_true", dest="ack")
    a = ap.parse_args(argv)

    d = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            d = json.load(f)
    cfg = BotConfig.from_dict(d)
    if a.symbol: cfg.symbol = a.symbol
    if a.exchange: cfg.exchange_id = a.exchange
    if a.timeframe: cfg.timeframe = a.timeframe
    if a.capital:
        cfg.initial_capital = a.capital
        if "global_stop_equity" not in d: cfg.global_stop_equity = a.capital * 0.9
        if "global_target_equity" not in d: cfg.global_target_equity = a.capital * 1.2
    if "session_tz" not in d:
        for k, v in SESSION_PRESETS[cfg.source].items():
            setattr(cfg, k, v)
    errs = validate_cfg(cfg)
    if errs:
        sys.exit("Configuración inválida: " + " | ".join(errs))

    if a.mode == "live":
        if cfg.source != "ccxt":
            sys.exit("El modo real solo está disponible con CCXT (cripto).")
        if not a.ack:
            sys.exit("Para operar con dinero real añade --i-understand-real-money")
        broker = LiveCCXTBroker(cfg, os.environ.get("BOT_API_KEY", ""), os.environ.get("BOT_API_SECRET", ""),
                                os.environ.get("BOT_API_PASSWORD", ""), a.sandbox)
        data_ex = broker.ex
    else:
        broker = PaperBroker(cfg)
        data_ex = make_public_exchange(cfg.exchange_id) if cfg.source == "ccxt" else None

    bot = TradingBot(cfg, broker)
    path = state_path("Real" if a.mode == "live" else "Paper Trading", cfg)
    if a.resume and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            bot.load_dict(json.load(f))
    closed, _ = split_closed_bars(fetch_data(cfg, 400, data_ex), cfg.timeframe)
    bot.prime(closed)
    printed = 0
    print(f"Bot {broker.name} en marcha sobre {cfg.symbol} {cfg.timeframe}. Ctrl+C para detener.")
    try:
        while not bot.halted_global:
            try:
                df = fetch_data(cfg, 400, data_ex)
                closed, live = split_closed_bars(df, cfg.timeframe)
                bot.step(compute_indicators(closed, cfg))
                price = float(live["close"].iloc[-1]) if len(live) else float(closed["close"].iloc[-1])
                bot.check_price(price, pd.Timestamp.now(tz="UTC"))
                save_state(bot, path)
            except Exception as e:
                print(f"[ERROR] {e}")
            for entry in bot.log[printed:]:
                print(f"[{entry['hora']}] {entry['tipo']:8s} {entry['mensaje']}")
            printed = len(bot.log)
            s = bot.stats()
            print(f"  · Capital {s['equity']:.2f} | flotante {s['floating']:+.2f} | día {s['day_pnl']:+.2f} | "
                  f"{s['wins']}G/{s['losses']}P | {bot.status}", flush=True)
            time.sleep(a.refresh)
    except KeyboardInterrupt:
        print("\nDeteniendo y cerrando posiciones...")
        bot.stop()
        save_state(bot, path)
        for entry in bot.log[printed:]:
            print(f"[{entry['hora']}] {entry['tipo']:8s} {entry['mensaje']}")


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == "__main__":
    if "--headless" in sys.argv or not _running_in_streamlit():
        if "--headless" not in sys.argv:
            print("Para la interfaz ejecuta:  streamlit run app.py\n"
                  "Para el modo sin interfaz: python app.py --headless --mode paper --symbol BTC/USDT\n")
            sys.exit(0)
        headless_main()
    else:
        streamlit_main()
