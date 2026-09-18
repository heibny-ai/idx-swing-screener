import time
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="IDX Swing Screener", page_icon="📈", layout="wide")


@st.cache_data
def load_all_idx_tickers():
    tickers = pd.read_csv("tickers_idx.csv")
    tickers["ticker"] = tickers["ticker"].astype(str).str.upper().str.strip()
    tickers = tickers[tickers["ticker"].str.endswith(".JK")]
    return tickers.drop_duplicates(subset=["ticker"])


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(ticker, period):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(subset=["Close", "High", "Low", "Volume"]).copy()


def add_indicators(data):
    df = data.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["ema5"] = EMAIndicator(close, window=5).ema_indicator()
    df["ema10"] = EMAIndicator(close, window=10).ema_indicator()
    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()

    df["rsi14"] = RSIIndicator(close, window=14).rsi()
    df["rsi14_prev"] = df["rsi14"].shift(1)

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["atr14"] = AverageTrueRange(
        high,
        low,
        close,
        window=14,
    ).average_true_range()

    df["volume_sma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma20"]
    df["high20_previous"] = high.rolling(20).max().shift(1)
    df["low10"] = low.rolling(10).min()
    df["high20"] = high.rolling(20).max()
    df["return_5d"] = close.pct_change(5) * 100
    df["drawdown_20d"] = (close / df["high20"] - 1) * 100
    df["value_sma20"] = (close * volume).rolling(20).mean()

    return df


def analyze_stock(ticker, data, min_value_traded):
    if len(data) < 70:
        return None

    df = add_indicators(data)
    row = df.iloc[-1]
    prev = df.iloc[-2]

    required = [
        "ema5",
        "ema10",
        "ema20",
        "ema50",
        "rsi14",
        "rsi14_prev",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr14",
        "volume_ratio",
        "high20_previous",
        "low10",
        "drawdown_20d",
        "value_sma20",
    ]

    if row[required].isna().any():
        return None

    close = float(row["Close"])
    atr_value = float(row["atr14"])
    atr_percent = atr_value / close * 100
    liquid = row["value_sma20"] >= min_value_traded

    # =========================
    # MOMENTUM / BREAKOUT
    # =========================
    momentum_entry = close
    momentum_stop = momentum_entry - 1.5 * atr_value
    momentum_target_5 = momentum_entry * 1.05
    momentum_target_10 = momentum_entry * 1.10

    momentum_risk = (
        (momentum_entry - momentum_stop)
        / momentum_entry
        * 100
    )

    momentum_rr = (
        (momentum_target_5 - momentum_entry)
        / (momentum_entry - momentum_stop)
    )

    # Filter tren utama: EMA 20 dan EMA 50.
    trend_bullish = close > row["ema20"] > row["ema50"]

    # Konfirmasi momentum
