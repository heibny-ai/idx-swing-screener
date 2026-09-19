import time
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange


st.set_page_config(
    page_title="IDX Swing Screener",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# DATA
# ============================================================

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

    return data.dropna(
        subset=["Open", "Close", "High", "Low", "Volume"]
    ).copy()


def download_price_data_with_retry(ticker, period, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            data = download_price_data(ticker, period)

            if not data.empty:
                return data

        except Exception:
            pass

        time.sleep(1.0 * (attempt + 1))

    return pd.DataFrame()


# ============================================================
# INDIKATOR
# ============================================================

def add_indicators(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend EMA
    df["ema5"] = EMAIndicator(close, window=5).ema_indicator()
    df["ema10"] = EMAIndicator(close, window=10).ema_indicator()
    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()

    # RSI
    df["rsi14"] = RSIIndicator(close, window=14).rsi()
    df["rsi14_prev"] = df["rsi14"].shift(1)

    # MACD
    macd = MACD(
        close,
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ADX: kekuatan tren
    adx = ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=14,
    )

    df["adx14"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ATR: volatilitas
    df["atr14"] = AverageTrueRange(
        high,
        low,
        close,
        window=14,
    ).average_true_range()

    # Volume
    df["volume_sma20"] = volume.rolling(20).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        df["volume_ratio"] = volume / df["volume_sma20"]

    df["volume_ratio"] = df["volume_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Nilai transaksi: Close x Volume
    df["value_today"] = close * volume
    df["value_sma20"] = df["value_today"].rolling(20).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        df["value_ratio"] = df["value_today"] / df["value_sma20"]

    df["value_ratio"] = df["value_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Support, breakout, resistance
    df["high20_previous"] = high.rolling(20).max().shift(1)
    df["high60_previous"] = high.rolling(60).max().shift(1)
    df["low10"] = low.rolling(10).min()
    df["high20"] = high.rolling(20).max()

    # Performa harga
    df["return_5d"] = close.pct_change(5) * 100
    df["drawdown_20d"] = (close / df["high20"] - 1) * 100

    return df


# ============================================================
# RISK, TARGET, DAN UKURAN POSISI
# ============================================================

def calc_risk_reward(entry, stop_loss, target):
    risk_per_share = entry - stop_loss
    reward_per_share = target - entry

    if risk_per_share <= 0:
        return np.nan, 0.0

    risk_percent = risk_per_share / entry * 100
    reward_risk = reward_per_share / risk_per_share

    return risk_percent, reward_risk


def calc_rr_targets(entry, stop_loss):
    risk_per_share = entry - stop_loss

    if risk_per_share <= 0:
        return np.nan, np.nan, np.nan

    target_1r = entry + (1.0 * risk_per_share)
    target_2r = entry + (2.0 * risk_per_share)
    target_3r = entry + (3.0 * risk_per_share)

    return target_1r, target_2r, target_3r


def calc_position_size(
    entry,
    stop_loss,
    capital,
    risk_percent_of_capital,
):
    risk_per_share = entry - stop_loss
    max_loss = capital * risk_percent_of_capital / 100

    if risk_per_share <= 0 or max_loss <= 0:
        return 0, 0, 0.0

    max_shares_by_risk = int(max_loss // risk_per_share)
    max_lots_by_risk = max_shares_by_risk // 100
    position_value = max_lots_by_risk * 100 * entry

    return max_lots_by_risk, max_shares_by_risk, position_value


def safe_round(value, decimals=2):
    if pd.isna(value) or not np.isfinite(value):
        return np.nan

    return round(float(value), decimals)


# ============================================================
# ANALISIS SAHAM
# ============================================================

def analyze_stock(
    ticker,
    data,
    min_value_traded,
    trading_capital,
    risk_per_trade_percent,
    min_price,
    atr_max_percent,
):
    if len(data) < 70:
        return None

    df = add_indicators(data)

    if len(df) < 2:
        return None

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
        "adx14",
        "adx_pos",
        "adx_neg",
        "atr14",
        "volume_ratio",
        "value_sma20",
        "value_ratio",
        "high20_previous",
        "high60_previous",
        "low10",
        "drawdown_20d",
        "return_5d",
    ]

    if row[required].isna().any():
        return None

    close = float(row["Close"])
    atr_value = float(row["atr14"])
    atr_percent = atr_value / close * 100 if close > 0 else np.nan

    volume_ratio = float(row["volume_ratio"])
    value_today = float(row["value_today"])
    value_sma20 = float(row["value_sma20"])
    value_ratio = float(row["value_ratio"])

    liquid = value_sma20 >= min_value_traded
    transaction_active = value_ratio >= 0.70
    price_valid = close >= min_price

    resistance_60 = float(row["high60_previous"])

    if resistance_60 > close:
        room_to_resistance = (
            (resistance_60 - close) / close * 100
        )
    else:
        room_to_resistance = 0.0

    room_for_target_5 = room_to_resistance >= 5.0

    # ========================================================
    # MOMENTUM / BREAKOUT
    # ========================================================

    momentum_entry = close
    momentum_stop = max(
        0.0,
        momentum_entry - (1.5 * atr_value),
    )

    momentum_target_5 = momentum_entry * 1.05
    momentum_target_10 = momentum_entry * 1.10

    momentum_risk, momentum_rr = calc_risk_reward(
        momentum_entry,
        momentum_stop,
        momentum_target_5,
    )

    (
        momentum_target_1r,
        momentum_target_2r,
        momentum_target_3r,
    ) = calc_rr_targets(
        momentum_entry,
        momentum_stop,
    )

    trend_bullish = close > row["ema20"] > row["ema50"]

    ema10_momentum = (
        close > row["ema10"]
        and row["ema10"] > row["ema20"]
    )

    breakout_20d = close > row["high20_previous"]

    volume_strong = volume_ratio >= 1.80

    rsi_healthy = 55 <= row["rsi14"] <= 72

    macd_bullish = (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    )

    adx_trending = (
        row["adx14"] >= 20
        and row["adx_pos"] > row["adx_neg"]
    )

    not_extended = (
        row["return_5d"] <= 10
        and close <= row["ema10"] * 1.06
    )

    atr_suitable = (
        pd.notna(atr_percent)
        and 1.0 <= atr_percent <= atr_max_percent
    )

    momentum_score = 0
    momentum_score += 20 if trend_bullish else 0
    momentum_score += 15 if ema10_momentum else 0
    momentum_score += 20 if breakout_20d else 0
    momentum_score += 15 if volume_strong else 0
    momentum_score += 10 if rsi_healthy and macd_bullish else 0
    momentum_score += 5 if adx_trending else 0
    momentum_score += 5 if liquid else 0
    momentum_score += 5 if transaction_active else 0
    momentum_score += 5 if atr_suitable else 0
    momentum_score += 5 if not_extended else 0
    momentum_score += 5 if room_for_target_5 else 0

    momentum_hard_fail = (
        not price_valid
        or not liquid
        or not trend_bullish
        or not np.isfinite(momentum_risk)
        or momentum_risk > 5
    )

    momentum_strong = (
        not momentum_hard_fail
        and momentum_score >= 80
        and ema10_momentum
        and breakout_20d
        and volume_strong
        and macd_bullish
        and not_extended
        and momentum_rr >= 1.50
    )

    # ========================================================
    # REBOUND
    # ========================================================

    rebound_entry = close

    # Stop loss yang lebih dekat antara low 10 hari dan 1.5 ATR.
    # Tetap berada di bawah harga entry.
    rebound_stop = max(
        float(row["low10"]),
        rebound_entry - (1.5 * atr_value),
    )

    rebound_stop = min(
        rebound_stop,
        rebound_entry - 0.01,
    )

    rebound_stop = max(0.0, rebound_stop)

    rebound_target_5 = rebound_entry * 1.05
    rebound_target_10 = rebound_entry * 1.10

    rebound_risk, rebound_rr = calc_risk_reward(
        rebound_entry,
        rebound_stop,
        rebound_target_5,
    )

    (
        rebound_target_1r,
        rebound_target_2r,
        rebound_target_3r,
    ) = calc_rr_targets(
        rebound_entry,
        rebound_stop,
    )

    was_down = (
        row["drawdown_20d"] <= -8
        or row["return_5d"] <= -5
    )

    rsi_recovering = (
        row["rsi14_prev"] < 35
        and row["rsi14"] >= 35
        and row["rsi14"] > row["rsi14_prev"]
    )

    price_recovering = (
        close > row["ema5"]
        and row["ema5"] > row["ema10"]
    )

    green_candle = (
        close > float(row["Open"])
        and close > float(prev["Close"])
    )

    rebound_volume = volume_ratio >= 1.30

    rebound_macd = (
        row["macd_hist"] > float(prev["macd_hist"])
    )

    rebound_atr_ok = (
        pd.notna(atr_percent)
        and 1.0 <= atr_percent <= atr_max_percent
    )

    rebound_score = 0
    rebound_score += 20 if was_down else 0
    rebound_score += 20 if rsi_recovering else 0
    rebound_score += 15 if price_recovering else 0
    rebound_score += 15 if green_candle else 0
    rebound_score += 10 if rebound_volume else 0
    rebound_score += 10 if rebound_macd else 0
    rebound_score += 5 if liquid else 0
    rebound_score += 5 if transaction_active else 0
    rebound_score += 5 if rebound_atr_ok else 0
    rebound_score += 5 if room_for_target_5 else 0

    rebound_hard_fail = (
        not price_valid
        or not liquid
        or not np.isfinite(rebound_risk)
        or rebound_risk > 6
    )

    rebound_strong = (
        not rebound_hard_fail
        and rebound_score >= 75
        and was_down
        and rsi_recovering
        and price_recovering
        and rebound_volume
        and rebound_rr >= 1.20
    )

    # ========================================================
    # STATUS DAN SETUP YANG DIPAKAI
    # Semua saham tetap memperoleh entry, SL, target, R:R,
    # dan alasan, termasuk saham TIDAK LAYAK.
    # ========================================================

    if momentum_strong:
        status = "BELI KUAT - MOMENTUM"
        strategy = "Momentum / Breakout"
        score = momentum_score

        entry = momentum_entry
        stop_loss = momentum_stop
        target_5 = momentum_target_5
        target_10 = momentum_target_10
        target_1r = momentum_target_1r
        target_2r = momentum_target_2r
        target_3r = momentum_target_3r
        risk_percent = momentum_risk
        reward_risk = momentum_rr

        reasons = [
            "tren bullish EMA 20 > EMA 50",
            "EMA 10 di atas EMA 20",
            "breakout high 20 hari",
            f"volume kuat {volume_ratio:.2f}x",
            f"ADX {row['adx14']:.1f}",
            "MACD bullish",
        ]

    elif rebound_strong:
        status = "BELI KUAT - REBOUND"
        strategy = "Rebound dari bawah"
        score = rebound_score

        entry = rebound_entry
        stop_loss = rebound_stop
        target_5 = rebound_target_5
        target_10 = rebound_target_10
        target_1r = rebound_target_1r
        target_2r = rebound_target_2r
        target_3r = rebound_target_3r
        risk_percent = rebound_risk
        reward_risk = rebound_rr

        reasons = [
            "harga sebelumnya terkoreksi",
            "RSI mulai pulih",
            "harga di atas EMA 5",
            "EMA 5 di atas EMA 10",
            "candle hijau konfirmasi",
            f"volume rebound {volume_ratio:.2f}x",
        ]

    elif not momentum_hard_fail and momentum_score >= 60:
        status = "BELI"
        strategy = "Momentum belum lengkap"
        score = momentum_score

        entry = momentum_entry
        stop_loss = momentum_stop
        target_5 = momentum_target_5
        target_10 = momentum_target_10
        target_1r = momentum_target_1r
        target_2r = momentum_target_2r
        target_3r = momentum_target_3r
        risk_percent = momentum_risk
        reward_risk = momentum_rr

        reasons = [
            "sebagian sinyal momentum positif",
            "belum memenuhi seluruh syarat beli kuat",
        ]

    elif not rebound_hard_fail and rebound_score >= 55:
        status = "BELI"
        strategy = "Rebound belum lengkap"
        score = rebound_score

        entry = rebound_entry
        stop_loss = rebound_stop
        target_5 = rebound_target_5
        target_10 = rebound_target_10
        target_1r = rebound_target_1r
        target_2r = rebound_target_2r
        target_3r = rebound_target_3r
        risk_percent = rebound_risk
        reward_risk = rebound_rr

        reasons = [
            "ada tanda pantulan",
            "belum memenuhi seluruh syarat rebound kuat",
        ]

    else:
        # Pilih setup dengan skor yang lebih baik agar indikator
        # risiko tetap muncul juga untuk saham TIDAK LAYAK.
        status = "TIDAK LAYAK"

        if momentum_score >= rebound_score:
            strategy = "Momentum tidak valid"
            score = momentum_score

            entry = momentum_entry
            stop_loss = momentum_stop
            target_5 = momentum_target_5
            target_10 = momentum_target_10
            target_1r = momentum_target_1r
            target_2r = momentum_target_2r
            target_3r = momentum_target_3r
            risk_percent = momentum_risk
            reward_risk = momentum_rr

            reasons = []

            if not price_valid:
                reasons.append(
                    f"harga Rp{close:.0f} di bawah minimum"
                )

            if not liquid:
                reasons.append(
                    "nilai transaksi rata-rata 20H terlalu kecil"
                )

            if not trend_bullish:
                reasons.append(
                    "trend EMA 20 dan EMA 50 belum bullish"
                )

            if not ema10_momentum:
                reasons.append(
                    "EMA 10 belum berada di atas EMA 20"
                )

            if not breakout_20d:
                reasons.append(
                    "belum breakout high 20 hari"
                )

            if not volume_strong:
                reasons.append(
                    f"volume lemah {volume_ratio:.2f}x "
                    "(perlu minimal 1.80x)"
                )

            if not rsi_healthy:
                reasons.append(
                    f"RSI belum ideal {row['rsi14']:.1f}"
                )

            if not macd_bullish:
                reasons.append("MACD belum bullish")

            if not adx_trending:
                reasons.append(
                    f"ADX belum mendukung tren {row['adx14']:.1f}"
                )

            if not transaction_active:
                reasons.append(
                    f"transaksi hari ini lemah {value_ratio:.2f}x"
                )

            if not atr_suitable:
                reasons.append(
                    f"ATR {atr_percent:.2f}% di luar batas"
                )

            if not room_for_target_5:
                reasons.append(
                    f"ruang ke resistance hanya "
                    f"{room_to_resistance:.2f}%"
                )

            if not np.isfinite(momentum_risk):
                reasons.append("stop loss tidak valid")

            elif momentum_risk > 5:
                reasons.append(
                    f"risiko {momentum_risk:.2f}% terlalu besar"
                )

            if momentum_rr < 1.5:
                reasons.append(
                    f"R:R target +5% rendah {momentum_rr:.2f}"
                )

        else:
            strategy = "Rebound tidak valid"
            score = rebound_score

            entry = rebound_entry
            stop_loss = rebound_stop
            target_5 = rebound_target_5
            target_10 = rebound_target_10
            target_1r = rebound_target_1r
            target_2r = rebound_target_2r
            target_3r = rebound_target_3r
            risk_percent = rebound_risk
            reward_risk = rebound_rr

            reasons = []

            if not price_valid:
                reasons.append(
                    f"harga Rp{close:.0f} di bawah minimum"
                )

            if not liquid:
                reasons.append(
                    "nilai transaksi rata-rata 20H terlalu kecil"
                )

            if not was_down:
                reasons.append(
                    "belum ada koreksi harga yang cukup"
                )

            if not rsi_recovering:
                reasons.append(
                    "RSI belum pulih dari area rendah"
                )

            if not price_recovering:
                reasons.append(
                    "harga dan EMA belum mengonfirmasi rebound"
                )

            if not green_candle:
                reasons.append(
                    "belum ada candle hijau konfirmasi"
                )

            if not rebound_volume:
                reasons.append(
                    f"volume rebound lemah {volume_ratio:.2f}x "
                    "(perlu minimal 1.30x)"
                )

            if not rebound_macd:
                reasons.append(
                    "histogram MACD belum membaik"
                )

            if not transaction_active:
                reasons.append(
                    f"transaksi hari ini lemah {value_ratio:.2f}x"
                )

            if not rebound_atr_ok:
                reasons.append(
                    f"ATR {atr_percent:.2f}% di luar batas"
                )

            if not room_for_target_5:
                reasons.append(
                    f"ruang ke resistance hanya "
                    f"{room_to_resistance:.2f}%"
                )

            if not np.isfinite(rebound_risk):
                reasons.append("stop loss tidak valid")

            elif rebound_risk > 6:
                reasons.append(
                    f"risiko {rebound_risk:.2f}% terlalu besar"
                )

            if rebound_rr < 1.2:
                reasons.append(
                    f"R:R target +5% rendah {rebound_rr:.2f}"
                )

        if not reasons:
            reasons = ["syarat strategi belum lengkap"]

    # Position size dihitung sesudah entry dan stop final dipilih.
    max_lots, max_shares, position_value = calc_position_size(
        entry=entry,
        stop_loss=stop_loss,
        capital=trading_capital,
        risk_percent_of_capital=risk_per_trade_percent,
    )

    return {
        # Identitas dan keputusan
        "Kode": ticker.replace(".JK", ""),
        "Status": status,
        "Strategi": strategy,
        "Skor": int(round(score)),
        "Alasan": "; ".join(reasons),
        "Tanggal Data": df.index[-1].strftime("%Y-%m-%d"),

        # Harga, stop, target
        "Harga Terakhir": safe_round(close, 0),
        "Entry": safe_round(entry, 0),
        "Stop Loss": safe_round(stop_loss, 0),
        "Risk %": safe_round(risk_percent, 2),
        "Target +5%": safe_round(target_5, 0),
        "Target +10%": safe_round(target_10, 0),
        "Target 1R": safe_round(target_1r, 0),
        "Target 2R": safe_round(target_2r, 0),
        "Target 3R": safe_round(target_3r, 0),
        "R:R Target 5%": safe_round(reward_risk, 2),

        # Ukuran posisi
        "Modal Trading": safe_round(trading_capital, 0),
        "Risk Modal %": safe_round(risk_per_trade_percent, 2),
        "Risk Modal Rp": safe_round(
            trading_capital * risk_per_trade_percent / 100,
            0,
        ),
        "Maksimal Lot": int(max_lots),
        "Maksimal Saham": int(max_shares),
        "Nilai Posisi Maks": safe_round(position_value, 0),

        # Momentum dan tren
        "RSI 14": safe_round(row["rsi14"], 1),
        "RSI 14 Sebelum": safe_round(row["rsi14_prev"], 1),
        "EMA 5": safe_round(row["ema5"], 2),
        "EMA 10": safe_round(row["ema10"], 2),
        "EMA 20": safe_round(row["ema20"], 2),
        "EMA 50": safe_round(row["ema50"], 2),
        "MACD": safe_round(row["macd"], 4),
        "MACD Signal": safe_round(row["macd_signal"], 4),
        "MACD Hist": safe_round(row["macd_hist"], 4),
        "ADX 14": safe_round(row["adx14"], 1),
        "+DI": safe_round(row["adx_pos"], 1),
        "-DI": safe_round(row["adx_neg"], 1),

        # Volume, transaksi, volatilitas
        "Volume Ratio": safe_round(volume_ratio, 2),
        "Nilai Transaksi Hari Ini": safe_round(value_today, 0),
        "Nilai Transaksi 20H": safe_round(value_sma20, 0),
        "Rasio Nilai Transaksi": safe_round(value_ratio, 2),
        "ATR 14": safe_round(atr_value, 2),
        "ATR %": safe_round(atr_percent, 2),

        # Struktur harga
        "Return 5H %": safe_round(row["return_5d"], 2),
        "Turun dari High 20H %": safe_round(
            row["drawdown_20d"],
            2,
        ),
        "High 20H Sebelumnya": safe_round(
            row["high20_previous"],
            0,
        ),
        "Low 10H": safe_round(row["low10"], 0),
        "Resistance 60H": safe_round(resistance_60, 0),
        "Ruang ke Resistance %": safe_round(
            room_to_resistance,
            2,
        ),

        # Kondisi True / False untuk audit sinyal
        "Likuid": bool(liquid),
        "Transaksi Aktif": bool(transaction_active),
        "Harga Valid": bool(price_valid),
        "Trend Bullish": bool(trend_bullish),
        "EMA Momentum": bool(ema10_momentum),
        "Breakout 20H": bool(breakout_20d),
        "Volume Kuat": bool(volume_strong),
        "RSI Sehat": bool(rsi_healthy),
        "MACD Bullish": bool(macd_bullish),
        "ADX Trending": bool(adx_trending),
        "ATR Sesuai": bool(atr_suitable),
        "Ruang Target +5%": bool(room_for_target_5),
    }


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("📈 IDX Swing Screener")

st.caption(
    "Screening saham BEI untuk swing 1–2 minggu. "
    "Hasil adalah alat edukasi dan penyaringan teknikal, "
    "bukan rekomendasi beli maupun jaminan profit."
)

all_tickers = load_all_idx_tickers()

with st.sidebar:
    st.header("Pengaturan")

    scan_mode = st.radio(
        "Universe screening",
        ["Semua saham BEI", "Ticker pilihan"],
        index=0,
    )

    strategy_filter = st.radio(
        "Strategi yang dicari",
        [
            "Gabungan: Momentum + Rebound",
            "Momentum / Breakout",
            "Rebound dari bawah",
        ],
        index=0,
    )

    period = st.selectbox(
        "Periode data",
        ["1y", "2y", "5y"],
        index=1,
    )

    min_value_billion = st.number_input(
        "Minimal nilai transaksi rata-rata 20 hari (Rp miliar)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help=(
            "Untuk saham lapis, Rp3–5 miliar adalah minimum longgar. "
            "Rp5–10 miliar lebih nyaman untuk swing. "
            "Nilai besar tidak menggantikan pemeriksaan order book."
        ),
    )

    min_price = st.number_input(
        "Harga saham minimum (Rp)",
        min_value=50,
        value=50,
        step=10,
    )

    atr_max_percent = st.slider(
        "ATR% maksimum",
        min_value=3.0,
        max_value=12.0,
        value=8.0,
        step=0.5,
        help=(
            "Untuk saham lapis yang agresif gunakan sekitar 8%. "
            "ATR tinggi berarti lot harus lebih kecil."
        ),
    )

    trading_capital = st.number_input(
        "Modal trading (Rp)",
        min_value=100_000,
        value=10_000_000,
        step=500_000,
    )

    risk_per_trade_percent = st.slider(
        "Risiko maksimal per transaksi (% dari modal)",
        min_value=0.25,
        max_value=3.00,
        value=1.00,
        step=0.25,
    )

    candidate_limit = st.slider(
        "Maksimal kandidat BELI/BELI KUAT utama",
        min_value=10,
        max_value=15,
        value=15,
        step=1,
        help=(
            "Hanya membatasi tabel kandidat utama. "
            "Tabel lengkap dan CSV tetap menampilkan seluruh saham."
        ),
    )

    max_tickers = st.number_input(
        "Maksimum ticker diproses per sekali scan",
        min_value=50,
        max_value=1000,
        value=750,
        step=50,
    )

    ticker_text = st.text_area(
        "Ticker pilihan (gunakan bila memilih mode Ticker pilihan)",
        value="BBCA BBRI BMRI TLKM ANTM",
        height=120,
    )

    run_screening = st.button(
        "Jalankan Screener",
        type="primary",
        use_container_width=True,
    )

st.info(
    f"Daftar aplikasi berisi {len(all_tickers)} ticker BEI. "
    "Momentum mencari breakout dengan tren, volume, ADX, dan "
    "risk/reward; rebound mencari pemulihan awal setelah koreksi."
)

if not run_screening:
    st.subheader("Cara menggunakan")

    st.markdown(
        "1. Atur nilai transaksi minimal; untuk saham lapis mulai dari "
        "**Rp5 miliar**.  \n"
        "2. Tetapkan modal dan risiko transaksi, misalnya modal "
        "**Rp10 juta** dan risiko **1%**.  \n"
        "3. Tekan **Jalankan Screener** setelah pasar tutup.  \n"
        "4. Fokus pada maksimal **10–15** kandidat status "
        "**BELI KUAT** atau **BELI**.  \n"
        "5. Sebelum entry, cek chart dan order book di aplikasi broker."
    )

    st.stop()


# ============================================================
# TENTUKAN TICKER
# ============================================================

if scan_mode == "Semua saham BEI":
    tickers = all_tickers["ticker"].head(
        int(max_tickers)
    ).tolist()

else:
    raw_tickers = (
        ticker_text.replace(",", " ")
        .replace("\n", " ")
        .split()
    )

    tickers = []

    for ticker in raw_tickers:
        ticker = ticker.strip().upper()

        if not ticker:
            continue

        if ticker.endswith(".JK"):
            tickers.append(ticker)
        else:
            tickers.append(f"{ticker}.JK")

    tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.error("Tidak ada ticker untuk diproses.")
    st.stop()

minimum_value = min_value_billion * 1_000_000_000


# ============================================================
# JALANKAN SCREENING
# ============================================================

results = []
failed = []

progress_bar = st.progress(
    0,
    text="Menyiapkan screening...",
)

for index, ticker in enumerate(tickers, start=1):
    try:
        price_data = download_price_data_with_retry(
            ticker,
            period,
        )

        result = analyze_stock(
            ticker=ticker,
            data=price_data,
            min_value_traded=minimum_value,
            trading_capital=trading_capital,
            risk_per_trade_percent=risk_per_trade_percent,
            min_price=min_price,
            atr_max_percent=atr_max_percent,
        )

        if result is not None:
            results.append(result)
        else:
            failed.append(ticker)

    except Exception:
        failed.append(ticker)

    progress_bar.progress(
        index / len(tickers),
        text=f"Memproses {index}/{len(tickers)}: {ticker}",
    )

    time.sleep(0.15)

progress_bar.empty()

if not results:
    st.error(
        "Data tidak berhasil diproses. "
        "Coba lagi, kurangi jumlah ticker, atau gunakan Ticker pilihan."
    )
    st.stop()


# ============================================================
# FILTER STRATEGI DAN URUTKAN SEMUA HASIL
# ============================================================

all_result_df = pd.DataFrame(results)

if strategy_filter == "Momentum / Breakout":
    all_result_df = all_result_df[
        all_result_df["Strategi"].isin(
            [
                "Momentum / Breakout",
                "Momentum belum lengkap",
                "Momentum tidak valid",
            ]
        )
    ].copy()

elif strategy_filter == "Rebound dari bawah":
    all_result_df = all_result_df[
        all_result_df["Strategi"].isin(
            [
                "Rebound dari bawah",
                "Rebound belum lengkap",
                "Rebound tidak valid",
            ]
        )
    ].copy()

status_order = {
    "BELI KUAT - MOMENTUM": 0,
    "BELI KUAT - REBOUND": 1,
    "BELI": 2,
    "TIDAK LAYAK": 3,
}

all_result_df["_status_order"] = all_result_df[
    "Status"
].map(status_order)

all_result_df = all_result_df.sort_values(
    by=[
        "_status_order",
        "Skor",
        "R:R Target 5%",
        "Volume Ratio",
        "Nilai Transaksi 20H",
    ],
    ascending=[True, False, False, False, False],
).drop(columns="_status_order")


# ============================================================
# METRIK HASIL PENUH
# ============================================================

strong_momentum = all_result_df[
    all_result_df["Status"] == "BELI KUAT - MOMENTUM"
]

strong_rebound = all_result_df[
    all_result_df["Status"] == "BELI KUAT - REBOUND"
]

buy = all_result_df[
    all_result_df["Status"] == "BELI"
]

not_eligible = all_result_df[
    all_result_df["Status"] == "TIDAK LAYAK"
]

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Saham Dianalisis", len(all_result_df))
c2.metric("BELI KUAT Momentum", len(strong_momentum))
c3.metric("BELI KUAT Rebound", len(strong_rebound))
c4.metric("BELI", len(buy))
c5.metric("Tidak Layak", len(not_eligible))


# ============================================================
# 10–15 KANDIDAT UTAMA SAJA
# ============================================================

st.subheader(
    f"Top {candidate_limit} Kandidat BELI KUAT dan BELI"
)

candidates_df = all_result_df[
    all_result_df["Status"].isin(
        [
            "BELI KUAT - MOMENTUM",
            "BELI KUAT - REBOUND",
            "BELI",
        ]
    )
].copy()

candidate_columns = [
    "Kode",
    "Status",
    "Strategi",
    "Skor",
    "Harga Terakhir",
    "Entry",
    "Stop Loss",
    "Risk %",
    "Target 1R",
    "Target 2R",
    "Target +5%",
    "Target +10%",
    "R:R Target 5%",
    "Maksimal Lot",
    "Nilai Posisi Maks",
    "RSI 14",
    "ADX 14",
    "Volume Ratio",
    "ATR %",
    "Nilai Transaksi 20H",
    "Alasan",
]

if candidates_df.empty:
    st.warning(
        "Belum ada kandidat BELI KUAT atau BELI yang lolos. "
        "Jangan memaksa entry; periksa lagi setelah penutupan pasar "
        "berikutnya."
    )

else:
    st.dataframe(
        candidates_df.head(int(candidate_limit))[
            candidate_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Maksimal lot dihitung dari modal dan risk per transaksi Anda. "
        "Tetap cek order book, spread bid–ask, dan chart sebelum entry."
    )


# ============================================================
# TABEL SEMUA HASIL, TERMASUK TIDAK LAYAK
# ============================================================

st.subheader("Semua Hasil dan Indikator")

selected_status = st.multiselect(
    "Status yang ditampilkan",
    [
        "BELI KUAT - MOMENTUM",
        "BELI KUAT - REBOUND",
        "BELI",
        "TIDAK LAYAK",
    ],
    default=[
        "BELI KUAT - MOMENTUM",
        "BELI KUAT - REBOUND",
        "BELI",
        "TIDAK LAYAK",
    ],
)

display_df = all_result_df[
    all_result_df["Status"].isin(selected_status)
].copy()

summary_columns = [
    "Kode",
    "Status",
    "Strategi",
    "Skor",
    "Harga Terakhir",
    "Entry",
    "Stop Loss",
    "Risk %",
    "R:R Target 5%",
    "RSI 14",
    "ADX 14",
    "Volume Ratio",
    "ATR %",
    "Nilai Transaksi 20H",
    "Rasio Nilai Transaksi",
    "Ruang ke Resistance %",
    "Alasan",
]

st.caption(
    f"Menampilkan {len(display_df)} saham. "
    "Saham TIDAK LAYAK tetap tampil agar indikator dan alasannya "
    "bisa dipelajari."
)

st.dataframe(
    display_df[summary_columns],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INDIKATOR DETAIL
# ============================================================

detail_columns = [
    "Kode",
    "Status",
    "Strategi",
    "Skor",
    "Tanggal Data",
    "Harga Terakhir",
    "Entry",
    "Stop Loss",
    "Risk %",
    "Target +5%",
    "Target +10%",
    "Target 1R",
    "Target 2R",
    "Target 3R",
    "R:R Target 5%",
    "Modal Trading",
    "Risk Modal %",
    "Risk Modal Rp",
    "Maksimal Lot",
    "Maksimal Saham",
    "Nilai Posisi Maks",
    "RSI 14",
    "RSI 14 Sebelum",
    "EMA 5",
    "EMA 10",
    "EMA 20",
    "EMA 50",
    "MACD",
    "MACD Signal",
    "MACD Hist",
    "ADX 14",
    "+DI",
    "-DI",
    "Volume Ratio",
    "Nilai Transaksi Hari Ini",
    "Nilai Transaksi 20H",
    "Rasio Nilai Transaksi",
    "ATR 14",
    "ATR %",
    "Return 5H %",
    "Turun dari High 20H %",
    "High 20H Sebelumnya",
    "Low 10H",
    "Resistance 60H",
    "Ruang ke Resistance %",
    "Likuid",
    "Transaksi Aktif",
    "Harga Valid",
    "Trend Bullish",
    "EMA Momentum",
    "Breakout 20H",
    "Volume Kuat",
    "RSI Sehat",
    "MACD Bullish",
    "ADX Trending",
    "ATR Sesuai",
    "Ruang Target +5%",
    "Alasan",
]

with st.expander(
    "Lihat seluruh indikator untuk semua saham",
    expanded=False,
):
    st.dataframe(
        display_df[detail_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CSV LENGKAP
# ============================================================

st.download_button(
    "Unduh seluruh hasil CSV",
    data=all_result_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=(
        f"hasil_idx_screener_"
        f"{date.today().isoformat()}.csv"
    ),
    mime="text/csv",
)

if failed:
    st.caption(
        f"Tidak dapat diproses: {len(failed)} ticker. "
        "Penyebabnya dapat berupa data Yahoo Finance tidak tersedia, "
        "ticker tidak aktif, atau data historis belum cukup."
    )


# ============================================================
# CATATAN PENGGUNAAN
# ============================================================

st.divider()

st.subheader("Aturan penggunaan")

st.markdown(
    "- Prioritaskan maksimal **10–15 kandidat** pada tabel utama.  \n"
    "- Status **BELI KUAT** adalah prioritas; status **BELI** perlu "
    "konfirmasi lebih ketat.  \n"
    "- Status **TIDAK LAYAK** bukan berarti saham buruk selamanya; "
    "baca kolom **Alasan** untuk mengetahui syarat yang belum terpenuhi.  \n"
    "- Gunakan kolom **Maksimal Lot**, bukan membeli jumlah lot sama "
    "pada setiap saham.  \n"
    "- Stop loss adalah batas risiko: jika levelnya tersentuh, "
    "jalankan rencana exit sesuai gaya trading Anda.  \n"
    "- Sebelum entry, cek order book secara manual di aplikasi broker: "
    "spread bid–ask, ketebalan bid, offer di atas harga entry, "
    "dan aktivitas transaksi hari itu.  \n"
    "- Screener adalah alat penyaring teknikal, bukan jaminan harga naik."
)
