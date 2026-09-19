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


def download_price_data_with_retry(ticker, period, max_retries=2):
    """PATCH 6: retry sederhana untuk mengatasi throttle Yahoo Finance."""
    for attempt in range(max_retries + 1):
        try:
            data = download_price_data(ticker, period)
            if not data.empty:
                return data
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    return pd.DataFrame()


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

    with np.errstate(divide="ignore", invalid="ignore"):
        df["volume_ratio"] = volume / df["volume_sma20"]
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan)

    df["high20_previous"] = high.rolling(20).max().shift(1)
    df["low10"] = low.rolling(10).min()
    df["high20"] = high.rolling(20).max()
    df["return_5d"] = close.pct_change(5) * 100
    df["drawdown_20d"] = (close / df["high20"] - 1) * 100
    df["value_sma20"] = (close * volume).rolling(20).mean()

    return df


def calc_momentum_risk_reward(momentum_entry, momentum_stop, momentum_target_5):
    """PATCH 2: guard ZeroDivisionError saat ATR=0 membuat entry == stop."""
    risk_denominator = momentum_entry - momentum_stop
    if risk_denominator > 0:
        momentum_risk = risk_denominator / momentum_entry * 100
        momentum_rr = (momentum_target_5 - momentum_entry) / risk_denominator
    else:
        momentum_risk = np.nan
        momentum_rr = 0
    return momentum_risk, momentum_rr


def analyze_stock(ticker, data, min_value_traded):
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
    atr_percent = atr_value / close * 100 if close > 0 else np.nan
    liquid = row["value_sma20"] >= min_value_traded

    # =========================
    # MOMENTUM / BREAKOUT
    # =========================
    momentum_entry = close
    momentum_stop = momentum_entry - 1.5 * atr_value
    momentum_target_5 = momentum_entry * 1.05
    momentum_target_10 = momentum_entry * 1.10

    momentum_risk, momentum_rr = calc_momentum_risk_reward(
        momentum_entry, momentum_stop, momentum_target_5
    )

    trend_bullish = close > row["ema20"] > row["ema50"]

    ema10_momentum = (
        close > row["ema10"]
        and row["ema10"] > row["ema20"]
    )

    breakout_20d = close > row["high20_previous"]
    volume_strong = pd.notna(row["volume_ratio"]) and row["volume_ratio"] >= 1.8

    rsi_healthy = 55 <= row["rsi14"] <= 72
    macd_bullish = (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    )

    not_extended = (
        row["return_5d"] <= 10
        and close <= row["ema10"] * 1.06
    )

    atr_suitable = pd.notna(atr_percent) and 1.0 <= atr_percent <= 5.0

    momentum_score = 0
    momentum_score += 20 if trend_bullish else 0
    momentum_score += 15 if ema10_momentum else 0
    momentum_score += 20 if breakout_20d else 0
    momentum_score += 15 if volume_strong else 0
    momentum_score += 10 if rsi_healthy and macd_bullish else 0
    momentum_score += 10 if liquid else 0
    momentum_score += 5 if atr_suitable else 0
    momentum_score += 5 if not_extended else 0

    momentum_hard_fail = (
        not liquid
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
        and not_extended
        and momentum_rr >= 1.5
    )

    # =========================
    # REBOUND
    # =========================
    rebound_entry = close
    rebound_stop = min(
        float(row["low10"]),
        rebound_entry - 1.5 * atr_value,
    )
    rebound_stop = max(0.0, rebound_stop)

    rebound_target_5 = rebound_entry * 1.05
    rebound_target_10 = rebound_entry * 1.10

    if rebound_stop < rebound_entry:
        rebound_risk = (
            (rebound_entry - rebound_stop)
            / rebound_entry
            * 100
        )
        rebound_rr = (
            (rebound_target_5 - rebound_entry)
            / (rebound_entry - rebound_stop)
        )
    else:
        rebound_risk = np.nan
        rebound_rr = 0

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

    rebound_volume = pd.notna(row["volume_ratio"]) and row["volume_ratio"] >= 1.3
    rebound_macd = row["macd_hist"] > float(prev["macd_hist"])
    rebound_atr_ok = pd.notna(atr_percent) and 1.0 <= atr_percent <= 6.0

    rebound_score = 0
    rebound_score += 20 if was_down else 0
    rebound_score += 20 if rsi_recovering else 0
    rebound_score += 15 if price_recovering else 0
    rebound_score += 15 if green_candle else 0
    rebound_score += 10 if rebound_volume else 0
    rebound_score += 10 if rebound_macd else 0
    rebound_score += 5 if liquid else 0
    rebound_score += 5 if rebound_atr_ok else 0

    rebound_hard_fail = (
        not liquid
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
        and rebound_rr >= 1.2
    )

    # =========================
    # STATUS HASIL
    # =========================
    if momentum_strong:
        status = "BELI KUAT - MOMENTUM"
        strategy = "Momentum / Breakout"
        score = momentum_score

        entry = momentum_entry
        stop_loss = momentum_stop
        target_5 = momentum_target_5
        target_10 = momentum_target_10
        risk_percent = momentum_risk
        reward_risk = momentum_rr

        reasons = [
            "tren EMA 20 > EMA 50",
            "EMA 10 di atas EMA 20",
            "breakout high 20 hari",
            f"volume {row['volume_ratio']:.1f}x",
            f"RSI {row['rsi14']:.0f}",
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
        risk_percent = rebound_risk
        reward_risk = rebound_rr

        reasons = [
            "harga sebelumnya turun",
            "RSI mulai pulih",
            "harga di atas EMA 5",
            "EMA 5 di atas EMA 10",
            "candle naik",
            f"volume {row['volume_ratio']:.1f}x",
        ]

    elif not momentum_hard_fail and momentum_score >= 60:
        status = "BELI"
        strategy = "Momentum belum lengkap"
        score = momentum_score

        entry = momentum_entry
        stop_loss = momentum_stop
        target_5 = momentum_target_5
        target_10 = momentum_target_10
        risk_percent = momentum_risk
        reward_risk = momentum_rr

        reasons = [
            "sebagian sinyal momentum positif",
            "belum memenuhi syarat BELI KUAT",
        ]

    elif not rebound_hard_fail and rebound_score >= 55:
        status = "BELI"
        strategy = "Rebound belum lengkap"
        score = rebound_score

        entry = rebound_entry
        stop_loss = rebound_stop
        target_5 = rebound_target_5
        target_10 = rebound_target_10
        risk_percent = rebound_risk
        reward_risk = rebound_rr

        reasons = [
            "ada tanda pantulan",
            "belum memenuhi syarat BELI KUAT REBOUND",
        ]

    else:
        status = "TIDAK LAYAK"
        strategy = "-"
        score = max(momentum_score, rebound_score)

        entry = close
        stop_loss = np.nan
        target_5 = close * 1.05
        target_10 = close * 1.10
        risk_percent = np.nan
        reward_risk = 0

        reasons = [
            "sinyal belum cukup kuat atau risiko terlalu tinggi"
        ]

    return {
        "Kode": ticker.replace(".JK", ""),
        "Status": status,
        "Strategi": strategy,
        "Skor": round(score),
        "Harga Terakhir": round(close),
        "Entry": round(entry),
        "Stop Loss": round(stop_loss) if pd.notna(stop_loss) else np.nan,
        "Risk %": round(risk_percent, 2) if pd.notna(risk_percent) else np.nan,
        "Target +5%": round(target_5),
        "Target +10%": round(target_10),
        "R:R Target 5%": round(reward_risk, 2),
        "RSI 14": round(float(row["rsi14"]), 1),
        "Volume Ratio": round(float(row["volume_ratio"]), 2) if pd.notna(row["volume_ratio"]) else np.nan,
        "Turun dari High 20H %": round(float(row["drawdown_20d"]), 2),
        "Nilai Transaksi 20H": round(float(row["value_sma20"])),
        "Alasan": "; ".join(reasons),
        "Tanggal Data": df.index[-1].strftime("%Y-%m-%d"),
    }


st.title("📈 IDX Swing Screener")

st.caption(
    "Screening saham BEI untuk swing 1–2 minggu. "
    "Hasil bersifat edukasi/ristek, bukan rekomendasi "
    "atau jaminan profit."
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
        value=10.0,
        step=1.0,
    )

    ticker_text = st.text_area(
        "Ticker pilihan (hanya digunakan bila memilih mode Ticker pilihan)",
        value="BBCA BBRI BMRI TLKM ANTM",
        height=120,
    )

    max_tickers = st.number_input(
        "Maksimum ticker diproses per sekali scan",
        min_value=50,
        max_value=1000,
        value=750,
        step=50,
    )

    run_screening = st.button(
        "Jalankan Screener",
        type="primary",
        use_container_width=True,
    )

st.info(
    f"Daftar aplikasi berisi {len(all_tickers)} ticker BEI. "
    "Momentum mencari breakout dengan konfirmasi EMA 10, EMA 20, "
    "EMA 50, dan volume; Rebound mencari pemulihan awal setelah turun."
)

if not run_screening:
    st.subheader("Cara menggunakan")

    st.markdown(
        "1. Pilih **Gabungan: Momentum + Rebound**.  \n"
        "2. Pilih **Semua saham BEI** atau masukkan ticker pilihan.  \n"
        "3. Tekan **Jalankan Screener** setelah pasar tutup.  \n"
        "4. Fokus pada **BELI KUAT - MOMENTUM** atau "
        "**BELI KUAT - REBOUND**.  \n"
        "5. Target +5% adalah target awal; gunakan Stop Loss yang tampil."
    )

    st.stop()

if scan_mode == "Semua saham BEI":
    tickers = all_tickers["ticker"].head(int(max_tickers)).tolist()
else:
    raw_tickers = ticker_text.replace(",", " ").replace("\n", " ").split()
    tickers = []

    for ticker in raw_tickers:
        ticker = ticker.strip().upper()

        if ticker:
            if ticker.endswith(".JK"):
                tickers.append(ticker)
            else:
                tickers.append(f"{ticker}.JK")

    tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.error("Tidak ada ticker untuk diproses.")
    st.stop()

minimum_value = min_value_billion * 1_000_000_000

results = []
failed = []

progress_bar = st.progress(
    0,
    text="Menyiapkan screening...",
)

for index, ticker in enumerate(tickers, start=1):
    try:
        price_data = download_price_data_with_retry(ticker, period)
        result = analyze_stock(
            ticker,
            price_data,
            minimum_value,
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
        "Coba lagi atau gunakan mode Ticker pilihan."
    )
    st.stop()

result_df = pd.DataFrame(results)

if strategy_filter == "Momentum / Breakout":
    result_df = result_df[
        result_df["Strategi"].isin(
            [
                "Momentum / Breakout",
                "Momentum belum lengkap",
                "-",
            ]
        )
    ]

elif strategy_filter == "Rebound dari bawah":
    result_df = result_df[
        result_df["Strategi"].isin(
            [
                "Rebound dari bawah",
                "Rebound belum lengkap",
                "-",
            ]
        )
    ]

status_order = {
    "BELI KUAT - MOMENTUM": 0,
    "BELI KUAT - REBOUND": 1,
    "BELI": 2,
    "TIDAK LAYAK": 3,
}

result_df["_status_order"] = result_df["Status"].map(status_order)

result_df = result_df.sort_values(
    by=["_status_order", "Skor", "Volume Ratio"],
    ascending=[True, False, False],
).drop(columns="_status_order")

result_df = result_df.head(10)

strong_momentum = result_df[
    result_df["Status"] == "BELI KUAT - MOMENTUM"
]

strong_rebound = result_df[
    result_df["Status"] == "BELI KUAT - REBOUND"
]

buy = result_df[
    result_df["Status"] == "BELI"
]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Saham Diproses", len(result_df))
c2.metric("BELI KUAT Momentum", len(strong_momentum))
c3.metric("BELI KUAT Rebound", len(strong_rebound))
c4.metric("BELI", len(buy))

st.subheader("Kandidat BELI KUAT")

strong_all = pd.concat([
    strong_momentum,
    strong_rebound,
])

if strong_all.empty:
    st.warning(
        "Belum ada BELI KUAT Momentum atau Rebound. "
        "Jangan memaksa membeli; jalankan lagi setelah pasar tutup berikutnya."
    )
else:
    st.dataframe(
        strong_all,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Hasil Screening")

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
    ],
)

st.dataframe(
    result_df[result_df["Status"].isin(selected_status)],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Unduh hasil CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name=f"hasil_idx_screener_{date.today().isoformat()}.csv",
    mime="text/csv",
)

if failed:
    st.caption(
        f"Tidak dapat diproses: {len(failed)} ticker. "
        "Ini dapat terjadi bila data belum tersedia atau ticker sudah tidak aktif."
    )

st.divider()

st.subheader("Aturan sederhana")

st.markdown(
    "- Prioritaskan hanya **BELI KUAT - MOMENTUM** atau "
    "**BELI KUAT - REBOUND**.  \n"
    "- Gunakan Entry, Stop Loss, dan Target +5% yang tampil.  \n"
    "- Setelah target +5% tercapai, pertimbangkan menjual sebagian "
    "atau menaikkan stop ke harga beli.  \n"
    "- Jangan membeli hanya karena harga sudah turun; tunggu "
    "konfirmasi BELI KUAT.  \n"
    "- Target bukan kepastian. Batasi risiko dan jangan memakai "
    "seluruh modal pada satu saham."
)
