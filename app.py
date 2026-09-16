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
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()
    df["rsi14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["atr14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["volume_sma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma20"]
    df["high20_previous"] = high.rolling(20).max().shift(1)
    df["return_5d"] = close.pct_change(5) * 100
    df["value_sma20"] = (close * volume).rolling(20).mean()
    return df


def analyze_stock(ticker, data, min_value_traded):
    if len(data) < 70:
        return None

    df = add_indicators(data)
    row = df.iloc[-1]
    required = [
        "ema20", "ema50", "rsi14", "macd", "macd_signal", "atr14",
        "volume_ratio", "high20_previous", "value_sma20",
    ]
    if row[required].isna().any():
        return None

    close = float(row["Close"])
    atr_value = float(row["atr14"])
    entry = close
    stop_loss = entry - (1.5 * atr_value)
    target_5 = entry * 1.05
    target_10 = entry * 1.10
    risk_percent = (entry - stop_loss) / entry * 100
    reward_risk_5 = (target_5 - entry) / (entry - stop_loss)
    atr_percent = atr_value / close * 100

    trend_bullish = close > row["ema20"] > row["ema50"]
    breakout_20d = close > row["high20_previous"]
    volume_strong = row["volume_ratio"] >= 1.8
    rsi_healthy = 55 <= row["rsi14"] <= 72
    macd_bullish = row["macd"] > row["macd_signal"] and row["macd_hist"] > 0
    liquid = row["value_sma20"] >= min_value_traded
    not_extended = row["return_5d"] <= 15 and close <= row["ema20"] * 1.08
    atr_suitable = 1.0 <= atr_percent <= 8.0

    score = 0
    score += 20 if trend_bullish else 0
    score += 20 if breakout_20d else 0
    score += 15 if volume_strong else 0
    score += 15 if rsi_healthy and macd_bullish else 0
    score += 10 if liquid else 0
    score += 10 if atr_suitable else 0
    score += 5 if not_extended else 0
    score += 5 if reward_risk_5 >= 1.5 else 0

    hard_fail = not liquid or not trend_bullish or risk_percent > 7
    if not hard_fail and score >= 80 and breakout_20d and volume_strong and reward_risk_5 >= 1.5:
        status = "BELI KUAT"
    elif not hard_fail and score >= 60:
        status = "BELI"
    else:
        status = "TIDAK LAYAK"

    reasons = []
    if trend_bullish:
        reasons.append("tren EMA bullish")
    if breakout_20d:
        reasons.append("breakout high 20 hari")
    if volume_strong:
        reasons.append(f"volume {row['volume_ratio']:.1f}x")
    if rsi_healthy:
        reasons.append(f"RSI {row['rsi14']:.0f}")
    if macd_bullish:
        reasons.append("MACD bullish")
    if not liquid:
        reasons.append("likuiditas kurang")
    if not not_extended:
        reasons.append("sudah extended")

    return {
        "Kode": ticker.replace(".JK", ""),
        "Status": status,
        "Skor": round(score),
        "Harga Terakhir": round(close),
        "Entry": round(entry),
        "Stop Loss": round(stop_loss),
        "Risk %": round(risk_percent, 2),
        "Target +5%": round(target_5),
        "Target +10%": round(target_10),
        "R:R Target 5%": round(reward_risk_5, 2),
        "RSI 14": round(float(row["rsi14"]), 1),
        "Volume Ratio": round(float(row["volume_ratio"]), 2),
        "ATR %": round(atr_percent, 2),
        "Return 5 Hari %": round(float(row["return_5d"]), 2),
        "Nilai Transaksi 20H": round(float(row["value_sma20"])),
        "Alasan": "; ".join(reasons) or "Sinyal belum cukup kuat",
        "Tanggal Data": df.index[-1].strftime("%Y-%m-%d"),
    }


st.title("📈 IDX Swing Screener")
st.caption("Screening saham BEI untuk horizon 1–5 hari. Hasil bersifat edukasi/ristek, bukan rekomendasi atau jaminan profit.")

all_tickers = load_all_idx_tickers()

with st.sidebar:
    st.header("Pengaturan")
    scan_mode = st.radio("Universe screening", ["Semua saham BEI", "Ticker pilihan"], index=0)
    period = st.selectbox("Periode data", ["1y", "2y", "5y"], index=1)
    min_value_billion = st.number_input(
        "Minimal nilai transaksi rata-rata 20 hari (Rp miliar)",
        min_value=0.0,
        value=5.0,
        step=1.0,
    )
    ticker_text = st.text_area(
        "Ticker pilihan (hanya digunakan bila memilih mode Ticker pilihan)",
        value="BBCA BBRI BMRI TLKM ANTM",
        height=130,
    )
    max_tickers = st.number_input(
        "Maksimum ticker diproses per sekali scan",
        min_value=50,
        max_value=1000,
        value=750,
        step=50,
        help="Gunakan 750 agar hampir seluruh daftar dipindai. Proses dapat memerlukan beberapa menit.",
    )
    run_screening = st.button("Jalankan Screener", type="primary", use_container_width=True)

st.info(
    f"Daftar aplikasi berisi {len(all_tickers)} ticker BEI. Mode Semua saham BEI akan memproses daftar tersebut, "
    "lalu memfilter likuiditas, tren, breakout, volume, momentum, dan risiko."
)

if not run_screening:
    st.subheader("Cara menggunakan")
    st.markdown(
        "1. Pilih **Semua saham BEI** untuk scan lengkap.  \n"
        "2. Tekan **Jalankan Screener** setelah pasar tutup.  \n"
        "3. Tunggu proses; hasil diurutkan dari **BELI KUAT**.  \n"
        "4. Gunakan kandidat sebagai shortlist, lalu cek chart dan kondisi pasar."
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
            tickers.append(ticker if ticker.endswith(".JK") else f"{ticker}.JK")
    tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.error("Tidak ada ticker untuk diproses.")
    st.stop()

minimum_value = min_value_billion * 1_000_000_000
results, failed = [], []
progress_bar = st.progress(0, text="Menyiapkan screening...")

for index, ticker in enumerate(tickers, start=1):
    try:
        price_data = download_price_data(ticker, period)
        result = analyze_stock(ticker, price_data, minimum_value)
        if result is not None:
            results.append(result)
        else:
            failed.append(ticker)
    except Exception:
        failed.append(ticker)

    progress_bar.progress(index / len(tickers), text=f"Memproses {index}/{len(tickers)}: {ticker}")
    time.sleep(0.03)

progress_bar.empty()

if not results:
    st.error("Data tidak berhasil diproses. Coba lagi atau gunakan mode Ticker pilihan.")
    st.stop()

result_df = pd.DataFrame(results)
status_order = {"BELI KUAT": 0, "BELI": 1, "TIDAK LAYAK": 2}
result_df["_status_order"] = result_df["Status"].map(status_order)
result_df = result_df.sort_values(
    by=["_status_order", "Skor", "Volume Ratio"],
    ascending=[True, False, False],
).drop(columns="_status_order")

strong_buy = result_df[result_df["Status"] == "BELI KUAT"]
buy = result_df[result_df["Status"] == "BELI"]
not_suitable = result_df[result_df["Status"] == "TIDAK LAYAK"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Saham Diproses", len(result_df))
col2.metric("BELI KUAT", len(strong_buy))
col3.metric("BELI", len(buy))
col4.metric("TIDAK LAYAK", len(not_suitable))

st.subheader("Kandidat BELI KUAT")
if strong_buy.empty:
    st.warning("Belum ada BELI KUAT. Kriteria memang dibuat ketat agar sinyal tidak terlalu banyak.")
else:
    st.dataframe(strong_buy, use_container_width=True, hide_index=True)

st.subheader("Hasil Screening")
selected_status = st.multiselect(
    "Status yang ditampilkan",
    ["BELI KUAT", "BELI", "TIDAK LAYAK"],
    default=["BELI KUAT", "BELI"],
)
st.dataframe(result_df[result_df["Status"].isin(selected_status)], use_container_width=True, hide_index=True)

st.download_button(
    "Unduh hasil CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name=f"hasil_idx_screener_{date.today().isoformat()}.csv",
    mime="text/csv",
)

if failed:
    st.caption(f"Tidak dapat diproses: {len(failed)} ticker. Ini dapat terjadi bila data belum tersedia atau ticker sudah tidak aktif.")

st.divider()
st.subheader("Catatan risiko")
st.markdown(
    "- Target +5% dan +10% adalah target perencanaan, bukan kepastian.  \n"
    "- Tetap gunakan stop-loss dan batas risiko per posisi.  \n"
    "- Uji dengan paper trading/backtest sebelum memakai dana riil."
)
