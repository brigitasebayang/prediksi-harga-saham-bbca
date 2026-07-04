import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Prediksi Harga Saham BBCA",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("preprocessor.pkl")

    # Nama file scaler sesuai output notebook: xgboost_scaler.pkl
    scaler = joblib.load("preprocessor.pkl")

    with open("xgboost_summary_lb3.json", "r") as f:
        results = json.load(f)

    return model, scaler, results


model, scaler, results = load_model()

# ==========================================================
# FEATURE ENGINEERING
# ----------------------------------------------------------
# Model XGBoost dilatih dengan lookback = 3 hari.
# Untuk menghasilkan 3 lag features yang valid, dibutuhkan
# 4 baris data historis: T-3, T-2, T-1, T.
#
# Log-return dihitung: ln(P_t / P_{t-1})
#   → baris pertama (T-3) selalu NaN karena tidak ada T-4
#   → baris T-2, T-1, T masing-masing menghasilkan 1 log-return
#
# Fitur lag:
#   lag1 = log-return di hari T   (paling baru) = iloc[-1]
#   lag2 = log-return di hari T-1              = iloc[-2]
#   lag3 = log-return di hari T-2              = iloc[-3]
#
# Urutan fitur: [lag1 × 5, lag2 × 5, lag3 × 5] = 15 fitur
# Ini konsisten dengan urutan pelatihan di notebook XGBoost.
# ==========================================================
def create_input_features(df: pd.DataFrame) -> np.ndarray:
    """
    Parameters
    ----------
    df : pd.DataFrame, shape (4, 5)
        Baris berurutan: T-3, T-2, T-1, T
        Kolom: Open, High, Low, Close, Volume

    Returns
    -------
    np.ndarray, shape (1, 15)
        Vektor fitur siap ditransformasi scaler & diprediksi model
    """
    transformed = pd.DataFrame(index=df.index)
    transformed["Open_logret"]  = np.log(df["Open"]  / df["Open"].shift(1))
    transformed["High_logret"]  = np.log(df["High"]  / df["High"].shift(1))
    transformed["Low_logret"]   = np.log(df["Low"]   / df["Low"].shift(1))
    transformed["Close_logret"] = np.log(df["Close"] / df["Close"].shift(1))
    transformed["Volume_log"]   = np.log1p(df["Volume"])

    # Susun fitur: lag1 (T) → lag2 (T-1) → lag3 (T-2)
    feat = []
    for lag in range(1, 4):           # lag = 1, 2, 3
        feat.extend(transformed.iloc[-lag].values.tolist())

    return np.array(feat).reshape(1, -1)


# Nama fitur sesuai urutan pelatihan notebook
FEATURE_NAMES = [
    "Open logret (T)",      "High logret (T)",      "Low logret (T)",
    "Close logret (T)",     "Volume log (T)",
    "Open logret (T-1)",    "High logret (T-1)",    "Low logret (T-1)",
    "Close logret (T-1)",   "Volume log (T-1)",
    "Open logret (T-2)",    "High logret (T-2)",    "Low logret (T-2)",
    "Close logret (T-2)",   "Volume log (T-2)",
]

COLS = ["Open", "High", "Low", "Close", "Volume"]

# ==========================================================
# SIDEBAR
# ==========================================================
menu = st.sidebar.radio(
    "📂 Menu",
    ["🔮 Prediksi Harga", "📖 Panduan Penggunaan"]
)

# ==========================================================
# HALAMAN PREDIKSI
# ==========================================================
if menu == "🔮 Prediksi Harga":

    st.title("📈 Prediksi Harga Penutupan Saham BBCA")
    st.caption(
        "Model: XGBoost · Lookback: 3 hari · "
        "RMSE uji: 140,84 IDR · R²: 0,9605"
    )
    st.info(
        "Masukkan data OHLCV untuk **4 hari perdagangan terakhir** "
        "(T-3, T-2, T-1, T) agar semua fitur lag dapat dihitung dengan benar."
    )

    # ── Input form ──────────────────────────────────────────
    day_labels = [
        ("T-3 (3 Hari Lalu)", "t3"),
        ("T-2 (2 Hari Lalu)", "t2"),
        ("T-1 (Kemarin)",     "t1"),
        ("T  (Hari Ini)",     "t0"),
    ]

    data = []
    for label, key in day_labels:
        st.subheader(label)
        c1, c2, c3, c4, c5 = st.columns(5)
        vals = [
            c1.number_input("Open",   min_value=0.0, value=0.0, key=f"Open_{key}",   format="%.2f"),
            c2.number_input("High",   min_value=0.0, value=0.0, key=f"High_{key}",   format="%.2f"),
            c3.number_input("Low",    min_value=0.0, value=0.0, key=f"Low_{key}",    format="%.2f"),
            c4.number_input("Close",  min_value=0.0, value=0.0, key=f"Close_{key}",  format="%.2f"),
            c5.number_input("Volume", min_value=0.0, value=0.0, key=f"Volume_{key}", format="%.0f"),
        ]
        data.append(vals)

    st.markdown("---")

    if st.button("🔮 Prediksi Harga Besok (T+1)", use_container_width=True):

        df = pd.DataFrame(data, columns=COLS)

        # ── Validasi input ───────────────────────────────────
        if (df == 0).any().any():
            st.error("⚠️  Semua kolom untuk keempat hari harus diisi dengan nilai lebih dari 0.")
            st.stop()

        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            st.error("⚠️  Nilai harga (Open/High/Low/Close) harus lebih dari 0.")
            st.stop()

        try:
            X = create_input_features(df)

            if np.isnan(X).any():
                nan_count = int(np.isnan(X).sum())
                st.error(
                    f"⚠️  Terdapat {nan_count} nilai NaN dalam fitur. "
                    "Pastikan semua harga lebih dari 0."
                )
                st.stop()

            # ── Prediksi ─────────────────────────────────────
            X_scaled        = scaler.transform(X)
            pred_log_return = model.predict(X_scaled)[0]
            close_today     = df.iloc[-1]["Close"]
            predicted_close = close_today * np.exp(pred_log_return)

            perubahan     = predicted_close - close_today
            perubahan_pct = (perubahan / close_today) * 100
            arah          = "▲" if perubahan >= 0 else "▼"

            # ── Hasil utama ───────────────────────────────────
            if perubahan >= 0:
                st.success(
                    f"✅  Prediksi Harga Close Besok (T+1): "
                    f"**Rp {predicted_close:,.2f}**  "
                    f"{arah} {abs(perubahan_pct):.2f}%"
                )
            else:
                st.warning(
                    f"📉  Prediksi Harga Close Besok (T+1): "
                    f"**Rp {predicted_close:,.2f}**  "
                    f"{arah} {abs(perubahan_pct):.2f}%"
                )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Harga Close Hari Ini (T)",
                f"Rp {close_today:,.2f}"
            )
            m2.metric(
                "Prediksi Close Besok (T+1)",
                f"Rp {predicted_close:,.2f}",
                delta=f"Rp {perubahan:+,.2f}"
            )
            m3.metric(
                "Perkiraan Perubahan",
                f"{arah} {abs(perubahan_pct):.2f}%"
            )

            # ── Feature importance ────────────────────────────
            st.markdown("---")
            st.subheader("📊 Faktor yang Mempengaruhi Prediksi")

            importance    = model.feature_importances_
            feat_labels   = FEATURE_NAMES[:len(importance)]

            importance_df = (
                pd.DataFrame({"Faktor": feat_labels, "Tingkat Pengaruh": importance})
                .sort_values("Tingkat Pengaruh", ascending=False)
                .reset_index(drop=True)
            )
            importance_df.index += 1

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(
                importance_df["Faktor"].head(10),
                importance_df["Tingkat Pengaruh"].head(10),
                color="steelblue", edgecolor="white"
            )
            ax.set_xlabel("Feature Importance (Gain)")
            ax.set_title("Top-10 Fitur Paling Berpengaruh", fontweight="bold")
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(
                "**Interpretasi:** Log-return merepresentasikan perubahan harga relatif harian. "
                "Semakin besar nilai importance, semakin besar kontribusi fitur tersebut "
                "terhadap prediksi yang dihasilkan model XGBoost."
            )

            with st.expander("📋 Lihat tabel lengkap feature importance"):
                st.dataframe(importance_df, use_container_width=True)

            st.info(
                f"Faktor paling berpengaruh: "
                f"**{importance_df.iloc[0]['Faktor']}** "
                f"(importance = {importance_df.iloc[0]['Tingkat Pengaruh']:.4f})"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# ==========================================================
# PANDUAN PENGGUNAAN
# ==========================================================
else:

    st.title("📖 Panduan Penggunaan")

    st.markdown("""
    ### Langkah-langkah

    1. Pilih menu **🔮 Prediksi Harga** di sidebar kiri.
    2. Masukkan data **Open, High, Low, Close, dan Volume** untuk **4 hari perdagangan terakhir** (T-3, T-2, T-1, T).
    3. Klik tombol **🔮 Prediksi Harga Besok (T+1)**.
    4. Sistem akan menampilkan prediksi harga Close untuk hari berikutnya beserta grafik feature importance.

    ### Keterangan Input

    | Label | Keterangan |
    |-------|-----------|
    | **T-3** | Tiga hari perdagangan sebelum hari ini |
    | **T-2** | Dua hari perdagangan sebelum hari ini |
    | **T-1** | Kemarin (satu hari perdagangan sebelum hari ini) |
    | **T**   | Hari perdagangan terakhir yang sudah diketahui (hari ini) |

    ### Mengapa 4 hari data?

    Model XGBoost menggunakan **lookback = 3 hari**. Log-return dihitung dari rasio harga dua hari
    berurutan, sehingga untuk mendapatkan 3 lag log-return yang valid dibutuhkan 4 titik harga historis:

    - **log-return T** = ln(Close_T / Close_{T-1}) → fitur lag-1
    - **log-return T-1** = ln(Close_{T-1} / Close_{T-2}) → fitur lag-2
    - **log-return T-2** = ln(Close_{T-2} / Close_{T-3}) → fitur lag-3

    ### Keterangan Kolom

    | Kolom | Keterangan |
    |-------|-----------|
    | **Open** | Harga pembukaan (IDR per lembar) |
    | **High** | Harga tertinggi hari itu (IDR) |
    | **Low** | Harga terendah hari itu (IDR) |
    | **Close** | Harga penutupan (IDR) — **target prediksi** |
    | **Volume** | Volume perdagangan (jumlah lembar saham) |

    ### Sumber Data

    Data OHLCV BBCA dapat diperoleh dari:
    - [Yahoo Finance — BBCA.JK](https://finance.yahoo.com/quote/BBCA.JK/)
    - [IDX (Bursa Efek Indonesia)](https://www.idx.co.id/)
    - Platform trading/broker masing-masing

    ### Performa Model

    | Metrik | Nilai |
    |--------|-------|
    | RMSE (data uji) | 140,84 IDR |
    | MAE (data uji) | 105,53 IDR |
    | R² (data uji) | 0,9605 |
    | Algoritma | XGBoost (Extreme Gradient Boosting) |
    | Lookback | 3 hari perdagangan |
    | Optimasi | Optuna TPE (50 trial) |

    ### Catatan Penting

    > ⚠️ Prediksi ini bersifat **alat bantu analisis** dan **tidak menjamin** pergerakan
    > harga saham di masa mendatang. Keputusan investasi sepenuhnya adalah tanggung jawab pengguna.
    """)
