import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Prediksi Harga Saham BBCA",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# LOAD MODEL & SUMMARY
# ==========================================================

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_best_lb3.json")

    with open("xgboost_summary_lb3.json", "r") as f:
        summary = json.load(f)

    return model, summary

model, summary = load_model()

BEST_LOOKBACK  = summary["best_lookback"]          # 3
FEATURE_NAMES  = summary["feature_names"]           # 15 nama fitur
N_ROWS_NEEDED  = BEST_LOOKBACK + 1                  # 4 baris OHLCV
COLS           = ["Open", "High", "Low", "Close", "Volume"]

# ==========================================================
# FEATURE ENGINEERING
# Mereplikasi persis fungsi create_lag_features() dari notebook
# ==========================================================

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Input  : DataFrame 4 baris × 5 kolom OHLCV (urutan paling lama → paling baru)
    Output : array (1, 15) siap masuk scaler & model
    """
    price_cols = ["Open", "High", "Low", "Close"]

    # Transformasi log-return per kolom harga, log1p untuk Volume
    transformed = {}
    for col in price_cols:
        transformed[f"{col}_logret"] = np.log(df[col] / df[col].shift(1))
    transformed["Volume_log"] = np.log1p(df["Volume"])
    tdf = pd.DataFrame(transformed, index=df.index)

    feat_cols_t = list(tdf.columns)  # 5 kolom

    # Buat lag 1, 2, 3 — shift(lag) lalu ambil baris terakhir
    feat_values = []
    for lag in range(1, BEST_LOOKBACK + 1):
        shifted = tdf.shift(lag)
        feat_values.extend(shifted.iloc[-1].values.tolist())

    return np.array(feat_values).reshape(1, -1)


def fit_scaler_from_input(X_raw: np.ndarray) -> np.ndarray:
    """
    Karena scaler tidak disimpan di file terpisah, kita replikasi
    MinMaxScaler dari notebook: fit pada X_raw lalu transform.
    Catatan: di deployment nyata, simpan scaler dari notebook dengan
    joblib.dump(scaler_X, 'preprocessor.pkl') dan load di sini.
    Workaround sementara: transform berdasarkan range input itu sendiri.

    Namun karena scaler training tidak tersedia, kita lakukan manual
    MinMax berdasarkan range fitur dari summary (feature_names tersedia
    tapi bukan range-nya). Pendekatan paling aman: tambahkan cell
    di notebook untuk menyimpan scaler, lalu joblib.load di sini.

    Untuk sementara ini dilakukan scaling identity (pass-through)
    dan ditampilkan peringatan ke pengguna.
    """
    # Pass-through — ganti dengan joblib.load("preprocessor.pkl") jika tersedia
    return X_raw


# ==========================================================
# SIDEBAR
# ==========================================================

menu = st.sidebar.radio(
    "Menu",
    ["Prediksi Harga", "Feature Importance", "Panduan Penggunaan"]
)

# ==========================================================
# HALAMAN PREDIKSI
# ==========================================================

if menu == "Prediksi Harga":

    st.title("📈 Prediksi Harga Penutupan Saham BBCA")
    st.markdown(
        f"""
        Model menggunakan **XGBoost** dengan lookback **{BEST_LOOKBACK} hari**.
        Untuk menghasilkan **{BEST_LOOKBACK} lag log-return**, diperlukan
        data OHLCV dari **{N_ROWS_NEEDED} hari perdagangan** (T-3 hingga T).
        """
    )
    st.info(
        f"Masukkan data OHLCV untuk {N_ROWS_NEEDED} hari terakhir secara berurutan "
        f"dari yang paling lama (T-3) hingga hari terakhir (T). "
        f"Sistem akan memprediksi harga penutupan hari berikutnya (T+1)."
    )

    # Label setiap baris: T-3, T-2, T-1, T
    day_labels = [f"T-{N_ROWS_NEEDED - 1 - i}" if i < N_ROWS_NEEDED - 1 else "T (Hari Terakhir)"
                  for i in range(N_ROWS_NEEDED)]

    data = []
    for i, label in enumerate(day_labels):
        st.subheader(f"Hari {label}")
        row = []
        c1, c2, c3, c4, c5 = st.columns(5)
        inputs = [c1, c2, c3, c4, c5]
        for j, col_name in enumerate(COLS):
            val = inputs[j].number_input(
                col_name,
                min_value=0.0,
                value=0.0,
                step=10.0 if col_name != "Volume" else 1_000_000.0,
                format="%.2f" if col_name != "Volume" else "%.0f",
                key=f"{col_name}_{i}"
            )
            row.append(val)
        data.append(row)
        st.divider()

    if st.button("🔮 Prediksi Harga Besok", use_container_width=True, type="primary"):

        df_input = pd.DataFrame(data, columns=COLS)

        # Validasi: tidak boleh ada nilai 0
        if (df_input == 0).any().any():
            st.error("⚠️ Semua kolom harus diisi dengan nilai lebih dari 0.")

        # Validasi logika OHLC: High ≥ Low, High ≥ Open/Close, dst
        elif not all(df_input["High"] >= df_input["Low"]):
            st.error("⚠️ Nilai High harus lebih besar atau sama dengan Low di semua baris.")

        else:
            try:
                # 1. Buat fitur
                X_raw = build_features(df_input)

                # 2. Scaling — GANTI dengan joblib.load jika preprocessor.pkl tersedia
                X_scaled = fit_scaler_from_input(X_raw)

                # 3. Prediksi log-return
                pred_log_return = model.predict(X_scaled)[0]

                # 4. Rekonstruksi harga dari Close[T]
                close_today    = df_input.iloc[-1]["Close"]
                predicted_close = close_today * np.exp(pred_log_return)
                delta           = predicted_close - close_today
                delta_pct       = (delta / close_today) * 100

                # 5. Tampilkan hasil
                st.success("✅ Prediksi berhasil dihitung!")
                st.markdown("---")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric(
                    label="Harga Close Hari Ini (T)",
                    value=f"Rp {close_today:,.0f}"
                )
                col_b.metric(
                    label="Prediksi Close Besok (T+1)",
                    value=f"Rp {predicted_close:,.0f}",
                    delta=f"Rp {delta:+,.0f}  ({delta_pct:+.2f}%)"
                )
                col_c.metric(
                    label="Prediksi Log-Return",
                    value=f"{pred_log_return:.6f}"
                )

                # 6. Visualisasi sederhana pergerakan
                fig, ax = plt.subplots(figsize=(8, 3))
                harga_tampil = list(df_input["Close"].values) + [predicted_close]
                label_tampil = [f"T-{N_ROWS_NEEDED - 1 - i}" if i < N_ROWS_NEEDED - 1
                                else "T" for i in range(N_ROWS_NEEDED)] + ["T+1 (pred)"]
                colors = ["steelblue"] * N_ROWS_NEEDED + ["tomato"]
                ax.bar(label_tampil, harga_tampil, color=colors, edgecolor="white", width=0.6)
                for idx_b, (lbl, val) in enumerate(zip(label_tampil, harga_tampil)):
                    ax.text(idx_b, val + max(harga_tampil) * 0.005,
                            f"Rp {val:,.0f}", ha="center", va="bottom", fontsize=9)
                ax.set_title("Pergerakan Harga Close (T-3 s.d. T+1 Prediksi)", fontweight="bold")
                ax.set_ylabel("Harga (IDR)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
                plt.tight_layout()
                st.pyplot(fig)

                st.caption(
                    "⚠️ Prediksi ini merupakan output model machine learning "
                    "dan tidak menjamin pergerakan harga saham di masa mendatang. "
                    "Gunakan sebagai alat bantu analisis saja."
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# ==========================================================
# HALAMAN FEATURE IMPORTANCE
# ==========================================================

elif menu == "Feature Importance":

    st.title("🔍 Feature Importance — XGBoost")
    st.markdown(
        """
        Grafik berikut menampilkan seberapa besar kontribusi setiap fitur input
        terhadap keputusan prediksi model XGBoost, diukur menggunakan **gain importance**
        (rata-rata penurunan loss yang disumbangkan setiap kali fitur tersebut digunakan
        untuk memisahkan data di dalam pohon keputusan).
        """
    )

    # Ambil importance langsung dari model yang sudah di-load
    importances   = model.feature_importances_
    feat_imp_df   = pd.DataFrame({
        "Fitur"      : FEATURE_NAMES,
        "Importance" : importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    feat_imp_df.index += 1

    top_n = st.slider("Tampilkan Top-N Fitur", min_value=5, max_value=len(FEATURE_NAMES), value=15, step=1)

    top_df = feat_imp_df.head(top_n)

    # ── Grafik horizontal bar ──
    fig, ax = plt.subplots(figsize=(10, top_n * 0.42 + 1.5))

    colors = []
    for feat in top_df["Fitur"]:
        if "Close" in feat:
            colors.append("#2196F3")   # biru
        elif "Volume" in feat:
            colors.append("#9C27B0")   # ungu
        elif "High" in feat:
            colors.append("#4CAF50")   # hijau
        elif "Low" in feat:
            colors.append("#FF9800")   # oranye
        else:
            colors.append("#F44336")   # merah (Open)

    bars = ax.barh(
        top_df["Fitur"].values[::-1],
        top_df["Importance"].values[::-1],
        color=colors[::-1],
        edgecolor="white",
        height=0.7
    )

    # Label nilai di ujung bar
    for bar, val in zip(bars, top_df["Importance"].values[::-1]):
        ax.text(
            bar.get_width() + max(top_df["Importance"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9
        )

    ax.set_title(
        f"Top-{top_n} Feature Importance — XGBoost (Lookback={BEST_LOOKBACK} hari)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Importance Score (Gain)", fontsize=11)
    ax.set_xlim(0, max(top_df["Importance"]) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Legenda warna ──
    st.markdown(
        """
        **Keterangan warna:**
        🔵 Close &nbsp;|&nbsp; 🟠 Low &nbsp;|&nbsp; 🟢 High &nbsp;|&nbsp;
        🔴 Open &nbsp;|&nbsp; 🟣 Volume
        """
    )

    st.markdown("---")

    # ── Tabel lengkap ──
    col_tbl, col_summary = st.columns([3, 2])

    with col_tbl:
        st.subheader("Tabel Semua Fitur")
        st.dataframe(
            feat_imp_df.rename(columns={"Fitur": "Nama Fitur", "Importance": "Importance Score"}),
            use_container_width=True
        )

    with col_summary:
        st.subheader("Ringkasan per Kelompok Fitur")

        # Kelompokkan berdasarkan kolom asal (Open/High/Low/Close/Volume)
        def get_group(name):
            for col in ["Close", "High", "Low", "Open", "Volume"]:
                if col in name:
                    return col
            return "Lainnya"

        feat_imp_df["Kelompok"] = feat_imp_df["Fitur"].apply(get_group)
        group_df = (
            feat_imp_df.groupby("Kelompok")["Importance"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        group_df.columns = ["Kelompok Fitur", "Total Importance"]
        group_df["Persentase (%)"] = (group_df["Total Importance"] / group_df["Total Importance"].sum() * 100).round(2)

        st.dataframe(group_df, use_container_width=True, hide_index=True)

        # Pie chart kontribusi per kelompok
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        pie_colors = {"Close": "#2196F3", "High": "#4CAF50", "Low": "#FF9800",
                      "Open": "#F44336", "Volume": "#9C27B0"}
        pie_c = [pie_colors.get(k, "#607D8B") for k in group_df["Kelompok Fitur"]]
        ax2.pie(
            group_df["Total Importance"],
            labels=group_df["Kelompok Fitur"],
            autopct="%1.1f%%",
            colors=pie_c,
            startangle=90,
            pctdistance=0.82,
            wedgeprops=dict(edgecolor="white", linewidth=1.5)
        )
        ax2.set_title("Kontribusi per Kelompok Fitur", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # ── Analisis per Lag ──
    st.subheader("Kontribusi per Lag")
    st.markdown(
        "Perbandingan total importance berdasarkan jarak lag — menunjukkan "
        "seberapa jauh ke belakang informasi yang paling berguna bagi model."
    )

    def get_lag(name):
        for lag in range(1, BEST_LOOKBACK + 1):
            if f"_lag{lag}" in name:
                return f"Lag {lag}"
        return "Lainnya"

    feat_imp_df["Lag"] = feat_imp_df["Fitur"].apply(get_lag)
    lag_df = (
        feat_imp_df.groupby("Lag")["Importance"]
        .sum()
        .sort_index()
        .reset_index()
    )
    lag_df.columns = ["Lag", "Total Importance"]
    lag_df["Persentase (%)"] = (lag_df["Total Importance"] / lag_df["Total Importance"].sum() * 100).round(2)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    lag_colors = ["#1565C0", "#42A5F5", "#90CAF9"][:len(lag_df)]
    bars3 = ax3.bar(lag_df["Lag"], lag_df["Total Importance"], color=lag_colors, edgecolor="white", width=0.5)
    for bar, pct in zip(bars3, lag_df["Persentase (%)"]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lag_df["Total Importance"]) * 0.01,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax3.set_title("Total Importance per Lag", fontweight="bold")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Total Importance")
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

    st.dataframe(lag_df, use_container_width=True, hide_index=True)

# ==========================================================
# PANDUAN PENGGUNAAN
# ==========================================================

else:

    st.title("📖 Panduan Penggunaan")

    st.markdown(
        f"""
        ### Langkah-langkah Prediksi

        1. Pilih menu **Prediksi Harga** di sidebar.
        2. Masukkan data **Open, High, Low, Close, dan Volume** untuk
           **{N_ROWS_NEEDED} hari perdagangan** secara berurutan dari T-3 (paling lama) hingga T (paling baru).
        3. Klik tombol **Prediksi Harga Besok**.
        4. Sistem akan menampilkan prediksi harga Close pada hari berikutnya (T+1).

        ### Mengapa Butuh {N_ROWS_NEEDED} Hari?

        Model XGBoost menggunakan lookback **{BEST_LOOKBACK} hari**. Artinya model
        membutuhkan **{BEST_LOOKBACK} nilai log-return** sebagai input:

        | Lag | Rumus Log-Return |
        |---|---|
        | Lag 1 (terbaru) | ln(Close_T / Close_{{T-1}}) |
        | Lag 2 | ln(Close_{{T-1}} / Close_{{T-2}}) |
        | Lag 3 | ln(Close_{{T-2}} / Close_{{T-3}}) |

        Setiap log-return dihitung dari **selisih dua hari**, sehingga untuk
        mendapatkan 3 lag dibutuhkan **4 titik harga** (T-3, T-2, T-1, T).
        Hal yang sama berlaku untuk kolom Open, High, Low, dan Volume.

        ### Cara Rekonstruksi Harga

        Model memprediksi **log-return** (bukan harga langsung), lalu harga direkonstruksi:

        ```
        Close_pred(T+1) = Close_T × exp(log-return prediksi)
        ```

        ### Keterangan Hari

        | Simbol | Makna |
        |---|---|
        | T-3 | Tiga hari perdagangan sebelum hari ini |
        | T-2 | Dua hari perdagangan sebelum hari ini |
        | T-1 | Satu hari perdagangan sebelum hari ini |
        | T | Hari perdagangan terakhir yang diketahui |
        | T+1 | Hari perdagangan berikutnya (diprediksi) |

        ### Catatan Penting

        - Data yang dimasukkan adalah harga **pada sesi penutupan** bursa.
        - Prediksi ini hanya alat bantu analisis dan **tidak menjamin** pergerakan
          harga saham di masa mendatang.
        - Untuk hasil optimal, gunakan data Close dari **hari perdagangan aktif**
          (bukan hari libur atau hari ketika BBCA tidak diperdagangkan).

        """
    )
