import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
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
    scaler = joblib.load("preprocessor.pkl")
    return model, summary, scaler

model, summary, scaler = load_model()

BEST_LOOKBACK = summary["best_lookback"]   # 3
FEATURE_NAMES = summary["feature_names"]   # 15 nama fitur
# Fitur lag_k pada model ini dibentuk dari create_lag_features() di XGBoost.ipynb,
# yaitu tdf.shift(k) yang HANYA memakai lag ke-1 s.d. lag ke-lookback (lag ke-0 /
# return hari T sendiri TIDAK dipakai). Untuk mendapatkan lag ke-`lookback` yang
# valid (non-NaN) dari log-return, dibutuhkan lookback + 2 titik harga mentah,
# bukan lookback + 1 -- lag ke-1 butuh Close T-2, lag ke-lookback butuh
# Close (T-lookback-1). Sudah diverifikasi menghasilkan angka yang identik
# dengan create_lag_features() versi notebook.
N_ROWS_NEEDED = BEST_LOOKBACK + 3          # 5 baris OHLCV (T-4 .. T)
COLS          = ["Open", "High", "Low", "Close", "Volume"]

# ==========================================================
# HELPER: FEATURE ENGINEERING
# Mereplikasi persis create_lag_features() dari notebook
# ==========================================================

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Input  : DataFrame (N_ROWS_NEEDED) baris × 5 kolom OHLCV,
             urutan dari paling lama (baris 0) → paling baru (baris -1)
    Output : array (1, 15) siap masuk model
    """
    price_cols = ["Open", "High", "Low", "Close"]
    transformed = {}
    for col in price_cols:
        transformed[f"{col}_logret"] = np.log(df[col] / df[col].shift(1))
    transformed["Volume_log"] = np.log1p(df["Volume"])
    tdf = pd.DataFrame(transformed, index=df.index)

    feat_values = []
    for lag in range(1, BEST_LOOKBACK + 1):
        shifted = tdf.shift(lag)
        feat_values.extend(shifted.iloc[-1].values.tolist())

    return np.array(feat_values).reshape(1, -1)


# ==========================================================
# HELPER: FEATURE IMPORTANCE PLOT
# ==========================================================

def plot_feature_importance():
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Fitur"      : FEATURE_NAMES,
        "Importance" : importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    top_df = feat_imp_df  # semua 15 fitur

    # Warna per kelompok fitur
    color_map = {
        "Close": "#2196F3", "High": "#4CAF50",
        "Low"  : "#FF9800", "Open": "#F44336",
        "Volume": "#9C27B0"
    }
    def get_color(name):
        for key in color_map:
            if key in name:
                return color_map[key]
        return "#607D8B"

    bar_colors = [get_color(f) for f in top_df["Fitur"]]

    # ── Bar chart horizontal ──
    fig, ax = plt.subplots(figsize=(9, len(top_df) * 0.42 + 1.5))
    bars = ax.barh(
        top_df["Fitur"].values[::-1],
        top_df["Importance"].values[::-1],
        color=bar_colors[::-1],
        edgecolor="white",
        height=0.7
    )
    for bar, val in zip(bars, top_df["Importance"].values[::-1]):
        ax.text(
            bar.get_width() + max(top_df["Importance"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9
        )
    ax.set_title(
        f"Feature Importance — XGBoost (Lookback={BEST_LOOKBACK} hari)",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_xlim(0, max(top_df["Importance"]) * 1.20)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    return fig, feat_imp_df, color_map


def plot_group_pie(feat_imp_df, color_map):
    def get_group(name):
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in name:
                return col
        return "Lainnya"

    feat_imp_df = feat_imp_df.copy()
    feat_imp_df["Kelompok"] = feat_imp_df["Fitur"].apply(get_group)
    group_df = (
        feat_imp_df.groupby("Kelompok")["Importance"]
        .sum().sort_values(ascending=False).reset_index()
    )
    group_df["Persentase (%)"] = (
        group_df["Importance"] / group_df["Importance"].sum() * 100
    ).round(2)

    pie_c = [color_map.get(k, "#607D8B") for k in group_df["Kelompok"]]
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    ax2.pie(
        group_df["Importance"],
        labels=group_df["Kelompok"],
        autopct="%1.1f%%",
        colors=pie_c,
        startangle=90,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    ax2.set_title("Kontribusi per Kelompok Fitur", fontweight="bold")
    plt.tight_layout()
    return fig2, group_df


def plot_lag_bar(feat_imp_df):
    feat_imp_df = feat_imp_df.copy()

    def get_lag(name):
        for lag in range(1, BEST_LOOKBACK + 1):
            if f"_lag{lag}" in name:
                return f"Lag {lag}"
        return "Lainnya"

    feat_imp_df["Lag"] = feat_imp_df["Fitur"].apply(get_lag)
    lag_df = (
        feat_imp_df.groupby("Lag")["Importance"]
        .sum().sort_index().reset_index()
    )
    lag_df["Persentase (%)"] = (
        lag_df["Importance"] / lag_df["Importance"].sum() * 100
    ).round(2)

    lag_colors = ["#1565C0", "#42A5F5", "#90CAF9"][:len(lag_df)]
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    bars3 = ax3.bar(
        lag_df["Lag"], lag_df["Importance"],
        color=lag_colors, edgecolor="white", width=0.5
    )
    for bar, pct in zip(bars3, lag_df["Persentase (%)"]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lag_df["Importance"]) * 0.02,
            f"{pct:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    ax3.set_title("Kontribusi per Lag", fontweight="bold")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Total Importance")
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig3, lag_df


# ==========================================================
# SIDEBAR
# ==========================================================

menu = st.sidebar.radio(
    "Menu",
    ["Prediksi Harga", "Panduan Penggunaan"]
)

# ==========================================================
# HALAMAN PREDIKSI
# ==========================================================

if menu == "Prediksi Harga":

    st.title("📈 Prediksi Harga Penutupan Saham BBCA")
    st.info(
        f"Masukkan data OHLCV untuk **{N_ROWS_NEEDED} hari perdagangan** "
        f"secara berurutan dari yang paling lama (T-{N_ROWS_NEEDED - 1}) hingga hari terakhir (T). "
        f"Sistem akan memprediksi harga penutupan hari berikutnya (T+1)."
    )

    # ── Form input ──
    day_labels = [
        f"T-{N_ROWS_NEEDED - 1 - i}" if i < N_ROWS_NEEDED - 1 else "T  (Hari Terakhir)"
        for i in range(N_ROWS_NEEDED)
    ]

    data = []
    for i, label in enumerate(day_labels):
        st.subheader(f"Hari {label}")
        c1, c2, c3, c4, c5 = st.columns(5)
        inputs = [c1, c2, c3, c4, c5]
        row = []
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

        if (df_input == 0).any().any():
            st.error("⚠️ Semua kolom harus diisi dengan nilai lebih dari 0.")

        elif not all(df_input["High"] >= df_input["Low"]):
            st.error("⚠️ Nilai High harus lebih besar atau sama dengan Low di semua baris.")

        else:
            try:
                # 1. Buat fitur & prediksi
                X_raw           = build_features(df_input)
                X_scaled        = scaler.transform(X_raw)
                pred_log_return = model.predict(X_scaled)[0]

                # 2. Rekonstruksi harga
                close_today     = df_input.iloc[-1]["Close"]
                predicted_close = close_today * np.exp(pred_log_return)
                delta           = predicted_close - close_today
                delta_pct       = (delta / close_today) * 100

                # ── Hasil prediksi ──
                st.success("✅ Prediksi berhasil dihitung!")
                st.markdown("---")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric(
                    "Harga Close Hari Ini (T)",
                    f"Rp {close_today:,.0f}"
                )
                col_b.metric(
                    "Prediksi Close Besok (T+1)",
                    f"Rp {predicted_close:,.0f}",
                    delta=f"Rp {delta:+,.0f}  ({delta_pct:+.2f}%)"
                )
                col_c.metric(
                    "Prediksi Log-Return",
                    f"{pred_log_return:.6f}"
                )

                # ── Grafik pergerakan harga ──
                fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
                harga_tampil = list(df_input["Close"].values) + [predicted_close]
                label_tampil = (
                    [f"T-{N_ROWS_NEEDED - 1 - i}" if i < N_ROWS_NEEDED - 1 else "T"
                     for i in range(N_ROWS_NEEDED)]
                    + ["T+1\n(prediksi)"]
                )
                bar_colors_pred = ["steelblue"] * N_ROWS_NEEDED + ["tomato"]
                ax_bar.bar(label_tampil, harga_tampil,
                           color=bar_colors_pred, edgecolor="white", width=0.6)
                for idx_b, val in enumerate(harga_tampil):
                    ax_bar.text(idx_b, val + max(harga_tampil) * 0.005,
                                f"Rp {val:,.0f}", ha="center", va="bottom", fontsize=9)
                ax_bar.set_title(f"Pergerakan Harga Close (T-{N_ROWS_NEEDED - 1} s.d. T+1 Prediksi)",
                                 fontweight="bold")
                ax_bar.set_ylabel("Harga (IDR)")
                ax_bar.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f"Rp {x:,.0f}")
                )
                plt.tight_layout()
                st.pyplot(fig_bar)

                st.caption(
                    "⚠️ Prediksi ini merupakan output model machine learning "
                    "dan tidak menjamin pergerakan harga saham di masa mendatang."
                )

                # ── Feature Importance ──
                st.markdown("---")
                st.subheader("🔍 Feature Importance")
                st.markdown(
                    "Kontribusi setiap fitur input terhadap prediksi model XGBoost, "
                    "diukur menggunakan **importance score** dari seluruh pohon keputusan."
                )

                fig_imp, feat_imp_df, color_map = plot_feature_importance()

                col_imp, col_pie = st.columns([3, 2])

                with col_imp:
                    st.pyplot(fig_imp)
                    st.markdown(
                        "**Keterangan warna:** "
                        "🔵 Close &nbsp;|&nbsp; 🟢 High &nbsp;|&nbsp; "
                        "🟠 Low &nbsp;|&nbsp; 🔴 Open &nbsp;|&nbsp; 🟣 Volume"
                    )

                with col_pie:
                    fig_pie, group_df = plot_group_pie(feat_imp_df, color_map)
                    st.pyplot(fig_pie)
                    st.dataframe(
                        group_df.rename(columns={
                            "Kelompok": "Kelompok Fitur",
                            "Importance": "Total Importance"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown("---")

                col_lag, col_tbl = st.columns([2, 3])

                with col_lag:
                    st.markdown("**Kontribusi per Lag**")
                    st.caption(
                        "Seberapa jauh ke belakang informasi yang paling berguna bagi model."
                    )
                    fig_lag, lag_df = plot_lag_bar(feat_imp_df)
                    st.pyplot(fig_lag)
                    st.dataframe(
                        lag_df.rename(columns={"Importance": "Total Importance"}),
                        use_container_width=True,
                        hide_index=True
                    )

                with col_tbl:
                    st.markdown("**Tabel Semua Fitur**")
                    display_df = feat_imp_df[["Fitur", "Importance"]].copy()
                    display_df.index = range(1, len(display_df) + 1)
                    st.dataframe(
                        display_df.rename(columns={
                            "Fitur": "Nama Fitur",
                            "Importance": "Importance Score"
                        }),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

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
           **{N_ROWS_NEEDED} hari perdagangan** secara berurutan
           dari T-{N_ROWS_NEEDED - 1} (paling lama) hingga T (paling baru).
        3. Klik tombol **Prediksi Harga Besok**.
        4. Sistem menampilkan prediksi harga Close T+1 beserta
           grafik pergerakan harga dan feature importance model.

        ### Mengapa Butuh {N_ROWS_NEEDED} Hari?

        Model XGBoost menggunakan lookback **{BEST_LOOKBACK} hari**, dengan fitur
        lag ke-1 hingga lag ke-{BEST_LOOKBACK} (persis seperti fungsi
        `create_lag_features()` pada notebook training). Return hari T itu sendiri
        (lag ke-0) **tidak** dipakai sebagai fitur, sehingga lag ke-1 dihitung dari
        T-1 terhadap T-2, bukan dari T terhadap T-1. Akibatnya, untuk mendapatkan
        **{BEST_LOOKBACK} nilai log-return** yang lengkap dibutuhkan
        **{N_ROWS_NEEDED} titik harga** (lookback + 2):

        | Lag | Rumus Log-Return |
        |---|---|
        | Lag 1 | ln(Close_{{T-1}} / Close_{{T-2}}) |
        | Lag 2 | ln(Close_{{T-2}} / Close_{{T-3}}) |
        | Lag 3 (terjauh) | ln(Close_{{T-3}} / Close_{{T-4}}) |

        Hal yang sama berlaku untuk kolom Open, High, Low, dan Volume,
        sehingga total terdapat **15 fitur input** (5 kolom × 3 lag).

        ### Cara Rekonstruksi Harga

        Fitur yang sudah dibentuk dinormalisasi terlebih dahulu menggunakan
        **MinMaxScaler** (`preprocessor.pkl`) yang sama seperti saat pelatihan model,
        baru kemudian diproses XGBoost untuk menghasilkan prediksi **log-return**
        (bukan harga langsung):

        ```
        Close_pred(T+1) = Close_T × exp(log-return prediksi)
        ```

        ### Keterangan Hari

        | Simbol | Makna |
        |---|---|
        | T-4 | Empat hari perdagangan sebelum hari ini |
        | T-3 | Tiga hari perdagangan sebelum hari ini |
        | T-2 | Dua hari perdagangan sebelum hari ini |
        | T-1 | Satu hari perdagangan sebelum hari ini |
        | T | Hari perdagangan terakhir yang diketahui |
        | T+1 | Hari perdagangan berikutnya (diprediksi) |

        ### File yang Diperlukan

        Letakkan file berikut satu folder dengan `app_streamlit.py`:

        | File | Keterangan |
        |---|---|
        | `xgboost_best_lb3.json` | Bobot model XGBoost |
        | `xgboost_summary_lb3.json` | Metrik & hyperparameter |
        | `preprocessor.pkl` | MinMaxScaler hasil fit pada data training (wajib, dipakai untuk menormalisasi fitur sebelum prediksi) |

        ### Catatan

        - Gunakan data harga dari **hari perdagangan aktif** bursa.
        - Prediksi ini hanya alat bantu analisis dan **tidak menjamin**
          pergerakan harga saham di masa mendatang.
        """
    )
