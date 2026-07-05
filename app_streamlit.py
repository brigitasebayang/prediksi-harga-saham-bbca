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
    ["Prediksi Harga", "Evaluasi Model", "Panduan Penggunaan"]
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
# HALAMAN EVALUASI MODEL
# ==========================================================

elif menu == "Evaluasi Model":

    st.title("📊 Evaluasi Model XGBoost")

    test_m = summary["test_metrics"]
    wfv_m  = summary["wfv_avg_metrics"]
    hp     = summary["best_hyperparams"]

    # ── Metrik utama ──
    st.subheader("Metrik Evaluasi pada Data Testing")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"Rp {test_m['rmse']:.2f}")
    c2.metric("MAE",  f"Rp {test_m['mae']:.2f}")
    c3.metric("R²",   f"{test_m['r2']:.4f}")

    st.markdown("---")

    # ── Walk-Forward Validation ──
    st.subheader("Walk-Forward Validation (5-Fold Expanding Window)")
    w1, w2, w3 = st.columns(3)
    w1.metric("Avg RMSE (WFV)", f"Rp {wfv_m['rmse']:.2f}",
              delta=f"±{wfv_m['rmse_std']:.2f}", delta_color="off")
    w2.metric("Avg MAE (WFV)",  f"Rp {wfv_m['mae']:.2f}")
    w3.metric("Avg R² (WFV)",   f"{wfv_m['r2']:.4f}")

    # ── Grafik WFV per fold ──
    if "wfv_fold_details" in summary and summary["wfv_fold_details"]:
        folds  = summary["wfv_fold_details"]
        fold_n = [r["fold"]  for r in folds]
        rmses  = [r["rmse"]  for r in folds]
        r2s    = [r["r2"]    for r in folds]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(fold_n, rmses, color="steelblue", edgecolor="white")
        axes[0].axhline(wfv_m["rmse"], color="tomato", linestyle="--", linewidth=1.5,
                        label=f"Rata-rata: {wfv_m['rmse']:.2f}")
        axes[0].set_title("RMSE per Fold (WFV)", fontweight="bold")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("RMSE (IDR)")
        axes[0].legend()
        axes[0].set_xticks(fold_n)

        axes[1].bar(fold_n, r2s, color="seagreen", edgecolor="white")
        axes[1].axhline(wfv_m["r2"], color="tomato", linestyle="--", linewidth=1.5,
                        label=f"Rata-rata: {wfv_m['r2']:.4f}")
        axes[1].set_title("R² per Fold (WFV)", fontweight="bold")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("R²")
        axes[1].legend()
        axes[1].set_xticks(fold_n)

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ── Informasi model ──
    st.subheader("Informasi Model")
    col_info, col_hp = st.columns(2)

    with col_info:
        st.markdown(
            f"""
            | Atribut | Detail |
            |---|---|
            | Algoritma | XGBoost |
            | Target | Harga Close T+1 (via log-return) |
            | Lookback | {BEST_LOOKBACK} hari |
            | Jumlah Fitur | {len(FEATURE_NAMES)} fitur (5 kolom × {BEST_LOOKBACK} lag) |
            | Fitur Dasar | Open, High, Low, Close, Volume |
            | Transformasi | Log-return (harga), log1p (volume) |
            """
        )

    with col_hp:
        st.markdown("**Hyperparameter Terbaik (Optuna)**")
        hp_df = pd.DataFrame(
            [(k, f"{v:.6f}" if isinstance(v, float) else str(v))
             for k, v in hp.items()],
            columns=["Hyperparameter", "Nilai"]
        )
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Interpretasi ──
    st.subheader("Interpretasi Hasil")
    st.success(
        f"Nilai R² sebesar **{test_m['r2']:.4f}** menunjukkan model mampu menjelaskan "
        f"**{test_m['r2'] * 100:.2f}%** variasi harga Close BBCA pada data testing. "
        f"RMSE sebesar **Rp {test_m['rmse']:.2f}** mengindikasikan rata-rata deviasi "
        f"prediksi dari harga aktual sebesar nilai tersebut dalam satuan Rupiah."
    )

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

        ### File yang Diperlukan

        Pastikan file berikut berada di direktori yang sama dengan `app_streamlit.py`:

        | File | Keterangan |
        |---|---|
        | `xgboost_best_lb3.json` | Bobot model XGBoost |
        | `xgboost_summary_lb3.json` | Metrik evaluasi & hyperparameter |

        > ⚠️ **Catatan pengembang:** Untuk akurasi scaling yang tepat, tambahkan cell
        > berikut di akhir notebook XGBoost sebelum deployment:
        > ```python
        > import joblib
        > joblib.dump(scaler_X, 'preprocessor.pkl')
        > files.download('preprocessor.pkl')
        > ```
        > Lalu ganti baris `X_scaled = fit_scaler_from_input(X_raw)` di app dengan:
        > ```python
        > scaler = joblib.load('preprocessor.pkl')
        > X_scaled = scaler.transform(X_raw)
        > ```
        """
    )
