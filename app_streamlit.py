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
    model.load_model("xgboost_best_lb3.json")

    scaler = joblib.load("preprocessor.pkl")

    with open("xgboost_summary_lb3.json", "r") as f:
        results = json.load(f)

    return model, scaler, results


model, scaler, results = load_model()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def create_input_features(df):
    transformed = pd.DataFrame()

    transformed["Open_logret"] = np.log(
        df["Open"] / df["Open"].shift(1)
    )
    transformed["High_logret"] = np.log(
        df["High"] / df["High"].shift(1)
    )
    transformed["Low_logret"] = np.log(
        df["Low"] / df["Low"].shift(1)
    )
    transformed["Close_logret"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )
    transformed["Volume_log"] = np.log1p(
        df["Volume"]
    )

    feat = []

    for lag in range(1, 4):
        row = transformed.shift(lag)
        feat.extend(
            row.iloc[-1].values.tolist()
        )

    return np.array(feat).reshape(1, -1)


# ==========================================================
# SIDEBAR
# ==========================================================
menu = st.sidebar.radio(
    "Menu",
    [
        "Prediksi Harga",
        "Panduan Penggunaan"
    ]
)

# ==========================================================
# HALAMAN PREDIKSI
# ==========================================================
if menu == "Prediksi Harga":

    st.title("📈 Prediksi Harga Penutupan Saham BBCA")

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = []

    # T-2
    st.subheader("Hari T-2")
    row1 = []

    for c in cols:
        row1.append(
            st.number_input(
                f"{c} (T-2)",
                min_value=0.0,
                value=0.0,
                key=f"{c}_1"
            )
        )

    data.append(row1)

    # T-1
    st.subheader("Hari T-1")
    row2 = []

    for c in cols:
        row2.append(
            st.number_input(
                f"{c} (T-1)",
                min_value=0.0,
                value=0.0,
                key=f"{c}_2"
            )
        )

    data.append(row2)

    # T
    st.subheader("Hari T")
    row3 = []

    for c in cols:
        row3.append(
            st.number_input(
                f"{c} (T)",
                min_value=0.0,
                value=0.0,
                key=f"{c}_3"
            )
        )

    data.append(row3)

    if st.button("Prediksi Harga Besok"):

        df = pd.DataFrame(
            data,
            columns=cols
        )

        if (df == 0).any().any():
            st.error("Semua nilai harus diisi.")
        else:
            try:
                X = create_input_features(df)

                X_scaled = scaler.transform(X)

                pred_log_return = model.predict(
                    X_scaled
                )[0]

                close_today = df.iloc[-1]["Close"]

                predicted_close = (
                    close_today *
                    np.exp(pred_log_return)
                )

                st.success(
                    f"Prediksi Harga Close Besok : Rp {predicted_close:,.2f}"
                )

                st.metric(
                    "Harga Close Hari Ini",
                    f"Rp {close_today:,.2f}"
                )

                st.metric(
                    "Prediksi Close Besok",
                    f"Rp {predicted_close:,.2f}"
                )

            except Exception as e:
                st.error(str(e))

# ==========================================================
# PANDUAN
# ==========================================================
else:

    st.title("📖 Panduan Penggunaan")

    st.markdown("""
    ### Langkah-langkah

    1. Pilih menu **Prediksi Harga**.
    2. Masukkan data Open, High, Low, Close, dan Volume untuk 3 hari terakhir.
    3. Klik tombol **Prediksi Harga Besok**.
    4. Sistem akan menampilkan prediksi harga penutupan.
    5. Menu **Evaluasi Model** digunakan untuk melihat performa model XGBoost.

    ### Keterangan

    - T-2 = Dua hari sebelum hari ini
    - T-1 = Satu hari sebelum hari ini
    - T = Hari terakhir yang diketahui

    ### Catatan

    Prediksi ini hanya digunakan sebagai alat bantu analisis dan tidak menjamin
    pergerakan harga saham di masa mendatang.
    """)
