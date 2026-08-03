from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Volume Sampah Jawa Barat",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).parent / "data_timbulan_sampah_jabar_2019_2025.csv"


# =========================================================
# CSS TAMPILAN WEBSITE
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(34, 197, 94, .24), transparent 30%),
        radial-gradient(circle at 95% 6%, rgba(20, 184, 166, .20), transparent 32%),
        radial-gradient(circle at 50% 90%, rgba(132, 204, 22, .10), transparent 32%),
        linear-gradient(135deg, #06110d 0%, #0b1f17 48%, #03100b 100%);
    color: #ecfdf5;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: rgba(4, 20, 13, .82);
    border-right: 1px solid rgba(187, 247, 208, .18);
}

.hero {
    padding: 34px 34px;
    border-radius: 30px;
    border: 1px solid rgba(187, 247, 208, .18);
    background:
        linear-gradient(135deg, rgba(34, 197, 94, .22), rgba(20, 184, 166, .10)),
        rgba(255, 255, 255, .05);
    box-shadow: 0 24px 90px rgba(0, 0, 0, .38);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 48px;
    line-height: 1.04;
    margin: 0 0 12px 0;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f0fdf4;
}

.hero p {
    font-size: 16px;
    margin: 0;
    color: #bbf7d0;
    max-width: 980px;
}

.badge {
    display: inline-flex;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(34, 197, 94, .16);
    color: #bbf7d0;
    border: 1px solid rgba(134, 239, 172, .30);
    font-weight: 800;
    margin-bottom: 14px;
    font-size: 13px;
}

.metric-card {
    padding: 22px;
    border-radius: 24px;
    background: rgba(255, 255, 255, .065);
    border: 1px solid rgba(187, 247, 208, .16);
    box-shadow: 0 18px 55px rgba(0, 0, 0, .24);
    min-height: 128px;
}

.metric-label {
    color: #bbf7d0;
    font-size: 13px;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    color: #f0fdf4;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.metric-caption {
    color: #86efac;
    font-size: 12px;
    margin-top: 7px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin: 28px 0 12px 0;
    color: #f0fdf4;
}

.card {
    border-radius: 24px;
    background: rgba(255, 255, 255, .055);
    border: 1px solid rgba(187, 247, 208, .15);
    padding: 18px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, .20);
}

.stButton>button {
    border-radius: 14px;
    border: 1px solid rgba(134, 239, 172, .35);
    background: linear-gradient(135deg, #22c55e, #14b8a6);
    color: #052e1a;
    font-weight: 800;
}

.stDownloadButton>button {
    border-radius: 14px;
    border: 1px solid rgba(134, 239, 172, .35);
    background: rgba(255, 255, 255, .08);
    color: #dcfce7;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

hr {
    border: none;
    border-top: 1px solid rgba(187, 247, 208, .14);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNGSI BANTUAN
# =========================================================
def parse_number(value):
    """Mengubah angka format Indonesia ke float."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    text = text.replace("\ufeff", "")
    text = text.replace(" ", "")

    # Format Indonesia: 1.234,56 -> 1234.56
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        # Kalau hanya ada titik, cek apakah titik ribuan atau desimal
        parts = text.split(".")
        if len(parts) > 2:
            text = text.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) == 3:
            text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def read_any_file(file):
    """Membaca CSV atau Excel."""
    if file is None:
        if not DATA_PATH.exists():
            st.error("File data bawaan tidak ditemukan. Upload file CSV/Excel dulu.")
            st.stop()

        return pd.read_csv(DATA_PATH, sep=None, engine="python", encoding="utf-8-sig")

    name = file.name.lower()

    if name.endswith(".csv"):
        raw = file.read()
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    st.error("Format file belum didukung. Pakai CSV, XLSX, atau XLS.")
    st.stop()


@st.cache_data
def load_default_data():
    return read_any_file(None)


def clean_data(df):
    """Membersihkan nama kolom dan tipe data."""
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    column_map = {}
    for c in df.columns:
        low = c.lower().strip()

        if low == "tahun":
            column_map[c] = "Tahun"
        elif "provinsi" in low:
            column_map[c] = "Provinsi"
        elif "kabupaten" in low or "kota" in low:
            column_map[c] = "Kabupaten/Kota"
        elif "harian" in low:
            column_map[c] = "Timbulan Sampah Harian(ton)"
        elif "tahunan" in low:
            column_map[c] = "Timbulan Sampah Tahunan(ton)"

    df = df.rename(columns=column_map)

    required = [
        "Tahun",
        "Provinsi",
        "Kabupaten/Kota",
        "Timbulan Sampah Harian(ton)",
        "Timbulan Sampah Tahunan(ton)"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom belum sesuai: {missing}")
        st.info("Kolom wajib: Tahun, Provinsi, Kabupaten/Kota, Timbulan Sampah Harian(ton), Timbulan Sampah Tahunan(ton)")
        st.stop()

    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df["Timbulan Sampah Harian(ton)"] = df["Timbulan Sampah Harian(ton)"].apply(parse_number)
    df["Timbulan Sampah Tahunan(ton)"] = df["Timbulan Sampah Tahunan(ton)"].apply(parse_number)

    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].astype(str).str.strip()

    df = df.dropna(subset=[
        "Tahun",
        "Provinsi",
        "Kabupaten/Kota",
        "Timbulan Sampah Tahunan(ton)"
    ])

    df["Tahun"] = df["Tahun"].astype(int)

    # Filter Jawa Barat kalau kolom provinsi tersedia
    df = df[df["Provinsi"].str.lower().str.contains("jawa barat", na=False)]

    df = df.sort_values(["Kabupaten/Kota", "Tahun"]).reset_index(drop=True)

    return df


def format_angka(value):
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def interpretasi_model(mae, rmse, r2, skala_rata2):
    """Menyusun narasi otomatis seberapa akurat model, berdasarkan MAE, RMSE, R²."""

    # --- Kategori R² ---
    if r2 >= 0.85:
        label_r2 = "sangat baik"
        emoji_r2 = "🟢"
    elif r2 >= 0.65:
        label_r2 = "cukup baik"
        emoji_r2 = "🟡"
    elif r2 >= 0.30:
        label_r2 = "kurang optimal"
        emoji_r2 = "🟠"
    else:
        label_r2 = "lemah"
        emoji_r2 = "🔴"

    # --- MAE relatif terhadap skala rata-rata data (dalam %) ---
    if skala_rata2 and skala_rata2 != 0:
        mae_persen = (mae / skala_rata2) * 100
    else:
        mae_persen = 0.0

    if mae_persen <= 5:
        label_mae = "sangat kecil"
    elif mae_persen <= 15:
        label_mae = "relatif kecil"
    elif mae_persen <= 30:
        label_mae = "cukup besar"
    else:
        label_mae = "besar"

    # --- Selisih RMSE vs MAE, indikasi outlier ---
    if mae > 0:
        rasio_rmse_mae = rmse / mae
    else:
        rasio_rmse_mae = 1.0

    if rasio_rmse_mae <= 1.15:
        catatan_outlier = "error tersebar cukup merata di semua titik data (tidak banyak prediksi yang meleset ekstrem)."
    elif rasio_rmse_mae <= 1.5:
        catatan_outlier = "ada beberapa titik data dengan selisih prediksi yang lebih besar dari rata-rata."
    else:
        catatan_outlier = "terdapat beberapa titik data dengan selisih prediksi yang jauh lebih besar dibanding titik lainnya (outlier)."

    kesimpulan = (
        f"{emoji_r2} **Model tergolong {label_r2}** untuk menjelaskan pola data historis "
        f"(R² = {r2:.4f}). Rata-rata selisih prediksi terhadap data aktual (MAE) adalah "
        f"**{format_angka(mae)} ton/tahun**, yang tergolong **{label_mae}** "
        f"(~{mae_persen:.2f}% dari rata-rata volume sampah tahunan). "
        f"RMSE tercatat **{format_angka(rmse)} ton/tahun** — {catatan_outlier}"
    )

    if r2 >= 0.65 and mae_persen <= 15:
        rekomendasi = (
            "Secara umum, hasil prediksi pada grafik di atas cukup bisa diandalkan sebagai "
            "estimasi, meski tetap perlu dibaca sebagai perkiraan, bukan angka pasti."
        )
    else:
        rekomendasi = (
            "Karena akurasi model masih terbatas, sebaiknya anggap garis prediksi pada grafik "
            "sebagai estimasi kasar. Menambah jumlah data historis atau fitur lain bisa membantu "
            "meningkatkan akurasi model."
        )

    return kesimpulan + " " + rekomendasi, label_r2


def make_features(df):
    """Membuat fitur model dari data historis."""
    data = df.copy()

    le_city = LabelEncoder()
    le_prov = LabelEncoder()

    data["city_code"] = le_city.fit_transform(data["Kabupaten/Kota"])
    data["prov_code"] = le_prov.fit_transform(data["Provinsi"])

    data["lag_1"] = data.groupby("Kabupaten/Kota")["Timbulan Sampah Tahunan(ton)"].shift(1)
    data["lag_2"] = data.groupby("Kabupaten/Kota")["Timbulan Sampah Tahunan(ton)"].shift(2)

    data["rolling_3"] = (
        data.groupby("Kabupaten/Kota")["Timbulan Sampah Tahunan(ton)"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    global_mean = data["Timbulan Sampah Tahunan(ton)"].mean()

    data["lag_1"] = data["lag_1"].fillna(global_mean)
    data["lag_2"] = data["lag_2"].fillna(global_mean)
    data["rolling_3"] = data["rolling_3"].fillna(global_mean)

    features = ["Tahun", "city_code", "prov_code", "lag_1", "lag_2", "rolling_3"]
    target = "Timbulan Sampah Tahunan(ton)"

    return data, features, target, le_city, le_prov


@st.cache_resource
def train_model_cached(df_json):
    df = pd.read_json(io.StringIO(df_json))
    return train_model(df)


def train_model(df):
    data, features, target, le_city, le_prov = make_features(df)

    X = data[features]
    y = data[target]

    if len(data) >= 12:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.22,
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=9,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    if len(y_test) > 1:
        r2 = float(r2_score(y_test, y_pred))
    else:
        r2 = 0.0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, data, features, le_city, le_prov, metrics


def forecast_city(model, df, features, le_city, le_prov, city, start_year, end_year):
    """Prediksi per wilayah untuk beberapa tahun ke depan."""
    history = df[df["Kabupaten/Kota"] == city].sort_values("Tahun").copy()

    if history.empty:
        return pd.DataFrame()

    prov = history["Provinsi"].iloc[-1]
    city_code = int(le_city.transform([city])[0])
    prov_code = int(le_prov.transform([prov])[0])

    annual_values = history["Timbulan Sampah Tahunan(ton)"].tolist()

    rows = []

    for year in range(start_year, end_year + 1):
        lag_1 = annual_values[-1] if len(annual_values) >= 1 else np.mean(annual_values)
        lag_2 = annual_values[-2] if len(annual_values) >= 2 else lag_1
        rolling_3 = np.mean(annual_values[-3:]) if len(annual_values) >= 1 else lag_1

        x_new = pd.DataFrame([{
            "Tahun": year,
            "city_code": city_code,
            "prov_code": prov_code,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "rolling_3": rolling_3
        }])[features]

        annual_pred = float(model.predict(x_new)[0])
        daily_pred = annual_pred / 365

        annual_values.append(annual_pred)

        rows.append({
            "Tahun": year,
            "Provinsi": prov,
            "Kabupaten/Kota": city,
            "Prediksi Timbulan Sampah Harian(ton)": round(daily_pred, 2),
            "Prediksi Timbulan Sampah Tahunan(ton)": round(annual_pred, 2)
        })

    return pd.DataFrame(rows)


def forecast_all_cities(model, df, features, le_city, le_prov, year):
    rows = []
    for city in sorted(df["Kabupaten/Kota"].unique()):
        result = forecast_city(model, df, features, le_city, le_prov, city, year, year)
        if not result.empty:
            rows.append(result)

    if rows:
        return pd.concat(rows, ignore_index=True)

    return pd.DataFrame()


# =========================================================
# SIDEBAR INPUT
# =========================================================
st.sidebar.title("♻️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload data CSV/Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    raw_df = read_any_file(uploaded_file)
else:
    raw_df = load_default_data()

df = clean_data(raw_df)

if df.empty:
    st.error("Data Jawa Barat tidak ditemukan setelah proses filter.")
    st.stop()

years = sorted(df["Tahun"].unique())
cities = sorted(df["Kabupaten/Kota"].unique())

selected_city = st.sidebar.selectbox(
    "Pilih Kabupaten/Kota",
    cities,
    index=cities.index("Kota Bandung") if "Kota Bandung" in cities else 0
)

default_start_year = 2025
min_start_year = min(int(max(years)), default_start_year)

start_pred = st.sidebar.number_input(
    "Mulai tahun prediksi",
    min_value=min_start_year,
    max_value=int(max(years)) + 20,
    value=default_start_year
)

end_pred = st.sidebar.number_input(
    "Sampai tahun prediksi",
    min_value=int(start_pred),
    max_value=int(max(years)) + 20,
    value=default_start_year + 4
)

ranking_year = st.sidebar.selectbox(
    "Tahun ranking historis",
    years,
    index=len(years) - 1
)


# =========================================================
# TRAIN MODEL
# =========================================================
model, train_df, features, le_city, le_prov, metrics = train_model_cached(
    df.to_json(orient="records")
)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Prediksi Volume Sampah<br>Provinsi Jawa Barat</h1>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRIK UTAMA
# =========================================================
total_tahunan = df["Timbulan Sampah Tahunan(ton)"].sum()
rata_harian = df["Timbulan Sampah Harian(ton)"].mean()
latest_year = int(df["Tahun"].max())
total_latest_year = df[df["Tahun"] == latest_year]["Timbulan Sampah Tahunan(ton)"].sum()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Historis</div>
        <div class="metric-value">{format_angka(total_tahunan)}</div>
        <div class="metric-caption">ton/tahun kumulatif</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rata-rata Harian</div>
        <div class="metric-value">{format_angka(rata_harian)}</div>
        <div class="metric-caption">ton/hari</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Tahun {latest_year}</div>
        <div class="metric-value">{format_angka(total_latest_year)}</div>
        <div class="metric-caption">seluruh wilayah tersedia</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Nilai R² Model</div>
        <div class="metric-value">{metrics["R2"]:.3f}</div>
        <div class="metric-caption">testing split</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB DASHBOARD
# =========================================================
tab2, tab3, tab4 = st.tabs([
    "🔮 Prediksi",
    "🏙️ Ranking Wilayah",
    "📄 Dataset"
])


# =========================================================
# TAB PREDIKSI
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Prediksi Volume Sampah Tahun Berikutnya</div>', unsafe_allow_html=True)

    pred_df = forecast_city(
        model=model,
        df=df,
        features=features,
        le_city=le_city,
        le_prov=le_prov,
        city=selected_city,
        start_year=int(start_pred),
        end_year=int(end_pred)
    )

    # ---------------------------------------------------
    # GRAFIK GABUNGAN: data AKTUAL (historis) bersandingan
    # dengan garis PREDIKSI di satu chart yang sama
    # ---------------------------------------------------
    actual_for_chart = (
        df[df["Kabupaten/Kota"] == selected_city]
        .sort_values("Tahun")[["Tahun", "Timbulan Sampah Tahunan(ton)"]]
        .rename(columns={"Timbulan Sampah Tahunan(ton)": "Volume"})
    )

    prediction_for_chart = pred_df[[
        "Tahun",
        "Prediksi Timbulan Sampah Tahunan(ton)"
    ]].rename(columns={"Prediksi Timbulan Sampah Tahunan(ton)": "Volume"})

    # Sambungkan titik terakhir data aktual ke titik pertama prediksi
    # supaya garis prediksi terlihat bersambung (bukan terputus)
    if not actual_for_chart.empty and not prediction_for_chart.empty:
        last_actual_point = actual_for_chart.tail(1)
        prediction_line_df = pd.concat([last_actual_point, prediction_for_chart], ignore_index=True)
    else:
        prediction_line_df = prediction_for_chart

    fig_pred = go.Figure()

    # Area bayangan tipis di sepanjang zona prediksi, biar langsung
    # kelihatan mana rentang tahun yang sifatnya perkiraan/estimasi
    if not prediction_line_df.empty:
        batas_bawah_shade = prediction_line_df["Tahun"].min()
        batas_atas_shade = prediction_line_df["Tahun"].max()
        fig_pred.add_vrect(
            x0=batas_bawah_shade,
            x1=batas_atas_shade,
            fillcolor="#f97316",
            opacity=0.06,
            line_width=0,
            layer="below"
        )

    fig_pred.add_trace(go.Scatter(
        x=actual_for_chart["Tahun"],
        y=actual_for_chart["Volume"],
        mode="lines+markers+text",
        name="Data Aktual",
        line=dict(width=4, color="#22c55e"),
        marker=dict(size=9, color="#22c55e", line=dict(width=1.5, color="#052e1a")),
        text=[format_angka(v) for v in actual_for_chart["Volume"]],
        textposition="top center",
        textfont=dict(size=11, color="#bbf7d0"),
        hovertemplate="Tahun %{x}<br>Aktual: %{y:,.2f} ton<extra></extra>"
    ))

    fig_pred.add_trace(go.Scatter(
        x=prediction_line_df["Tahun"],
        y=prediction_line_df["Volume"],
        mode="lines+markers+text",
        name="Prediksi",
        line=dict(width=4, color="#f97316", dash="dash"),
        marker=dict(size=10, color="#f97316", symbol="diamond", line=dict(width=1.5, color="#3b1a04")),
        text=[format_angka(v) for v in prediction_line_df["Volume"]],
        textposition="bottom center",
        textfont=dict(size=11, color="#fed7aa"),
        hovertemplate="Tahun %{x}<br>Prediksi: %{y:,.2f} ton<extra></extra>"
    ))

    # Garis vertikal penanda batas: sebelum garis ini = data aktual,
    # sesudah garis ini = mulai wilayah prediksi
    if not actual_for_chart.empty and not prediction_for_chart.empty:
        batas_tahun = float(actual_for_chart["Tahun"].max())
        fig_pred.add_vline(
            x=batas_tahun,
            line_width=2,
            line_dash="dot",
            line_color="rgba(187,247,208,.55)",
            annotation_text="Mulai Prediksi",
            annotation_position="top",
            annotation_font=dict(size=12, color="#bbf7d0")
        )

    fig_pred.update_layout(
        title=dict(
            text=f"Data Aktual vs Prediksi - {selected_city}",
            font=dict(size=20)
        ),
        xaxis_title="Tahun",
        yaxis_title="Ton/Tahun",
        template="plotly_dark",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.06,
            xanchor="right", x=1,
            font=dict(size=13)
        ),
        xaxis=dict(
            dtick=1,
            showgrid=True,
            gridcolor="rgba(187,247,208,.08)"
        ),
        yaxis=dict(
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(187,247,208,.10)"
        )
    )
    st.plotly_chart(fig_pred, width="stretch")

    st.caption(
        "🟢 Garis hijau = data timbulan sampah **aktual** (historis). "
        "🟠 Garis oranye putus-putus = hasil **prediksi** model untuk tahun-tahun ke depan. "
        "Garis putus-putus vertikal menandai batas awal tahun prediksi, dan area oranye pudar "
        "menunjukkan rentang tahun yang diprediksi."
    )

    # ---------------------------------------------------
    # TABEL SANDING: Tahun | Data Aktual | Data Prediksi
    # digabung jadi satu tabel supaya angka aktual dan
    # prediksi mudah dibandingkan berdampingan per tahun
    # ---------------------------------------------------
    st.markdown("Tabel sanding: data aktual vs prediksi")

    tabel_aktual = actual_for_chart.rename(columns={"Volume": "Data Aktual (ton)"})
    tabel_prediksi = prediction_for_chart.rename(columns={"Volume": "Data Prediksi (ton)"})

    tabel_sanding = pd.merge(
        tabel_aktual,
        tabel_prediksi,
        on="Tahun",
        how="outer"
    ).sort_values("Tahun").reset_index(drop=True)

    tabel_sanding["Tahun"] = tabel_sanding["Tahun"].astype(int)
    tabel_sanding["Keterangan"] = np.where(
        tabel_sanding["Data Prediksi (ton)"].notna(),
        "Prediksi",
        "Aktual"
    )

    tabel_sanding_tampil = tabel_sanding.copy()
    tabel_sanding_tampil["Data Aktual (ton)"] = tabel_sanding_tampil["Data Aktual (ton)"].apply(
        lambda v: format_angka(v) if pd.notna(v) else "-"
    )
    tabel_sanding_tampil["Data Prediksi (ton)"] = tabel_sanding_tampil["Data Prediksi (ton)"].apply(
        lambda v: format_angka(v) if pd.notna(v) else "-"
    )

    st.dataframe(
        tabel_sanding_tampil[["Tahun", "Data Aktual (ton)", "Data Prediksi (ton)", "Keterangan"]],
        width="stretch",
        hide_index=True
    )

    csv_sanding = tabel_sanding[["Tahun", "Data Aktual (ton)", "Data Prediksi (ton)", "Keterangan"]].to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        "Download tabel sanding aktual vs prediksi CSV",
        data=csv_sanding,
        file_name=f"sanding_aktual_prediksi_{selected_city.replace(' ', '_')}.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------
    # GRAFIK KHUSUS: Data Aktual 2025 vs Data Prediksi 2026
    # (tetap tampil apa pun pengaturan rentang tahun di sidebar)
    # ---------------------------------------------------
    st.markdown("### Perbandingan Data Aktual 2025 vs Data Prediksi 2026")

    TAHUN_AKTUAL_TETAP = 2025
    TAHUN_PREDIKSI_TETAP = 2026

    baris_aktual_2025 = df[
        (df["Kabupaten/Kota"] == selected_city) &
        (df["Tahun"] == TAHUN_AKTUAL_TETAP)
    ]

    if baris_aktual_2025.empty:
        st.warning(f"Data aktual tahun {TAHUN_AKTUAL_TETAP} untuk {selected_city} tidak tersedia di dataset.")
    else:
        nilai_aktual_2025 = float(baris_aktual_2025["Timbulan Sampah Tahunan(ton)"].iloc[0])

        pred_2026_df = forecast_city(
            model=model,
            df=df,
            features=features,
            le_city=le_city,
            le_prov=le_prov,
            city=selected_city,
            start_year=TAHUN_PREDIKSI_TETAP,
            end_year=TAHUN_PREDIKSI_TETAP
        )
        nilai_prediksi_2026 = float(pred_2026_df["Prediksi Timbulan Sampah Tahunan(ton)"].iloc[0])

        selisih_2526 = nilai_prediksi_2026 - nilai_aktual_2025
        persen_2526 = (selisih_2526 / nilai_aktual_2025) * 100 if nilai_aktual_2025 != 0 else 0
        arah_2526 = "naik" if selisih_2526 > 0 else ("turun" if selisih_2526 < 0 else "tetap")

        fig_2025_2026 = go.Figure()

        fig_2025_2026.add_trace(go.Bar(
            x=[f"Aktual {TAHUN_AKTUAL_TETAP}"],
            y=[nilai_aktual_2025],
            name="Data Aktual",
            marker_color="#22c55e",
            text=[format_angka(nilai_aktual_2025)],
            textposition="outside",
            width=0.45
        ))

        fig_2025_2026.add_trace(go.Bar(
            x=[f"Prediksi {TAHUN_PREDIKSI_TETAP}"],
            y=[nilai_prediksi_2026],
            name="Data Prediksi",
            marker_color="#f97316",
            text=[format_angka(nilai_prediksi_2026)],
            textposition="outside",
            width=0.45
        ))

        fig_2025_2026.update_layout(
            title=f"Data Aktual {TAHUN_AKTUAL_TETAP} vs Data Prediksi {TAHUN_PREDIKSI_TETAP} - {selected_city}",
            yaxis_title="Ton/Tahun",
            template="plotly_dark",
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            yaxis=dict(tickformat=",.0f", showgrid=True, gridcolor="rgba(187,247,208,.10)")
        )
        st.plotly_chart(fig_2025_2026, width="stretch")

        k2025, k2026, kstatus = st.columns(3)

        with k2025:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Aktual {TAHUN_AKTUAL_TETAP}</div>
                <div class="metric-value">{format_angka(nilai_aktual_2025)}</div>
                <div class="metric-caption">ton/tahun</div>
            </div>
            """, unsafe_allow_html=True)

        with k2026:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Prediksi {TAHUN_PREDIKSI_TETAP}</div>
                <div class="metric-value">{format_angka(nilai_prediksi_2026)}</div>
                <div class="metric-caption">ton/tahun</div>
            </div>
            """, unsafe_allow_html=True)

        with kstatus:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value">{arah_2526.upper()}</div>
                <div class="metric-caption">Selisih {format_angka(abs(selisih_2526))} ton ({persen_2526:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # ERROR / EVALUASI MODEL - ditampilkan langsung
    # di bawah grafik prediksi supaya mudah dibaca
    # ---------------------------------------------------
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE (Mean Absolute Error)</div>
            <div class="metric-value">{format_angka(metrics["MAE"])}</div>
            <div class="metric-caption">rata-rata selisih absolut, ton/tahun</div>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE (Root Mean Squared Error)</div>
            <div class="metric-value">{format_angka(metrics["RMSE"])}</div>
            <div class="metric-caption">akar rata-rata kuadrat error, ton/tahun</div>
        </div>
        """, unsafe_allow_html=True)

    with e3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metrics["R2"]:.4f}</div>
            <div class="metric-caption">semakin dekat ke 1 semakin akurat</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "MAE dan RMSE dihitung dari data uji (test split) hasil historis, semakin kecil nilainya "
        "semakin akurat model. R² menunjukkan seberapa baik model menjelaskan variasi data "
        "(1.0 = sempurna, mendekati 0 atau negatif = model kurang cocok)."
    )

    # ---------------------------------------------------
    # PENJELASAN OTOMATIS: narasi kualitas model berdasarkan
    # nilai MAE, RMSE, R² yang sudah dihitung di atas
    # ---------------------------------------------------
    skala_rata2_kota = actual_for_chart["Volume"].mean() if not actual_for_chart.empty else df["Timbulan Sampah Tahunan(ton)"].mean()

    narasi_error, label_kualitas = interpretasi_model(
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        r2=metrics["R2"],
        skala_rata2=skala_rata2_kota
    )

    st.markdown("Penjelasan hasil error prediksi vs aktual")

    if label_kualitas in ["sangat baik", "cukup baik"]:
        st.success(narasi_error)
    elif label_kualitas == "kurang optimal":
        st.warning(narasi_error)
    else:
        st.error(narasi_error)

    p1, p2 = st.columns([1.2, .8])

    with p1:
        st.markdown("Hasil prediksi wilayah terpilih")
        st.dataframe(pred_df, width="stretch", hide_index=True)

        csv_pred = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediksi wilayah CSV",
            data=csv_pred,
            file_name=f"prediksi_{selected_city.replace(' ', '_')}_{int(start_pred)}_{int(end_pred)}.csv",
            mime="text/csv"
        )

    with p2:
        st.markdown("Detail evaluasi model")
        eval_df = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "R²"],
            "Nilai": [
                round(metrics["MAE"], 2),
                round(metrics["RMSE"], 2),
                round(metrics["R2"], 4)
            ]
        })
        st.dataframe(eval_df, width="stretch", hide_index=True)

        st.info(
            "Model memakai fitur tahun, kode wilayah, lag tahun sebelumnya, "
            "lag dua tahun sebelumnya, dan rolling mean tiga tahun."
        )

    # =====================================================
    # PERBANDINGAN PREDIKSI DENGAN TAHUN SEBELUMNYA
    # =====================================================
    st.markdown("### Perbandingan prediksi dengan tahun sebelumnya")

    prev_year = int(start_pred) - 1
    prev_data = df[
        (df["Kabupaten/Kota"] == selected_city) &
        (df["Tahun"] == prev_year)
    ]

    if not prev_data.empty and not pred_df.empty:
        prev_value = float(prev_data["Timbulan Sampah Tahunan(ton)"].iloc[0])
        pred_value = float(pred_df[pred_df["Tahun"] == int(start_pred)]["Prediksi Timbulan Sampah Tahunan(ton)"].iloc[0])
        selisih = pred_value - prev_value
        persen = (selisih / prev_value) * 100 if prev_value != 0 else 0

        if selisih > 0:
            status = "Lebih besar dari tahun sebelumnya"
            arah = "naik"
        elif selisih < 0:
            status = "Lebih sedikit dari tahun sebelumnya"
            arah = "turun"
        else:
            status = "Sama dengan tahun sebelumnya"
            arah = "tetap"

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tahun Sebelumnya ({prev_year})</div>
                <div class="metric-value">{format_angka(prev_value)}</div>
                <div class="metric-caption">ton/tahun</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediksi Tahun {int(start_pred)}</div>
                <div class="metric-value">{format_angka(pred_value)}</div>
                <div class="metric-caption">ton/tahun</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Status Prediksi</div>
                <div class="metric-value">{arah.upper()}</div>
                <div class="metric-caption">{status}<br>Selisih {format_angka(abs(selisih))} ton ({persen:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        compare_df = pd.DataFrame({
            "Kategori": [f"Realisasi {prev_year}", f"Prediksi {int(start_pred)}"],
            "Volume Sampah (ton/tahun)": [prev_value, pred_value]
        })

        fig_compare = px.bar(
            compare_df,
            x="Kategori",
            y="Volume Sampah (ton/tahun)",
            text="Volume Sampah (ton/tahun)",
            title=f"Perbandingan Realisasi {prev_year} dan Prediksi {int(start_pred)} - {selected_city}"
        )
        fig_compare.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside"
        )
        fig_compare.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_compare, width="stretch")

        st.success(
            f"Hasil prediksi volume sampah {selected_city} tahun {int(start_pred)} adalah "
            f"{format_angka(pred_value)} ton/tahun. Nilai ini {status.lower()} "
            f"dengan selisih {format_angka(abs(selisih))} ton atau {persen:.2f}%."
        )
    else:
        st.warning("Data tahun sebelumnya belum tersedia, jadi perbandingan belum bisa dihitung.")

    st.markdown("Prediksi semua kabupaten/kota")
    all_city_pred = forecast_all_cities(
        model=model,
        df=df,
        features=features,
        le_city=le_city,
        le_prov=le_prov,
        year=int(start_pred)
    ).sort_values("Prediksi Timbulan Sampah Tahunan(ton)", ascending=False)

    # Tambahkan kolom perbandingan untuk semua wilayah
    compare_rows = []
    tahun_sebelumnya = int(start_pred) - 1

    for _, row in all_city_pred.iterrows():
        city_name = row["Kabupaten/Kota"]
        pred_total = float(row["Prediksi Timbulan Sampah Tahunan(ton)"])

        prev_row = df[
            (df["Kabupaten/Kota"] == city_name) &
            (df["Tahun"] == tahun_sebelumnya)
        ]

        if not prev_row.empty:
            realisasi_lalu = float(prev_row["Timbulan Sampah Tahunan(ton)"].iloc[0])
            beda = pred_total - realisasi_lalu
            beda_persen = (beda / realisasi_lalu) * 100 if realisasi_lalu != 0 else 0

            if beda > 0:
                status_beda = "Lebih besar"
            elif beda < 0:
                status_beda = "Lebih sedikit"
            else:
                status_beda = "Sama"
        else:
            realisasi_lalu = float("nan")
            beda = float("nan")
            beda_persen = float("nan")
            status_beda = "Data pembanding tidak ada"

        compare_rows.append({
            "Realisasi Tahun Sebelumnya(ton)": round(realisasi_lalu, 2) if not pd.isna(realisasi_lalu) else np.nan,
            "Selisih(ton)": round(beda, 2) if not pd.isna(beda) else np.nan,
            "Perubahan(%)": round(beda_persen, 2) if not pd.isna(beda_persen) else np.nan,
            "Status": status_beda
        })

    compare_all_df = pd.concat([all_city_pred.reset_index(drop=True), pd.DataFrame(compare_rows)], axis=1)

    st.dataframe(compare_all_df, width="stretch", hide_index=True)

    csv_all = compare_all_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediksi semua wilayah CSV",
        data=csv_all,
        file_name=f"prediksi_semua_wilayah_{int(start_pred)}.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 3 RANKING
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Ranking Volume Sampah Jawa Barat</div>', unsafe_allow_html=True)

    rank_df = (
        df[df["Tahun"] == ranking_year]
        .sort_values("Timbulan Sampah Tahunan(ton)", ascending=False)
        .head(15)
    )

    fig_rank = px.bar(
        rank_df,
        x="Timbulan Sampah Tahunan(ton)",
        y="Kabupaten/Kota",
        orientation="h",
        title=f"15 Wilayah dengan Timbulan Sampah Tertinggi Tahun {ranking_year}",
        labels={"Timbulan Sampah Tahunan(ton)": "Ton/Tahun"}
    )
    fig_rank.update_layout(
        template="plotly_dark",
        height=580,
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_rank, width="stretch")

    st.dataframe(rank_df, width="stretch", hide_index=True)


# =========================================================
# TAB 4 DATASET
# =========================================================
with tab4:
    st.markdown('<div class="section-title">Dataset SIPSN KLHK 2019–2025</div>', unsafe_allow_html=True)

    st.write("Jumlah baris:", len(df))
    st.write("Rentang tahun:", int(df["Tahun"].min()), "-", int(df["Tahun"].max()))
    st.write("Jumlah kabupaten/kota:", df["Kabupaten/Kota"].nunique())

    st.dataframe(df, width="stretch", hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download dataset bersih CSV",
        data=csv_data,
        file_name="dataset_sampah_jabar_bersih.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>
<div style="text-align:center;color:#86efac;font-size:13px;padding:16px;">
    Prediksi Volume Sampah Jawa Barat • Streamlit • Random Forest Regressor
</div>
""", unsafe_allow_html=True)
