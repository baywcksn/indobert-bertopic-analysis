import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="IndoBERT Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## 📌 Informasi Aplikasi")
    st.markdown(
        """
        **Model**: IndoBERT  
        **Klasifikasi**: Sentimen Biner  

        - 0 → Negatif  
        - 1 → Positif  

        Aplikasi ini digunakan untuk melakukan
        pelabelan sentimen otomatis pada data teks
        tanpa label.
        """
    )
    st.markdown("---")
    st.markdown("👨‍🎓 **Demo Akademik / Skripsi**")

# ======================================================
# HEADER UTAMA
# ======================================================
st.markdown(
    """
    <div style="
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #0e1117;
        border: 1px solid #262730;
    ">
        <h2>📊 Analisis Sentimen Menggunakan IndoBERT</h2>
        <p style="color: #b0b3b8;">
            Sistem ini melakukan analisis sentimen otomatis
            pada dataset <b>tanpa label</b> menggunakan model IndoBERT
            hasil fine-tuning untuk klasifikasi sentimen
            <b>positif</b> dan <b>negatif</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ======================================================
# KONFIGURASI MODEL
# ======================================================
MODEL_PATH = "./indobert-sentiment-tiktok"
device = torch.device("cpu")  # aman & stabil

with st.spinner("🔄 Memuat model IndoBERT..."):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

st.success("✅ Model berhasil dimuat")

label_map = {
    0: "Negatif",
    1: "Positif",
}


# ======================================================
# FUNGSI PREDIKSI
# ======================================================
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return label_map[pred.item()], confidence.item()


# ======================================================
# UPLOAD DATASET
# ======================================================
st.markdown("### 📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload file CSV (dataset tanpa label sentimen)",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🔍 Preview Dataset")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.markdown("#### ⚙️ Pengaturan")
        text_column = st.selectbox(
            "Pilih kolom teks",
            df.columns,
        )

    st.markdown("---")

    # ==================================================
    # PROSES ANALISIS
    # ==================================================
    if st.button("🚀 Jalankan Analisis Sentimen", use_container_width=True):
        sentiment_labels = []
        confidence_scores = []

        progress_bar = st.progress(0)
        total = len(df)

        for i, text in enumerate(df[text_column]):
            label, conf = predict_sentiment(str(text))
            sentiment_labels.append(label)
            confidence_scores.append(conf)
            progress_bar.progress((i + 1) / total)

        df["sentiment_label"] = sentiment_labels
        df["confidence"] = confidence_scores

        st.success("🎉 Analisis sentimen selesai")

        # ==============================================
        # OUTPUT DATA
        # ==============================================
        st.markdown("### 📊 Hasil Analisis")
        st.dataframe(df.head(), use_container_width=True)

        # ==============================================
        # VISUALISASI (ALTair – SIMPLE & CLEAN)
        # ==============================================
        st.markdown("#### 📈 Distribusi Sentimen")

        sentiment_df = df["sentiment_label"].value_counts().reset_index()
        sentiment_df.columns = ["Sentimen", "Jumlah"]

        chart = (
            alt.Chart(sentiment_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Sentimen:N",
                    sort=["Negatif", "Positif"],
                    axis=alt.Axis(
                        labelAngle=0,
                        title="Kategori Sentimen",
                    ),
                ),
                y=alt.Y(
                    "Jumlah:Q",
                    title="Jumlah Data",
                ),
                color=alt.Color(
                    "Sentimen:N",
                    scale=alt.Scale(
                        domain=["Negatif", "Positif"],
                        range=["#e74c3c", "#2ecc71"],  # merah & hijau
                    ),
                    legend=None,
                ),
                tooltip=["Sentimen", "Jumlah"],
            )
            .properties(height=400)
        )

        st.altair_chart(chart, use_container_width=True)

        # ==============================================
        # DOWNLOAD
        # ==============================================
        result_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Dataset Berlabel",
            data=result_csv,
            file_name="hasil_sentimen_indobert.csv",
            mime="text/csv",
            use_container_width=True,
        )
