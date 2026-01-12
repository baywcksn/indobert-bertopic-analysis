import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import altair as alt

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# BERTopic stack
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Evaluasi metrik
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="IndoBERT & BERTopic Analysis",
    page_icon="📊",
    layout="wide",
)

# ======================================================
# SESSION STATE (AGAR HASIL TIDAK RESET)
# ======================================================
if "df_labeled" not in st.session_state:
    st.session_state.df_labeled = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "text_column" not in st.session_state:
    st.session_state.text_column = None

# Simpan hasil BERTopic per sentimen
if "bertopic_results" not in st.session_state:
    st.session_state.bertopic_results = {}

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## 📌 Informasi Aplikasi")
    st.markdown(
        """
        **Model Sentimen**: IndoBERT  
        **Analisis Topik**: BERTopic  

        **Klasifikasi Sentimen**
        - Negatif  
        - Positif  

        Aplikasi ini mendemonstrasikan
        **analisis sentimen dan topik**
        pada data teks tanpa label
        untuk kebutuhan akademik.
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
        padding: 1.8rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0e1117, #151a22);
        border: 1px solid #262730;
    ">
        <h2 style="margin:0;">📊 Analisis Sentimen & Topik Teks</h2>
        <p style="color:#b0b3b8; font-size:0.95rem; margin-top:0.6rem;">
            Sistem ini melakukan <b>pelabelan sentimen otomatis</b>
            menggunakan IndoBERT dan <b>analisis topik dominan</b>
            menggunakan BERTopic untuk mengidentifikasi isu utama
            pada data teks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ======================================================
# LOAD MODEL INDOBERT (CACHE)
# ======================================================
MODEL_PATH = "./indobert-sentiment-tiktok"  # sesuaikan
device = torch.device("cpu")


@st.cache_resource
def load_indobert(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


with st.spinner("🔄 Memuat model IndoBERT..."):
    tokenizer, model = load_indobert(MODEL_PATH)
st.success("✅ Model IndoBERT siap digunakan")

label_map = {0: "Negatif", 1: "Positif"}


def predict_sentiment(text: str):
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
    return label_map[pred.item()], float(confidence.item())


# ======================================================
# LOAD MODEL BERTOPIC (CACHE)
# ======================================================
@st.cache_resource
def load_bertopic():
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
    )
    return topic_model


# ======================================================
# UPLOAD DATASET UTAMA (UNLABELED)
# ======================================================
st.markdown("## 📁 Upload Dataset Utama")
uploaded_file = st.file_uploader(
    "Upload file CSV (dataset tanpa label sentimen)",
    type=["csv"],
    key="main_dataset",
)

if uploaded_file is None:
    st.info("Silakan upload dataset CSV untuk memulai.")
    st.stop()

df = pd.read_csv(uploaded_file)

c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.markdown("### 🔍 Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)

with c2:
    st.markdown("### ⚙️ Pengaturan")
    text_column = st.selectbox("Pilih kolom teks", df.columns, key="text_col_main")
    st.session_state.text_column = text_column

st.markdown("---")

# ======================================================
# TOMBOL ANALISIS INDO BERT
# ======================================================
if st.button("🚀 Jalankan Analisis Sentimen (IndoBERT)", use_container_width=True):
    sentiment_labels = []
    confidence_scores = []
    progress_bar = st.progress(0)
    total = len(df)

    for i, text in enumerate(df[st.session_state.text_column]):
        label, conf = predict_sentiment(str(text))
        sentiment_labels.append(label)
        confidence_scores.append(conf)
        progress_bar.progress((i + 1) / total)

    df_result = df.copy()
    df_result["sentiment_label"] = sentiment_labels
    df_result["confidence"] = confidence_scores

    st.session_state.df_labeled = df_result
    st.session_state.analysis_done = True

    # Reset hasil BERTopic jika dataset diganti / analisis ulang
    st.session_state.bertopic_results = {}

    st.success("🎉 Analisis sentimen selesai dan disimpan!")

# ======================================================
# STOP JIKA BELUM INDO BERT
# ======================================================
if not st.session_state.analysis_done or st.session_state.df_labeled is None:
    st.warning("Klik tombol **Jalankan Analisis Sentimen (IndoBERT)** terlebih dahulu.")
    st.stop()

df_result = st.session_state.df_labeled

# ======================================================
# OUTPUT INDO BERT
# ======================================================
st.markdown("## 📊 Hasil Analisis Sentimen (IndoBERT)")
st.dataframe(df_result.head(), use_container_width=True)

# ======================================================
# DISTRIBUSI SENTIMEN (BAR + LABEL ANGKA + AXIS BOLD)
# ======================================================
st.markdown("### 📈 Distribusi Sentimen")

sentiment_df = df_result["sentiment_label"].value_counts().reset_index()
sentiment_df.columns = ["Sentimen", "Jumlah"]

bars = (
    alt.Chart(sentiment_df)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
    .encode(
        x=alt.X(
            "Sentimen:N",
            sort=["Positif", "Negatif"],  # 🔁 URUTAN DIBALIK
            axis=alt.Axis(
                labelAngle=0,
                title="Kategori Sentimen",
                labelFontSize=14,
                labelFontWeight="bold",
            ),
        ),
        y=alt.Y("Jumlah:Q", title="Jumlah Data"),
        color=alt.Color(
            "Sentimen:N",
            scale=alt.Scale(
                domain=["Positif", "Negatif"],  # 🔁 WARNA IKUT URUTAN
                range=["#2ecc71", "#e74c3c"],  # hijau = positif, merah = negatif
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Sentimen:N", title="Sentimen"),
            alt.Tooltip("Jumlah:Q", title="Jumlah Data"),
        ],
    )
)

labels = (
    alt.Chart(sentiment_df)
    .mark_text(
        dy=-8,
        size=14,
        color="white",
        fontWeight="bold",
    )
    .encode(
        x=alt.X("Sentimen:N", sort=["Positif", "Negatif"]),  # 🔁 SAMA DENGAN BAR
        y=alt.Y("Jumlah:Q"),
        text=alt.Text("Jumlah:Q"),
    )
)

chart = (bars + labels).properties(height=360)

st.altair_chart(chart, use_container_width=True)

# ======================================================
# DOWNLOAD HASIL INDO BERT
# ======================================================
result_csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Dataset Berlabel (IndoBERT)",
    data=result_csv,
    file_name="hasil_sentimen_indobert.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("---")

# ===============================
# PILIH SENTIMEN
# ===============================
st.markdown(
    """
    <div style="padding:1.2rem;border-radius:14px;
    background:linear-gradient(135deg,#0e1117,#151a22);
    border:1px solid #262730;">
        <h3>🎯 Pilih Sentimen untuk Analisis Topik</h3>
        <p style="color:#b0b3b8;">
            Analisis topik menggunakan BERTopic berdasarkan hasil IndoBERT
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_sentiment = st.radio(
    "Sentimen yang dianalisis:",
    ["Positif", "Negatif"],
    horizontal=True,
)

# ===============================
# SUBSET DATA
# ===============================
subset = df_result[df_result["sentiment_label"] == selected_sentiment].copy()
subset_texts = subset[st.session_state.text_column].dropna().astype(str).tolist()

if len(subset_texts) < 10:
    st.warning(
        f"Jumlah data sentimen {selected_sentiment} hanya {len(subset_texts)}. "
        "Disarankan ≥ 10 teks."
    )

# ===============================
# PANEL PENGATURAN
# ===============================
colA, colB, colC = st.columns([1.3, 1, 1])

with colA:
    st.metric("Jumlah Teks", len(subset_texts))

with colB:
    min_topic_size = st.number_input(
        "min_topic_size", min_value=2, max_value=200, value=15
    )

with colC:
    nr_topics_opt = st.selectbox("nr_topics", ["auto", 5, 10, 15, 20, 30], index=0)

run_topic = st.button("🚀 Jalankan BERTopic", use_container_width=True)

# ===============================
# PROSES BERTOPIC
# ===============================
if run_topic:
    if len(subset_texts) == 0:
        st.error("Tidak ada teks untuk diproses.")
        st.stop()

    with st.spinner("🔄 Memproses BERTopic..."):
        topic_model = load_bertopic()
        embedding_model = topic_model.embedding_model

        nr_topics_val = nr_topics_opt if nr_topics_opt != "auto" else "auto"

        model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            min_topic_size=int(min_topic_size),
            nr_topics=nr_topics_val,
            verbose=False,
        )

        topics, _ = model.fit_transform(subset_texts)

        df_topics = pd.DataFrame({"text": subset_texts, "topic": topics})

        topic_info = model.get_topic_info()

        # ===============================
        # TOP 3 TOPIK DOMINAN
        # ===============================
        top_topics = (
            df_topics["topic"]
            .value_counts()
            .drop(index=-1, errors="ignore")
            .head(3)
            .reset_index()
        )
        top_topics.columns = ["topic", "jumlah"]

        st.session_state.bertopic_results[selected_sentiment] = {
            "df_topics": df_topics,
            "topic_info": topic_info,
            "top_topics": top_topics,
        }

    st.success("✅ BERTopic berhasil dijalankan.")

# ===============================
# CEK HASIL
# ===============================
if selected_sentiment not in st.session_state.bertopic_results:
    st.info("Klik **Jalankan BERTopic** untuk melihat hasil.")
    st.stop()

df_topics = st.session_state.bertopic_results[selected_sentiment]["df_topics"]
topic_info = st.session_state.bertopic_results[selected_sentiment]["topic_info"]
top_topics = st.session_state.bertopic_results[selected_sentiment]["top_topics"]

badge_color = "#143a2a" if selected_sentiment == "Positif" else "#3a1f1f"

# ===============================
# TAMPILKAN TOP 3 TOPIK
# ===============================
st.markdown(
    f"""
    <div style="padding:1rem;border-radius:12px;
    background:{badge_color};border:1px solid #2a3b2f;">
        <b>🎯 Top 3 Topik Dominan ({selected_sentiment})</b>
        {''.join([f"<div>Topic {row.topic} (n={row.jumlah})</div>"
                  for _, row in top_topics.iterrows()])}
    </div>
    """,
    unsafe_allow_html=True,
)

# ===============================
# KATA KUNCI TOP 3 TOPIK
# ===============================
st.markdown("### 🔑 Kata Kunci Top 3 Topik")

topic_col = "Topic" if "Topic" in topic_info.columns else "topic"
repr_col = (
    "Representation" if "Representation" in topic_info.columns else "representation"
)

for _, row in top_topics.iterrows():
    st.markdown(f"**Topic {row.topic} (n={row.jumlah})**")
    info_row = topic_info[topic_info[topic_col] == row.topic]
    if not info_row.empty:
        st.dataframe(
            info_row[[repr_col]].reset_index(drop=True), use_container_width=True
        )

# ===============================
# DISTRIBUSI TOPIK (TOP 10)
# ===============================
st.markdown("### 📊 Distribusi Topik (Top 10)")

topic_counts = df_topics["topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Jumlah"]
topic_counts["Topic"] = topic_counts["Topic"].astype(int)

plot_df = topic_counts[topic_counts["Topic"] != -1].head(10)

chart = (
    alt.Chart(plot_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("Topic:O", sort="-y"),
        y="Jumlah:Q",
        tooltip=["Topic", "Jumlah"],
    )
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)

# ===============================
# DOWNLOAD HASIL
# ===============================
st.markdown("### ⬇️ Download Hasil BERTopic")

st.download_button(
    f"Download df_topics ({selected_sentiment})",
    df_topics.to_csv(index=False).encode("utf-8"),
    file_name=f"hasil_bertopic_{selected_sentiment.lower()}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.download_button(
    f"Download topic_info ({selected_sentiment})",
    topic_info.to_csv(index=False).encode("utf-8"),
    file_name=f"topic_info_{selected_sentiment.lower()}.csv",
    mime="text/csv",
    use_container_width=True,
)
