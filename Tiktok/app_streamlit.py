import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import altair as alt

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# BERTopic stack
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


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
    # format:
    # st.session_state.bertopic_results["Negatif"] = {"df_topics":..., "topic_info":..., "top_topic":..., "top_n":...}
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
    # Embedding multilingual (bagus untuk Indonesia)
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
# UPLOAD DATASET
# ======================================================
st.markdown("## 📁 Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload file CSV (dataset tanpa label sentimen)",
    type=["csv"],
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
    text_column = st.selectbox("Pilih kolom teks", df.columns)
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
# DISTRIBUSI SENTIMEN
# ======================================================
st.markdown("### 📈 Distribusi Sentimen")

sentiment_df = df_result["sentiment_label"].value_counts().reset_index()
sentiment_df.columns = ["Sentimen", "Jumlah"]

chart = (
    alt.Chart(sentiment_df)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
    .encode(
        x=alt.X(
            "Sentimen:N",
            sort=["Negatif", "Positif"],
            axis=alt.Axis(labelAngle=0, title="Kategori Sentimen"),
        ),
        y=alt.Y("Jumlah:Q", title="Jumlah Data"),
        color=alt.Color(
            "Sentimen:N",
            scale=alt.Scale(
                domain=["Negatif", "Positif"], range=["#e74c3c", "#2ecc71"]
            ),
            legend=None,
        ),
        tooltip=["Sentimen", "Jumlah"],
    )
    .properties(height=360)
)
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

# ======================================================
# PILIH SENTIMEN UNTUK TOPIK
# ======================================================
st.markdown("## 🎯 Pilih Sentimen untuk Analisis Topik")
selected_sentiment = st.radio(
    "Analisis topik berdasarkan sentimen:",
    ["Negatif", "Positif"],
    horizontal=True,
)

# ======================================================
# PROSES BERTOPIC LANGSUNG DI STREAMLIT
# ======================================================
st.markdown("## 🧠 Analisis Topik Dominan (BERTopic)")

# Ambil subset data sesuai sentimen
subset = df_result[df_result["sentiment_label"] == selected_sentiment].copy()
subset_texts = subset[st.session_state.text_column].dropna().astype(str).tolist()

if len(subset_texts) < 10:
    st.warning(
        f"Data untuk sentimen **{selected_sentiment}** terlalu sedikit ({len(subset_texts)}). "
        "Minimal disarankan >= 10 teks agar topik lebih stabil."
    )

colA, colB, colC = st.columns([1.2, 1, 1], gap="medium")

with colA:
    st.write("**Ringkasan Data Sentimen Terpilih**")
    st.metric("Jumlah teks", len(subset_texts))

with colB:
    min_topic_size = st.number_input(
        "min_topic_size", min_value=2, max_value=200, value=15, step=1
    )

with colC:
    nr_topics_opt = st.selectbox(
        "nr_topics (opsional)", ["auto", 5, 10, 15, 20, 30], index=0
    )

run_topic = st.button(
    "🧠 Jalankan BERTopic untuk Sentimen Terpilih", use_container_width=True
)


def run_bertopic_for_sentiment(texts, min_topic_size, nr_topics_opt):
    topic_model = load_bertopic()

    # Set parameter runtime (BERTopic object bisa dipakai, tapi param lebih aman dibuat ulang)
    # Agar simple, kita buat model baru dengan embedding_model yang sama.
    embedding_model = topic_model.embedding_model
    nr_topics_val = nr_topics_opt if nr_topics_opt != "auto" else "auto"

    model_runtime = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=int(min_topic_size),
        nr_topics=nr_topics_val,
    )

    topics, _ = model_runtime.fit_transform(texts)
    df_topics = pd.DataFrame({"text": texts, "topic": topics})
    topic_info = model_runtime.get_topic_info()

    # Topik dominan (kecuali -1 jika ada)
    counts = df_topics["topic"].value_counts()
    if -1 in counts.index and len(counts) > 1:
        counts = counts.drop(index=-1)

    top_topic = int(counts.idxmax()) if len(counts) else -1
    top_n = int(counts.loc[top_topic]) if top_topic in counts.index else 0

    return df_topics, topic_info, top_topic, top_n


# Jalankan BERTopic hanya jika diminta atau sudah ada hasil di session_state
if run_topic:
    if len(subset_texts) == 0:
        st.error("Tidak ada teks untuk diproses. Pastikan kolom teks tidak kosong.")
        st.stop()

    with st.spinner(f"🔄 Memproses BERTopic untuk sentimen {selected_sentiment}..."):
        df_topics, topic_info, top_topic, top_n = run_bertopic_for_sentiment(
            subset_texts, min_topic_size, nr_topics_opt
        )

    st.session_state.bertopic_results[selected_sentiment] = {
        "df_topics": df_topics,
        "topic_info": topic_info,
        "top_topic": top_topic,
        "top_n": top_n,
    }

    st.success(f"✅ BERTopic selesai diproses untuk sentimen {selected_sentiment}.")

# Jika belum pernah diproses untuk sentimen ini
if selected_sentiment not in st.session_state.bertopic_results:
    st.info(
        "Klik tombol **Jalankan BERTopic** untuk memproses topik pada sentimen terpilih."
    )
    st.stop()

# Ambil hasil dari session_state
df_topics = st.session_state.bertopic_results[selected_sentiment]["df_topics"]
topic_info = st.session_state.bertopic_results[selected_sentiment]["topic_info"]
topik_dominan = st.session_state.bertopic_results[selected_sentiment]["top_topic"]
jumlah_topik = st.session_state.bertopic_results[selected_sentiment]["top_n"]

badge_color = "#1f3b2c" if selected_sentiment == "Negatif" else "#143a2a"
st.markdown(
    f"""
    <div style="
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: {badge_color};
        border: 1px solid #2a3b2f;
        margin-top: 0.4rem;
    ">
        <span style="font-size: 1rem;">🎯</span>
        <b style="margin-left: 0.4rem;">
            Topik Dominan Sentimen {selected_sentiment}: Topic {topik_dominan}
        </b>
        <span style="color:#b0b3b8; margin-left: 0.6rem;">
            (n={jumlah_topik})
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# KATA KUNCI TOPIK DOMINAN
# ======================================================
st.markdown("### 🔑 Kata Kunci Topik (Dominan)")

# BERTopic topic_info biasanya punya kolom: Topic, Count, Name, Representation
topic_col = (
    "Topic"
    if "Topic" in topic_info.columns
    else ("topic" if "topic" in topic_info.columns else None)
)
repr_col = (
    "Representation"
    if "Representation" in topic_info.columns
    else ("representation" if "representation" in topic_info.columns else None)
)

if topic_col is None or repr_col is None:
    st.warning(
        "Kolom topic_info tidak sesuai standar.\n"
        f"Kolom yang ada: {list(topic_info.columns)}\n"
        "Pastikan topic_info punya kolom `Topic` dan `Representation`."
    )
else:
    info_row = topic_info[topic_info[topic_col] == topik_dominan]
    if info_row.empty:
        st.info("Info kata kunci untuk topik dominan tidak ditemukan.")
    else:
        st.dataframe(
            info_row[[repr_col]].reset_index(drop=True), use_container_width=True
        )

# ======================================================
# CONTOH KOMENTAR TOPIK DOMINAN
# ======================================================
st.markdown("### 💬 Contoh Komentar (Topik Dominan)")

contoh = (
    df_topics[df_topics["topic"] == topik_dominan]["text"]
    .dropna()
    .astype(str)
    .head(5)
    .tolist()
)
if not contoh:
    st.info("Tidak ada contoh komentar untuk topik ini.")
else:
    for i, teks in enumerate(contoh, 1):
        st.write(f"{i}. {teks}")

# ======================================================
# DISTRIBUSI TOPIK (TOP 10)
# ======================================================
st.markdown("### 📊 Distribusi Topik (Top 10)")

topic_counts = df_topics["topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Jumlah"]
topic_counts["Topic"] = topic_counts["Topic"].astype(int)

# opsional: buang outlier -1 agar chart lebih informatif
topic_counts_no_outlier = topic_counts[topic_counts["Topic"] != -1].copy()
if len(topic_counts_no_outlier) > 0:
    plot_df = topic_counts_no_outlier.head(10)
else:
    plot_df = topic_counts.head(10)

topic_chart = (
    alt.Chart(plot_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("Topic:O", sort="-y", title="Topic"),
        y=alt.Y("Jumlah:Q", title="Jumlah Teks"),
        tooltip=["Topic", "Jumlah"],
    )
    .properties(height=320)
)
st.altair_chart(topic_chart, use_container_width=True)

# ======================================================
# DOWNLOAD HASIL BERTOPIC
# ======================================================
st.markdown("### ⬇️ Download Hasil BERTopic")

csv_topics = df_topics.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download df_topics ({selected_sentiment})",
    data=csv_topics,
    file_name=f"hasil_bertopic_{selected_sentiment.lower()}.csv",
    mime="text/csv",
    use_container_width=True,
)

csv_info = topic_info.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download topic_info ({selected_sentiment})",
    data=csv_info,
    file_name=f"topic_info_{selected_sentiment.lower()}.csv",
    mime="text/csv",
    use_container_width=True,
)
