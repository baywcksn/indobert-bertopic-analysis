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
    page_title="IndoBERT & BERTopic Analysis",
    page_icon="📊",
    layout="wide",
)

# ======================================================
# SESSION STATE (BIAR TIDAK RESET KETIKA GANTI RADIO)
# ======================================================
if "df_labeled" not in st.session_state:
    st.session_state.df_labeled = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "text_column" not in st.session_state:
    st.session_state.text_column = None

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
# LOAD MODEL INDOBERT (CACHE BIAR CEPAT)
# ======================================================
MODEL_PATH = "./indobert-sentiment-tiktok"
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
# LOAD HASIL BERTOPIC (NEGATIF & POSITIF)
# ======================================================
@st.cache_data
def load_bertopic_outputs():
    # Pastikan file-file ini ada di folder output/
    df_neg = pd.read_csv("output/hasil_bertopic_negatif_tiktok.csv")
    info_neg = pd.read_csv("output/topic_info_negatif_tiktok.csv")

    df_pos = pd.read_csv("output/hasil_bertopic_positif_tiktok.csv")
    info_pos = pd.read_csv("output/topic_info_positif_tiktok.csv")

    return (df_neg, info_neg), (df_pos, info_pos)


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
# TOMBOL ANALISIS (HANYA SEKALI, HASIL DISIMPAN DI SESSION)
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

    # ✅ simpan supaya tidak hilang saat UI berubah
    st.session_state.df_labeled = df_result
    st.session_state.analysis_done = True

    st.success("🎉 Analisis sentimen selesai dan disimpan!")

# ======================================================
# JIKA BELUM ANALISIS, STOP DI SINI
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
# DISTRIBUSI SENTIMEN (ALTAIR: NEGATIF MERAH, POSITIF HIJAU)
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
                domain=["Negatif", "Positif"],
                range=["#e74c3c", "#2ecc71"],
            ),
            legend=None,
        ),
        tooltip=["Sentimen", "Jumlah"],
    )
    .properties(height=360)
)

st.altair_chart(chart, use_container_width=True)

# ======================================================
# DOWNLOAD HASIL INDO BERT (DI BAGIAN SENTIMEN, BUKAN BERTOPIC)
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
# PILIH SENTIMEN UNTUK TOPIK (TIDAK RESET!)
# ======================================================
st.markdown("## 🎯 Pilih Sentimen untuk Analisis Topik")

selected_sentiment = st.radio(
    "Analisis topik berdasarkan sentimen:",
    ["Negatif", "Positif"],
    horizontal=True,
)

# ======================================================
# ANALISIS TOPIK (BERTOPIC) - LOAD CSV HASIL NOTEBOOK
# ======================================================
st.markdown("## 🧠 Analisis Topik Dominan (BERTopic)")

try:
    (df_neg, info_neg), (df_pos, info_pos) = load_bertopic_outputs()
except Exception as e:
    st.error(
        "File output BERTopic belum ditemukan / salah path.\n\n"
        "Pastikan ada file berikut di folder `output/`:\n"
        "- hasil_bertopic_negatif_tiktok.csv\n"
        "- topic_info_negatif_tiktok.csv\n"
        "- hasil_bertopic_positif_tiktok.csv\n"
        "- topic_info_positif_tiktok.csv\n\n"
        f"Detail error: {e}"
    )
    st.stop()

if selected_sentiment == "Negatif":
    df_topics = df_neg
    topic_info = info_neg
else:
    df_topics = df_pos
    topic_info = info_pos

# Pastikan kolom penting ada
required_cols = {"topic"}
if not required_cols.issubset(set(df_topics.columns)):
    st.error(
        f"CSV BERTopic harus memiliki kolom: {required_cols}. Kolom saat ini: {list(df_topics.columns)}"
    )
    st.stop()

# Ambil topik dominan (yang count paling besar, selain -1 jika ada)
topic_counts = df_topics["topic"].value_counts()
if -1 in topic_counts.index and len(topic_counts) > 1:
    topic_counts = topic_counts.drop(index=-1)

topik_dominan = int(topic_counts.idxmax())
jumlah_topik = int(topic_counts.loc[topik_dominan])

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
# KATA KUNCI TOPIK
# ======================================================
st.markdown("### 🔑 Kata Kunci Topik")

# Beberapa versi topic_info pakai kolom "Topic" atau "topic"
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
# CONTOH KOMENTAR (ambil dari kolom teks yang tersedia di df_topics)
# ======================================================
st.markdown("### 💬 Contoh Komentar")

# pilih kolom teks terbaik yang tersedia
candidate_text_cols = ["cleaning", "normalisasi", "comment", "text", "review"]
text_col_in_topics = next(
    (c for c in candidate_text_cols if c in df_topics.columns), None
)

if text_col_in_topics is None:
    st.info(
        "Tidak menemukan kolom teks pada CSV hasil BERTopic.\n"
        f"Kolom tersedia: {list(df_topics.columns)}\n"
        "Minimal salah satu dari: cleaning / normalisasi / comment / text / review"
    )
else:
    contoh = (
        df_topics[df_topics["topic"] == topik_dominan][text_col_in_topics]
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
