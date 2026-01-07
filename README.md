# Analisis Sentimen Menggunakan IndoBERT dan BERTopic  
### Studi Kasus Data Media Sosial X dan TikTok (±20.000 Data)

Proyek ini merupakan implementasi analisis sentimen modern dengan memanfaatkan model *IndoBERT-base-p2* dan pendekatan topik dinamis menggunakan BERTopic. Seluruh proses eksplorasi, preprocessing, modeling, hingga evaluasi dilakukan menggunakan **Jupyter Notebook yang dijalankan di Visual Studio Code (VS Code)**.

---

## 📁 Struktur Proyek

<p align="center">
  <img width="500" height="700" alt="struktur-proyek" src="https://github.com/user-attachments/assets/976c161b-db1c-458b-9ce9-d3ad02e6d277" />
</p>



---

## 🧰 Tools & Teknologi

Proyek ini dikembangkan menggunakan:

- **Visual Studio Code** (VS Code)
- **Jupyter Notebook** kernel Conda environment
- **Python 3.10+**
- **IndoBERT-base-p2** (sentiment classification)
- **BERTopic** (topic modeling)
- **HuggingFace Transformers**
- **PyTorch with CUDA support**
- **UMAP, HDBSCAN, Sastrawi**
- **Pandas, NumPy, Matplotlib, Plotly**

---

## 📊 Deskripsi Project

Proyek ini bertujuan untuk:

1. **Menganalisis sentimen** dari postingan pengguna media sosial (X dan TikTok) dengan total data sekitar **20.000 entri**.
2. Melakukan **topic modeling** untuk menemukan pola pembahasan dominan pada kedua platform.
3. Membandingkan pola sentimen dan topik antara platform untuk menemukan insight yang relevan.

Fokus analisis berada pada:

- Sentimen (*positive, neutral, negative*)
- Persebaran topik dan cluster pembahasan
- Kategori opini pengguna berdasarkan model IndoBERT
- Korelasi antar-topik, trend, serta frekuensi kemunculan

---

## 🧪 Alur Kerja Proyek

### 1️⃣ *Scraping Data (X & TikTok)*  
Data dikumpulkan melalui proses scraping kemudian disimpan dalam format `.csv`.

### 2️⃣ *Preprocessing*  
- Lowercasing  
- Cleansing teks  
- Stopword removal  
- Tokenization  
- Normalisasi kata tidak baku  
- Deteksi bahasa  

### 3️⃣ *Sentiment Classification – IndoBERT*  
Model **IndoBERT-base-p2** digunakan untuk klasifikasi sentimen dengan fine-tuning tambahan pada dataset bahasa Indonesia.

### 4️⃣ *Topic Modeling – BERTopic*  
- Embedding menggunakan **SentenceTransformer IndoBERT**  
- Reduksi dimensi menggunakan **UMAP**
- Clustering menggunakan **HDBSCAN**
- Representasi topik menggunakan **class-based TF-IDF**

### 5️⃣ *Visualization & Insight*  
Visualisasi dilakukan menggunakan:

- Plotly
- BERTopic visualizer
- WordCloud
- Bar chart & trendline

---

## 🚀 Instalasi & Menjalankan Proyek

### 💻 Clone Repository
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
