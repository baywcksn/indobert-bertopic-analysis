import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import json
import os

# ======================================================
# KONFIGURASI PATH (ANTI HF ERROR)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "indobert-sentiment-tiktok"
TEST_SET_PATH = (
    BASE_DIR / "evaluation" / "Tiktok\dataset-tiktok\brimo_appstore_reviews.csv"
)
OUTPUT_DIR = BASE_DIR / "evaluation"

DEVICE = torch.device("cpu")
LABEL_MAP = {0: "Negatif", 1: "Positif"}

# ======================================================
# VALIDASI PATH
# ======================================================
print("BASE DIR :", BASE_DIR)
print("MODEL DIR:", MODEL_DIR)

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Folder model TIDAK ditemukan: {MODEL_DIR}")

# 👉 PAKSA JADI STRING ABSOLUTE (INI KUNCI UTAMA)
MODEL_PATH = os.path.abspath(MODEL_DIR)

print("MODEL PATH (ABS):", MODEL_PATH)

# ======================================================
# LOAD MODEL (ANTI REPO VALIDATION)
# ======================================================
print("Loading IndoBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, local_files_only=True
)

model.to(DEVICE)
model.eval()

print("Model IndoBERT berhasil dimuat")

# ======================================================
# LOAD TEST SET
# ======================================================
df_test = pd.read_csv(TEST_SET_PATH)


# ======================================================
# PREDIKSI
# ======================================================
def predict(text: str) -> str:
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
        _, pred = torch.max(probs, dim=1)
    return LABEL_MAP[pred.item()]


y_true, y_pred = [], []

print("🔎 Evaluating model...")
for _, row in df_test.iterrows():
    text = str(row["text"])
    label = str(row["label"]).strip()
    if label not in ["Negatif", "Positif"]:
        continue
    y_true.append(label)
    y_pred.append(predict(text))

# ======================================================
# METRIK
# ======================================================
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=["Negatif", "Positif"])
report = classification_report(
    y_true, y_pred, labels=["Negatif", "Positif"], output_dict=True, zero_division=0
)

# ======================================================
# SIMPAN HASIL
# ======================================================
OUTPUT_DIR.mkdir(exist_ok=True)

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "total_test_samples": len(y_true)}, f, indent=4)

pd.DataFrame(
    cm,
    index=["Aktual_Negatif", "Aktual_Positif"],
    columns=["Prediksi_Negatif", "Prediksi_Positif"],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")

print("\n================ EVALUATION RESULT ================")
print(f"Akurasi          : {accuracy:.2%}")
print(f"Jumlah data uji  : {len(y_true)}")
print("✅ Evaluasi selesai. File tersimpan di /evaluation/")
