import matplotlib.pyplot as plt
import numpy as np

# Data awal
aspek = [
    "Kecepatan Aplikasi",
    "Tampilan",
    "Transaksi",
    "Login",
    "Kartu Rekening",
    "Keamanan",
    "Customer Service",
    "Notifikasi Informasi",
    "OTP Verifikasi",
    "QRIS",
]

play_store = [25.4, 22.0, 17.3, 12.7, 8.4, 8.2, 5.2, 3.4, 3.0, 1.9]
app_store = [36.0, 48.0, 22.0, 12.0, 14.0, 10.0, 8.0, 4.0, 6.0, 4.0]

# Gabungkan lalu SORT DESCENDING berdasarkan App Store
data = list(zip(aspek, play_store, app_store))
data.sort(key=lambda x: x[2], reverse=True)

aspek_sorted, play_sorted, app_sorted = zip(*data)

y = np.arange(len(aspek_sorted))
height = 0.4

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.barh(y - height / 2, play_sorted, height, label="Play Store")
bars2 = ax.barh(y + height / 2, app_sorted, height, label="App Store")

# Label persentase
for bar in bars1:
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width()}%",
        va="center",
        fontsize=9,
    )

for bar in bars2:
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width()}%",
        va="center",
        fontsize=9,
    )

ax.set_yticks(y)
ax.set_yticklabels(aspek_sorted)
ax.set_xlabel("Persentase (%)")
ax.set_title(
    "Aspek dominan dalam ulasan kritis aplikasi Byond di Play Store dan App Store."
)

# Ruang kanan aman
ax.set_xlim(0, 60)
ax.margins(x=0.02)

ax.legend()
plt.tight_layout()
plt.show()
