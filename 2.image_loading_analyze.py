import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from library.data_utils import get_csv_path, load_dataframe, sample_images

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 10 görüntü seç
sample_df = sample_images(train_df, n=10, random_state=42)

fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(24, 32))
fig.suptitle('RGB ve Grayscale Görüntüler ve Histogramları', fontsize=18, fontweight='bold')

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    file_size = Path(img_path).stat().st_size if Path(img_path).exists() else 0
    if img is None:
        rgb_img = None
        gray_img = None
        rgb_info = f"Yüklenemedi | {row.filename} | Dosya boyutu: {file_size} bytes"
        gray_info = f"Yüklenemedi | {row.filename} | Dosya boyutu: {file_size} bytes"
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_info = f"{row.filename} | RGB | {row.width}x{row.height} | {file_size} bytes"
        gray_info = f"{row.filename} | Grayscale | {row.width}x{row.height} | {file_size} bytes"
    # RGB
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].text(0.5, -0.10, rgb_info, fontsize=8, ha='center', va='top', transform=axes[idx, 0].transAxes, wrap=False)
    # Grayscale
    axes[idx, 1].imshow(gray_img, cmap='gray')
    axes[idx, 1].axis('off')
    axes[idx, 1].text(0.5, -0.10, gray_info, fontsize=8, ha='center', va='top', transform=axes[idx, 1].transAxes, wrap=False)
    # RGB Histogram
    axes[idx, 2].clear()
    if img is not None:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([rgb_img], [i], None, [256], [0,256])
            axes[idx, 2].plot(hist, color=color, label=f'{color.upper()}')
        axes[idx, 2].set_xlim([0,256])
        axes[idx, 2].set_title('RGB Histogram', fontsize=10)
        axes[idx, 2].legend()
    else:
        axes[idx, 2].set_title('RGB Histogram (Yok)', fontsize=10)
    # Grayscale Histogram
    axes[idx, 3].clear()
    if img is not None:
        hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0,256])
        axes[idx, 3].plot(hist_gray, color='black')
        axes[idx, 3].set_xlim([0,256])
        axes[idx, 3].set_title('Grayscale Histogram', fontsize=10)
    else:
        axes[idx, 3].set_title('Grayscale Histogram (Yok)', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.6)
plt.show()
