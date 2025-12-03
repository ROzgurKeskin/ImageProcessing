import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from library.data_utils import get_csv_path, load_dataframe, sample_images

def contrast_stretch_channel(channel):
    import numpy as np
    min_val = np.min(channel)
    max_val = np.max(channel)
    if max_val > min_val:
        stretched = ((channel - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    else:
        stretched = channel.copy()
    return stretched

def contrast_stretch_rgb(img):
    import cv2
    channels = cv2.split(img)
    stretched_channels = [contrast_stretch_channel(c) for c in channels]
    return cv2.merge(stretched_channels)

def contrast_stretch_gray(img):
    return contrast_stretch_channel(img)

def equalize_histogram_rgb(img):
    import cv2
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    rgb_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    return rgb_eq

def equalize_histogram_gray(img):
    import cv2
    return cv2.equalizeHist(img)

def gamma_correction(img, gamma):
    import numpy as np
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, table)

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 5 görüntü seç (görselleştirme için daha kompakt)
sample_df = sample_images(train_df, n=5, random_state=42)

fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(42, 20))
fig.suptitle('Histogram Eşitleme ve Gamma Düzeltme Sonuçları', fontsize=18, fontweight='bold')

gamma_values = [0.5, 1.0, 2.0]

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    file_size = Path(img_path).stat().st_size if Path(img_path).exists() else 0
    if img is None:
        rgb_img = None
        gray_img = None
        rgb_eq = None
        gray_eq = None
        rgb_info = f"Yüklenemedi | {row.filename} | Dosya boyutu: {file_size} bytes"
        gray_info = f"Yüklenemedi | {row.filename} | Dosya boyutu: {file_size} bytes"
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_eq = equalize_histogram_rgb(rgb_img)
        gray_eq = equalize_histogram_gray(gray_img)
        rgb_info = f"{row.filename} | RGB | {row.width}x{row.height} | {file_size} bytes"
        gray_info = f"{row.filename} | Grayscale | {row.width}x{row.height} | {file_size} bytes"
    # Orijinal RGB
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('Orijinal RGB', fontsize=10)
    # Histogram Eşitlemeli RGB
    axes[idx, 1].imshow(rgb_eq)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Histogram Eşitlemeli RGB', fontsize=10)
    # Orijinal Grayscale
    axes[idx, 2].imshow(gray_img, cmap='gray')
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('Orijinal Grayscale', fontsize=10)
    # Histogram Eşitlemeli Grayscale
    axes[idx, 3].imshow(gray_eq, cmap='gray')
    axes[idx, 3].axis('off')
    axes[idx, 3].set_title('Histogram Eşitlemeli Grayscale', fontsize=10)
    # Gamma düzeltmeli RGB (gamma=0.5, 1.0, 2.0)
    for gidx, gamma in enumerate(gamma_values):
        if rgb_img is not None:
            rgb_gamma = gamma_correction(rgb_img, gamma)
            axes[idx, 4+gidx].imshow(rgb_gamma)
            axes[idx, 4+gidx].axis('off')
            axes[idx, 4+gidx].set_title(f'RGB Gamma={gamma}', fontsize=10)
        else:
            axes[idx, 4+gidx].axis('off')
            axes[idx, 4+gidx].set_title(f'RGB Gamma={gamma}', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.6)
plt.show()
