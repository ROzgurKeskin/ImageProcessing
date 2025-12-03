import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from library.data_utils import load_dataframe, sample_images

# Unsharp Masking fonksiyonu
def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened

# Bicubic enterpolasyon ile 2 kat büyütme
def bicubic_resize(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 10 görüntü seç
sample_df = sample_images(train_df, n=10, random_state=42)

# 1. Görsel: RGB ve Grayscale için keskinleştirme karşılaştırması
fig1, axes1 = plt.subplots(nrows=10, ncols=4, figsize=(14, 18))
fig1.suptitle('Unsharp Masking ile Keskinleştirme (RGB ve Grayscale)', fontsize=16, fontweight='bold')

# 2. Görsel: Bicubic büyütme ve boyut farkı
fig2, axes2 = plt.subplots(nrows=10, ncols=4, figsize=(16, 18))
fig2.suptitle('Bicubic Büyütme: Orijinal ve Keskinleştirilmiş Görsellerin Boyut Farkı', fontsize=16, fontweight='bold')

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    if img is None:
        rgb_img = None
        gray_img = None
        rgb_sharp = None
        gray_sharp = None
        rgb_bicubic = None
        gray_bicubic = None
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_sharp = unsharp_mask(rgb_img, kernel_size=(5,5), sigma=1.0, amount=1.0)
        gray_sharp = unsharp_mask(gray_img, kernel_size=(5,5), sigma=1.0, amount=1.0)
        rgb_bicubic = bicubic_resize(rgb_sharp)
        gray_bicubic = bicubic_resize(gray_sharp)
    # --- 1. Görsel ---
    # Orijinal RGB
    axes1[idx, 0].imshow(rgb_img)
    axes1[idx, 0].axis('off')
    axes1[idx, 0].set_title('RGB', fontsize=9)
    # Keskinleştirilmiş RGB
    axes1[idx, 1].imshow(rgb_sharp)
    axes1[idx, 1].axis('off')
    axes1[idx, 1].set_title('RGB Unsharp', fontsize=9)
    # Orijinal Grayscale
    axes1[idx, 2].imshow(gray_img, cmap='gray')
    axes1[idx, 2].axis('off')
    axes1[idx, 2].set_title('Grayscale', fontsize=9)
    # Keskinleştirilmiş Grayscale
    axes1[idx, 3].imshow(gray_sharp, cmap='gray')
    axes1[idx, 3].axis('off')
    axes1[idx, 3].set_title('Grayscale Unsharp', fontsize=9)
    # --- 2. Görsel ---
    # Orijinal RGB
    axes2[idx, 0].imshow(rgb_img)
    axes2[idx, 0].axis('off')
    axes2[idx, 0].set_title(f'RGB\n{rgb_img.shape}', fontsize=9)
    # Keskinleştirilmiş RGB (orijinal boyut)
    axes2[idx, 1].imshow(rgb_sharp)
    axes2[idx, 1].axis('off')
    axes2[idx, 1].set_title(f'RGB Unsharp\n{rgb_sharp.shape}', fontsize=9)
    # Bicubic büyütülmüş RGB
    axes2[idx, 2].imshow(rgb_bicubic)
    axes2[idx, 2].axis('off')
    axes2[idx, 2].set_title(f'RGB Bicubic x2\n{rgb_bicubic.shape}', fontsize=9)
    # Bicubic büyütülmüş Grayscale
    axes2[idx, 3].imshow(gray_bicubic, cmap='gray')
    axes2[idx, 3].axis('off')
    axes2[idx, 3].set_title(f'Grayscale Bicubic x2\n{gray_bicubic.shape}', fontsize=9)

fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.subplots_adjust(hspace=0.3)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.subplots_adjust(hspace=0.3)
plt.show()
