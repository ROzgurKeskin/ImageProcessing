import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from library.data_utils import load_dataframe, sample_images

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 10 görüntü seç
sample_df = sample_images(train_df, n=10, random_state=42)

fig, axes = plt.subplots(nrows=10, ncols=6, figsize=(18, 16))
fig.suptitle('RGB ve Grayscale: Orijinal, Rastgele Döndürme, Yatay Flip', fontsize=16, fontweight='bold')

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    if img is None:
        rgb_img = None
        gray_img = None
        rgb_rot = None
        gray_rot = None
        rgb_flip = None
        gray_flip = None
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Rastgele döndürme açısı
        angle = random.uniform(0, 10)
        # RGB döndürme
        h, w = rgb_img.shape[:2]
        M_rgb = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rgb_rot = cv2.warpAffine(rgb_img, M_rgb, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Grayscale döndürme
        h_g, w_g = gray_img.shape[:2]
        M_gray = cv2.getRotationMatrix2D((w_g/2, h_g/2), angle, 1)
        gray_rot = cv2.warpAffine(gray_img, M_gray, (w_g, h_g), borderMode=cv2.BORDER_REFLECT)
        # Yatay flip
        rgb_flip = cv2.flip(rgb_img, 1)
        gray_flip = cv2.flip(gray_img, 1)
    # Orijinal RGB
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('RGB', fontsize=9)
    # Orijinal Grayscale
    axes[idx, 1].imshow(gray_img, cmap='gray')
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Grayscale', fontsize=9)
    # Rastgele döndürülmüş RGB
    axes[idx, 2].imshow(rgb_rot)
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('RGB Rotated', fontsize=9)
    # Rastgele döndürülmüş Grayscale
    axes[idx, 3].imshow(gray_rot, cmap='gray')
    axes[idx, 3].axis('off')
    axes[idx, 3].set_title('Grayscale Rotated', fontsize=9)
    # Yatay flip RGB
    axes[idx, 4].imshow(rgb_flip)
    axes[idx, 4].axis('off')
    axes[idx, 4].set_title('RGB Flipped', fontsize=9)
    # Yatay flip Grayscale
    axes[idx, 5].imshow(gray_flip, cmap='gray')
    axes[idx, 5].axis('off')
    axes[idx, 5].set_title('Grayscale Flipped', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.3)
plt.show()
