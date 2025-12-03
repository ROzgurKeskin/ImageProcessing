import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from library.data_utils import load_dataframe, sample_images

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 10 görüntü seç
sample_df = sample_images(train_df, n=10, random_state=42)

fig, axes = plt.subplots(nrows=10, ncols=6, figsize=(18, 16))
fig.suptitle('RGB ve Grayscale: Orijinal, Median Blur, Gaussian Blur', fontsize=16, fontweight='bold')

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    if img is None:
        rgb_img = None
        gray_img = None
        rgb_median = None
        gray_median = None
        rgb_gaussian = None
        gray_gaussian = None
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_median = cv2.medianBlur(rgb_img, 5)
        gray_median = cv2.medianBlur(gray_img, 5)
        rgb_gaussian = cv2.GaussianBlur(rgb_img, (5, 5), 0)
        gray_gaussian = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Orijinal RGB
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('RGB', fontsize=9)
    # Orijinal Grayscale
    axes[idx, 1].imshow(gray_img, cmap='gray')
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Grayscale', fontsize=9)
    # RGB Median
    axes[idx, 2].imshow(rgb_median)
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('RGB Median', fontsize=9)
    # Grayscale Median
    axes[idx, 3].imshow(gray_median, cmap='gray')
    axes[idx, 3].axis('off')
    axes[idx, 3].set_title('Grayscale Median', fontsize=9)
    # RGB Gaussian
    axes[idx, 4].imshow(rgb_gaussian)
    axes[idx, 4].axis('off')
    axes[idx, 4].set_title('RGB Gaussian', fontsize=9)
    # Grayscale Gaussian
    axes[idx, 5].imshow(gray_gaussian, cmap='gray')
    axes[idx, 5].axis('off')
    axes[idx, 5].set_title('Grayscale Gaussian', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.3)
plt.show()
