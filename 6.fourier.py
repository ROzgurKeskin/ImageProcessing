import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from library.data_utils import load_dataframe, sample_images

# DataFrame'i yükle
train_df = load_dataframe()

# Rastgele 10 görüntü seç
sample_df = sample_images(train_df, n=10, random_state=42)

fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20, 18))
fig.suptitle('Fourier Dönüşümü ve Alçak Geçiren Filtre ile Ters FFT', fontsize=16, fontweight='bold')

for idx, row in enumerate(sample_df.itertuples()):
    img_path = row.filepath
    img = cv2.imread(img_path)
    if img is None:
        rgb_img = None
        gray_img = None
        spectrum = None
        mask_img = None
        recon = None
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # FFT
        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        spectrum = magnitude_spectrum
        # Alçak geçiren filtre maskesi
        rows, cols = gray_img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        r = min(rows, cols) // 8  # Maske yarıçapı
        mask[crow - r:crow + r, ccol - r:ccol + r] = 1
        # Maskeyi uygulama
        fshift_masked = fshift * mask
        magnitude_masked = 20 * np.log(np.abs(fshift_masked) + 1)
        mask_img = magnitude_masked
        # Ters FFT ile rekonstrüksiyon (uzamsal alana dönüş)
        f_ishift = np.fft.ifftshift(fshift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        recon = img_back
    # Orijinal RGB
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('RGB', fontsize=9)
    # Grayscale
    axes[idx, 1].imshow(gray_img, cmap='gray')
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Grayscale', fontsize=9)
    # FFT Spektrum
    axes[idx, 2].imshow(spectrum, cmap='gray')
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('FFT Spektrum', fontsize=9)
    # Maskeli Spektrum
    axes[idx, 3].imshow(mask_img, cmap='gray')
    axes[idx, 3].axis('off')
    axes[idx, 3].set_title('Maskeli Spektrum', fontsize=9)
    # Ters FFT ile rekonstrüksiyon
    axes[idx, 4].imshow(recon, cmap='gray')
    axes[idx, 4].axis('off')
    axes[idx, 4].set_title('Filtreli Rekonstrüksiyon', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.3)
plt.show()
