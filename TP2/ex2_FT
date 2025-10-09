import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.filters as flt
import numpy as np
import scipy

# List of image filenames
image_files = [f"TP2/s{i}.png" for i in range(1, 6)]

fig, axes = plt.subplots(5, 3, figsize=(12, 18))
for idx, fname in enumerate(image_files):
    # Load image
    print(fname)
    img = io.imread(fname)
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img

    # Compute FFT and shift
    fft_img = np.fft.fft2(img_gray)
    fft_shifted = np.fft.fftshift(fft_img)

    # Display original
    axes[idx, 0].imshow(img_gray, cmap='gray')
    axes[idx, 0].set_title(f"Original s{idx+1}")
    axes[idx, 0].axis('off')

    # Display absolute of FFT
    axes[idx, 1].imshow(np.abs(fft_shifted), cmap='gray')
    axes[idx, 1].set_title("FFT Absolute")
    axes[idx, 1].axis('off')

    # Display magnitude (same as absolute for real images)
    axes[idx, 2].imshow(np.angle((fft_shifted)), cmap='gray')
    axes[idx, 2].set_title("FFT Angle ")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.show()