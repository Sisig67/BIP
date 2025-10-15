import skimage.io as io
import scipy.fftpack as fft
import numpy as np

img = io.imread('TP3/squirrel.png', as_gray=True)
dct = fft.dctn(img, norm='ortho')

import matplotlib.pyplot as plt


print(dct.shape)
mask = np.zeros_like(dct)
flt = 127
mask[:127, :127] = 1

dct2 = np.multiply(dct, mask)

img2 = fft.idctn(dct2, norm='ortho')

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(np.log(np.abs(dct) + 1), cmap='gray')
axs[1].set_title('DCT Coefficients')
axs[1].axis('off')

axs[2].imshow(np.log(np.abs(dct2) + 1), cmap='gray')
axs[2].set_title('Masked DCT Coefficients')
axs[2].axis('off')

axs[3].imshow(img2, cmap='gray')
axs[3].set_title('Reconstructed Image')
axs[3].axis('off')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 7, figsize=(16, 4))
j = 1
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')
axs[1, 0].imshow(np.zeros_like(img), cmap='gray')
axs[1, 0].axis('off')
errs = []
flts = []
for i in range(6):
    flt = int(2**(6+i/2))
    mask = np.zeros_like(dct)
    mask[:flt, :flt] = 1
    dct2 = np.multiply(dct, mask)
    img2 = fft.idctn(dct2, norm='ortho')
    axs[0, j].imshow(img2, cmap='gray')
    axs[0, j].set_title(f'flt={flt}')
    axs[0, j].axis('off')
    error = img-img2
    SE = np.sum(np.multiply(error, error))
    SE_formatted = f"{SE:.2e}"
    flts.append(flt)
    errs.append(SE)
    axs[1, j].imshow(error, cmap ='gray')
    axs[1, j].set_title(f'SE={SE_formatted}')
    axs[1, j].axis('off')
    j += 1

plt.tight_layout()
plt.show()
Is = np.logspace(5, 8.9, num=10000, base=2, dtype=int)
# You can use CuPy for GPU acceleration if you have a compatible GPU and CuPy installed.
# Replace numpy/scipy operations with cupy equivalents.

import cupy as cp
dct_gpu = cp.asarray(dct)
img_gpu = cp.asarray(img)
flts_gpu = []
errs_gpu = []
for flt in Is:
    mask_gpu = cp.zeros_like(dct_gpu)
    mask_gpu[:flt, :flt] = 1
    dct2_gpu = dct_gpu * mask_gpu
    img2_gpu = cp.asnumpy(fft.idctn(cp.asnumpy(dct2_gpu), norm='ortho'))  # idctn not available in cupy
    error_gpu = img - img2_gpu
    SE_gpu = cp.sum(error_gpu * error_gpu)
    flts_gpu.append(flt)
    errs_gpu.append(SE_gpu)

# Note: scipy.fftpack's dctn/idctn are not GPU-accelerated. You can accelerate masking and array ops,
# but the DCT/iDCT steps will still run on CPU unless you use a GPU-accelerated library for DCT.
# For full GPU acceleration, consider using OpenCV's cv2.dct/cv2.idct with CUDA, or custom CUDA kernels.
for flt in Is:
    mask = np.zeros_like(dct)
    mask[:flt, :flt] = 1
    dct2 = np.multiply(dct, mask)
    img2 = fft.idctn(dct2, norm='ortho')
    error = img-img2
    SE = np.sum(np.multiply(error, error))
    SE_formatted = f"{SE:.2e}"
    flts.append(flt)
    errs.append(SE)



plt.figure(figsize=(8, 5))
plt.plot(np.array(flts), errs, marker='o', linestyle='None')
plt.xlabel('flt^2')
plt.ylabel('Squared Error (SE)')
plt.yscale('log')
plt.title('Squared Error vs. flt^2')
plt.grid(True)
plt.show()