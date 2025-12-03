import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.filters as flt
import skimage.util as u
import skimage.metrics as m
import skimage.restoration as r
import numpy as np
import scipy
import time

# Try to use CuPy (CUDA) when available for faster convolutions
use_gpu = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    use_gpu = False
    print('CuPy imported successfully, GPU acceleration enabled.')
except Exception:
    cp = None
    cnd = None

img = io.imread('TP6/squirrel.png', as_gray=True).astype(np.float32)
img_noisy = u.random_noise(img, mode='gaussian', clip=False, var=100)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy')
plt.axis('off')

plt.tight_layout()
plt.show()



# fig, axs = plt.subplots(4, 7, figsize=(15, 6))
# N = []
# dN = []
# e = []
# i=0
#
# for var in range(1, 5):
#     img_noisy = u.random_noise(img, mode='gaussian', clip=False, var=10**(var))
#     N.append(img_noisy)
#     noise = img_noisy - img
#     SNR = 10*np.log10(np.mean(np.multiply(img, img)) / np.mean(np.multiply(noise, noise)))
#     mse = m.mean_squared_error(img, img_noisy)
#     axs[i, 0].imshow(img_noisy, cmap='gray')
#     axs[i, 0].set_title(f'Noisy (var={10**(var)}), (SNR={SNR:.2f} dB),\n mse = {mse:.4f}')
#     axs[i, 0].axis('off')
#     print(var)
#     j=1
#     for w in range(0, 6):
#         w_ = w+var
#         h = np.ones((w_,w_)) / (w_*w_)
#         denoised = scipy.ndimage.convolve(img_noisy, h)
#         mse = m.mean_squared_error(img, denoised)
#         axs[i, j].imshow(denoised, cmap='gray')
#         axs[i, j].set_title(f'Filter size={w_},\n mse = {mse:.4f}')
#         axs[i, j].axis('off')
#         print(f'  Filter size={w}')
#         j+=1
#     i+=1
#
# plt.tight_layout()
# plt.show()
print(len(img))

V = np.logspace(1, 5, 30)
S = []
W = []
MSE = []

print(f"GPU acceleration available: {use_gpu}")

for var in V:
    print(f"noise var={var}")
    img_noisy = u.random_noise(img, mode='gaussian', clip=False, var=var).astype(np.float32)

    # compute SNR on CPU (cheap)
    noise = img_noisy - img
    SNR = 10*np.log10(np.mean(np.multiply(img, img)) / np.mean(np.multiply(noise, noise)))
    S.append(SNR)

    min_mse = None
    min_w = None

    if use_gpu:
        # move arrays to GPU once
        img_gpu = cp.array(img)
        img_noisy_gpu = cp.array(img_noisy)
        # use cupyx.scipy.ndimage.uniform_filter which implements mean filter efficiently
        for w in range(1, 512, 4):
            # uniform_filter expects an integer size; skip even/zero sizes less than 1
            size = int(w)
            if size < 1:
                continue
            print(f"  filter size={size} (GPU)")
            denoised_gpu = cnd.uniform_filter(img_noisy_gpu, size=size, mode='reflect')
            mse_gpu = cp.mean((img_gpu - denoised_gpu) ** 2)
            mse = float(mse_gpu.get())
            if min_mse is None or mse < min_mse:
                min_mse = mse
                min_w = size
    else:
        # CPU fallback using scipy's uniform_filter for mean filtering
        for w in range(1, 512, 1):
            size = int(w)
            if size < 1:
                continue
            denoised = scipy.ndimage.uniform_filter(img_noisy, size=size, mode='reflect')
            mse = m.mean_squared_error(img, denoised)
            if min_mse is None or mse < min_mse:
                min_mse = mse
                min_w = size

    W.append(min_w)
    MSE.append(min_mse)

# convert to arrays for plotting
V = np.array(V)
S = np.array(S)
MSE = np.array(MSE)
W = np.array(W)

fig, (ax, ax_w) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]})

# Top: MSE and SNR vs V
ax.set_xscale('log')
ln1, = ax.plot(V, MSE, 'o-', label='MSE')
ax.set_xlabel('V (noise variance)')
ax.set_ylabel('MSE')

ax2 = ax.twinx()
ln2, = ax2.plot(V, S, 's--', color='C1', label='SNR (dB)')
ax2.set_ylabel('SNR (dB)')

ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], loc='best')
ax.set_title('MSE and SNR vs Noise Variance')

# Bottom: minimum filter size (w) vs V
ax_w.set_xscale('log')
ax_w.plot(V, W, 'd-', color='C2')
ax_w.set_xlabel('V (noise variance)')
ax_w.set_ylabel('Min filter size (w)')
ax_w.set_title('Optimal (min MSE) filter size vs Noise Variance')

plt.tight_layout()
plt.show()
