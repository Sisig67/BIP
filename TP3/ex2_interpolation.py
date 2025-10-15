import skimage.io as io
import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as tr

img = io.imread('TP3/squirrel.png', as_gray=True)
print(img.shape)
fig, axs = plt.subplots(1, 5, figsize=(16, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
img2 = img
for i in range(4):
    img2 = tr.rotate(img2, angle=90)
    axs[i+1].imshow(img2, cmap='gray')
    
    axs[i+1].axis('off')
    error = img - (img2 * 255).astype(np.int32)
    SE = np.sum(np.multiply(error, error))/400
    axs[i+1].set_title(f'Rotated {90*(i+1)}°, SE={SE:.2e}')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
img2 = img
for i in range(9):
    img2 = tr.rotate(img2, angle=40)
    axs[i+1].imshow(img2, cmap='gray')
    error = img-(img2 * 255).astype(np.int32)
    SE = np.sum(np.multiply(error, error))/400
    axs[i+1].set_title(f'Rotated {40*(i+1)}°, SE={SE:.2e}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
img2 = img
for i in range(9):
    img2 = tr.rotate(img2, angle=40, order = 5)
    axs[i+1].imshow(img2, cmap='gray')
    error = img-(img2 * 255).astype(np.int32)
    SE = np.sum(np.multiply(error, error))/400
    axs[i+1].set_title(f'Rotated {40*(i+1)}°, SE={SE:.2e}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()