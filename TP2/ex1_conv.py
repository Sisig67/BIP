import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.filters as flt
import numpy as np
import scipy


emoji_url = "https://vincmazet.github.io/image-labs/_downloads/7023107ecc4b91d9b662b1961461a31e/smiley.png"
f = io.imread("TP2/smiley.png")
print("got image")
im = plt.imshow(f,"gray")
plt.show()

res1 = flt.gaussian(f, sigma=1, mode='reflect')


un = np.array([1, -1])
kernel = np.zeros_like(f, dtype=float)
center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
kernel[center[0], center[1]-1:center[1]+1] = un
res2 = scipy.ndimage.convolve(f,kernel , mode='reflect')


kernel_thirt = np.zeros_like(f, dtype=float)
center = (kernel_thirt.shape[0] // 2, kernel_thirt.shape[1] // 2)
thirt = 1/30 * np.ones((1, 30))
start = center[1] - 15
end = center[1] + 15
kernel_thirt[center[0], start:end] = thirt[0]
res3 = scipy.ndimage.convolve(f, kernel_thirt, mode='reflect')

print("calc")

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(f, cmap="gray")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Gaussian (sigma=1)")
plt.imshow(res1, cmap="gray")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Convolution [1, -1]")
plt.imshow(res2, cmap="gray")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Convolution 1x30")
plt.imshow(res3, cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.show()