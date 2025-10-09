import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.filters as flt
import numpy as np
import scipy

img = io.imread("TP2/roof.jpg")
img_dg = img[::3,::3]
io.imsave("TP2/roof_dg.png",img_dg)

img_gss = flt.gaussian(img, sigma=2, mode='reflect')
img_gss = (img_gss * 255).astype(np.uint8)
img_gss_dg = img_gss[::3,::3]
io.imsave("TP2/roof_gss_dg.png",img_gss_dg)

flt_size = int(np.size(img, axis=1) * 4 / 4)
print(flt_size)
mask = np.zeros_like(img, dtype=np.uint8)
center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
half_size = flt_size // 2
mask[center_x - half_size:center_x + half_size, center_y - half_size:center_y + half_size] = 1

fft = (np.fft.fft2(img))
img_filt = np.fft.ifft2(np.multiply(fft, mask))
img_filt = (img_filt * 255).astype(np.uint8)
img_filt = img_filt[::3,::3]
io.imsave("TP2/roof_filt_dg.png", img_filt)