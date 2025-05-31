import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image = cv2.imread('blackbox.png', cv2.IMREAD_GRAYSCALE)

# Compute Fourier transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Ideal high-pass filter
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
radius = 30
mask_high = np.ones((rows, cols), np.uint8)
mask_high[center_row - radius:center_row + radius, center_col - radius:center_col + radius] = 0

# Apply high-pass filter
f_shift_high = f_shift * mask_high
f_transform_high = np.fft.ifftshift(f_shift_high)
image_high = np.fft.ifft2(f_transform_high)
image_high = np.abs(image_high)

# Ideal low-pass filter
mask_low = np.zeros((rows, cols), np.uint8)
mask_low[center_row - radius:center_row + radius, center_col - radius:center_col + radius] = 1

# Apply low-pass filter
f_shift_low = f_shift * mask_low
f_transform_low = np.fft.ifftshift(f_shift_low)
image_low = np.fft.ifft2(f_transform_low)
image_low = np.abs(image_low)

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(132), plt.imshow(image_high, cmap='gray')
plt.title('High-pass Filtered'), plt.axis('off')

plt.subplot(133), plt.imshow(image_low, cmap='gray')
plt.title('Low-pass Filtered'), plt.axis('off')

plt.show()
