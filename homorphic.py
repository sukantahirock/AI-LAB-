import cv2
import numpy as np
from matplotlib import pyplot as plt

def homomorphic_filter(img, gamma_h, gamma_l, c, d0):
    # Convert image to float32 for processing
    img_float32 = np.float32(img)
    
    # Convert to log domain
    img_log = np.log1p(img_float32)
    
    # Perform Fourier Transform
    f = np.fft.fft2(img_log)
    fshift = np.fft.fftshift(f)
    
    # Construct high-pass filter
    rows, cols = img.shape
    u = np.arange(-cols/2, cols/2)
    v = np.arange(-rows/2, rows/2)
    U, V = np.meshgrid(u, v)
    D_uv = np.sqrt(U**2 + V**2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_uv**2 / d0**2))) + gamma_l
    
    # Apply filter
    filtered_fshift = fshift * H
    
    # Inverse Fourier Transform
    f_filtered = np.fft.ifftshift(filtered_fshift)
    img_filtered = np.fft.ifft2(f_filtered)
    img_filtered = np.real(np.exp(img_filtered) - 1)
    
    return img_filtered

# Read image
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Apply homomorphic filter
gamma_h = 2.0
gamma_l = 0.5
c = 1.0
d0 = 20.0
filtered_img = homomorphic_filter(img, gamma_h, gamma_l, c, d0)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Image')
plt.show()
