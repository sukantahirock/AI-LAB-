import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Apply 2D DCT
dct = cv2.dct(np.float32(image))

# Visualize the DCT coefficients
plt.figure(figsize=(8, 6))
plt.imshow(np.log(np.abs(dct)), cmap='gray')
plt.title('DCT Coefficients')
plt.colorbar()
plt.show()

# (Optional) Reconstruct the image from DCT coefficients
reconstructed_image = cv2.idct(dct)

# Show original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('DCT Image')
plt.show()
