import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('avarage.jpg', cv2.IMREAD_GRAYSCALE)

# Define sigma values
sigma_1 = 1.5
sigma_2 = 10

# Calculate kernel size based on sigma values (you can choose any appropriate size)
kernel_size_1 = int(6 * sigma_1 + 1)
kernel_size_2 = int(6 * sigma_2 + 1)

# Generate 1D Gaussian kernel in the x-direction for sigma_1
gaussian_kernel_x_1 = cv2.getGaussianKernel(kernel_size_1, sigma_1)

# Generate 1D Gaussian kernel in the y-direction for sigma_1
gaussian_kernel_y_1 = cv2.getGaussianKernel(kernel_size_1, sigma_1)

# Generate 1D Gaussian kernel in the x-direction for sigma_2
gaussian_kernel_x_2 = cv2.getGaussianKernel(kernel_size_2, sigma_2)

# Generate 1D Gaussian kernel in the y-direction for sigma_2
gaussian_kernel_y_2 = cv2.getGaussianKernel(kernel_size_2, sigma_2)

# Perform 2D convolution using the separable Gaussian kernels for sigma_1
blurred_image_1 = cv2.sepFilter2D(image, -1, gaussian_kernel_x_1, gaussian_kernel_y_1)

# Perform 2D convolution using the separable Gaussian kernels for sigma_2
blurred_image_2 = cv2.sepFilter2D(image, -1, gaussian_kernel_x_2, gaussian_kernel_y_2)

# Plot original and blurred images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred_image_1, cmap='gray')
plt.title('Blurred Image (Sigma = 1.5)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blurred_image_2, cmap='gray')
plt.title('Blurred Image (Sigma = 10)')
plt.axis('off')

plt.show()
