import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Define Laplacian kernels
laplacian_kernel_4 = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])

laplacian_kernel_neg_4 = np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])

laplacian_kernel_8 = np.array([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]])

# Apply Laplacian kernels to the image
laplacian_filtered_4 = cv2.filter2D(image, -1, laplacian_kernel_4)
laplacian_filtered_neg_4 = cv2.filter2D(image, -1, laplacian_kernel_neg_4)
laplacian_filtered_8 = cv2.filter2D(image, -1, laplacian_kernel_8)

# Enhance the images by adding Laplacian filtered images to the original image
enhanced_image_4 = cv2.add(image, laplacian_filtered_4)
enhanced_image_neg_4 = cv2.add(image, laplacian_filtered_neg_4)
enhanced_image_8 = cv2.add(image, laplacian_filtered_8)

# Plot the original image and enhanced images
plt.figure(figsize=(15, 15))

# Original Image
plt.subplot(4, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(4, 4, 5)
plt.hist(image.ravel(), bins=256, range=[0,256], color='k', alpha=0.5)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# Laplacian +4
plt.subplot(4, 4, 2)
plt.imshow(enhanced_image_4, cmap='gray')
plt.title('Laplacian +4')
plt.axis('off')
plt.subplot(4, 4, 6)
plt.hist(enhanced_image_4.ravel(), bins=256, range=[0,256], color='k', alpha=0.5)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# Laplacian -4
plt.subplot(4, 4, 3)
plt.imshow(enhanced_image_neg_4, cmap='gray')
plt.title('Laplacian -4')
plt.axis('off')
plt.subplot(4, 4, 7)
plt.hist(enhanced_image_neg_4.ravel(), bins=256, range=[0,256], color='k', alpha=0.5)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# Laplacian +8
plt.subplot(4, 4, 4)
plt.imshow(enhanced_image_8, cmap='gray')
plt.title('Laplacian +8')
plt.axis('off')
plt.subplot(4, 4, 8)
plt.hist(enhanced_image_8.ravel(), bins=256, range=[0,256], color='k', alpha=0.5)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.show()
