import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('point.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Harris corner detection parameters (adjust blockSize and ksize if needed)
blockSize = 2  # Neighborhood size for calculating corner response
ksize = 3  # Aperture parameter for Sobel derivative
k = 0.04  # Harris corner response threshold parameter

# Apply Harris corner detection
corners = cv2.cornerHarris(gray_image, blockSize, ksize, k)

# Thresholding and marking corners (adjust threshold value)
threshold = 0.1 * corners.max()  # Set threshold based on a proportion of the maximum corner response
corners_image = np.copy(image)
corners_image[corners > threshold] = [0, 0, 255]  # Mark corners with red

# Display results
plt.figure(figsize=(8, 6))

plt.subplot(121), plt.imshow(image), plt.title("Original Image")
plt.xticks([]), plt.yticks([])  # Hide ticks

plt.subplot(122), plt.imshow(corners_image), plt.title("Harris Corners")
plt.xticks([]), plt.yticks([])  # Hide ticks

plt.tight_layout()
plt.show()