import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image and the image with the change
original_image = cv2.imread('medicalpage3a.png', cv2.IMREAD_GRAYSCALE)
changed_image = cv2.imread('medicalpage3b.png', cv2.IMREAD_GRAYSCALE)

# Resize the images to have the same dimensions
height = min(original_image.shape[0], changed_image.shape[0])
width = min(original_image.shape[1], changed_image.shape[1])
original_image_resized = cv2.resize(original_image, (width, height))
changed_image_resized = cv2.resize(changed_image, (width, height))

# Compute the absolute difference between the images
absolute_difference = np.abs(original_image_resized - changed_image_resized)

# Normalize the absolute difference to [0, 1] range for visualization
normalized_difference = absolute_difference / 255.0  # Normalizing to [0, 1] range

# Threshold the absolute difference for black & white representation
_, binary_difference = cv2.threshold(absolute_difference, 0, 255, cv2.THRESH_BINARY)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('original')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(changed_image_resized, cmap='gray')
plt.title('Changed Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(normalized_difference, cmap='gray')
plt.title('Absolute Difference')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(binary_difference, cmap='gray')
plt.title('Absolute Difference (Black & White)')
plt.axis('off')



plt.show()
