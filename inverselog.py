import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread('spine.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if input_image is None:
    print("Error: Could not open or read the image.")
    exit()

# Define multiple gamma values
gamma_values = [1.5, 1.0, 0.5]

# Display the images using Matplotlib
plt.figure(figsize=(15, 5))

# Plot the original image
plt.subplot(1, len(gamma_values) + 1, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Loop over each gamma value
for i, gamma in enumerate(gamma_values, 1):
    # Apply gamma correction
    gamma_corrected_image = np.power(input_image, gamma)

    # Convert back to uint8 and scale to the 0-255 range
    gamma_corrected_image = np.uint8(255 * (gamma_corrected_image / np.max(gamma_corrected_image)))

    # Plot the images
    plt.subplot(1, len(gamma_values) + 1, i + 1)
    plt.imshow(gamma_corrected_image, cmap='gray')
    plt.title(f'Gamma = {gamma}')
    plt.axis('off')

plt.show()
