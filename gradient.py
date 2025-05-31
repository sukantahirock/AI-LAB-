import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load an example image
image = plt.imread('lena.png')

# Convert the image to grayscale if it's RGB
if len(image.shape) == 3:
    image = np.mean(image, axis=2)

# Sobel Operator
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Roberts Operator
roberts_x = np.array([[1, 0],
                      [0, -1]])

roberts_y = np.array([[0, 1],
                      [-1, 0]])

# Prewitt Operator
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

# Apply the gradient kernels to compute the gradients
sobel_gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
sobel_gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm')

roberts_gradient_x = convolve2d(image, roberts_x, mode='same', boundary='symm')
roberts_gradient_y = convolve2d(image, roberts_y, mode='same', boundary='symm')

prewitt_gradient_x = convolve2d(image, prewitt_x, mode='same', boundary='symm')
prewitt_gradient_y = convolve2d(image, prewitt_y, mode='same', boundary='symm')

# Plot the original image and gradients
plt.figure(figsize=(15, 10))

plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(sobel_gradient_x, cmap='gray')
plt.title('Sobel Gradient X')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(sobel_gradient_y, cmap='gray')
plt.title('Sobel Gradient Y')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(np.abs(sobel_gradient_x) + np.abs(sobel_gradient_y), cmap='gray')
plt.title('Sobel Gradient Magnitude')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(roberts_gradient_x, cmap='gray')
plt.title('Roberts Gradient X')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(roberts_gradient_y, cmap='gray')
plt.title('Roberts Gradient Y')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(np.abs(roberts_gradient_x) + np.abs(roberts_gradient_y), cmap='gray')
plt.title('Roberts Gradient Magnitude')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(prewitt_gradient_x, cmap='gray')
plt.title('Prewitt Gradient X')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(prewitt_gradient_y, cmap='gray')
plt.title('Prewitt Gradient Y')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(np.abs(prewitt_gradient_x) + np.abs(prewitt_gradient_y), cmap='gray')
plt.title('Prewitt Gradient Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()
