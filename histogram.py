import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image in greyscale
img = cv2.imread('clahe.jpg', 0)

# Apply histogram equalization (uncomment if needed)
# img_2 = cv2.equalizeHist(img)

# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
out = clahe.apply(img)

# Display the images side by side using Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(out, cmap='gray')
axs[1].set_title('CLAHE')
plt.show()
