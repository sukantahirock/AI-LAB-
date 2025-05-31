import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV
image_path = r'C:\Users\USER\Documents\MATLAB\lena.png'
image = cv2.imread(image_path)

# Convert BGR to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Split the image into its color channels
blue_channel, green_channel, red_channel = cv2.split(image_rgb)

# Plot original and color channels
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')

plt.subplot(1, 4, 3)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')

plt.subplot(1, 4, 4)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')

plt.show()

# Plot original and grayscale
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Scale')

plt.show()
