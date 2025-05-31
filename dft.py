import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image using OpenCV
image = cv2.imread('fourier.jpg', cv2.IMREAD_GRAYSCALE)

# Perform DFT
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Compute phase spectrum
phase_spectrum = np.angle(dft_shift)
phase_spectrum_single_channel = phase_spectrum[:,:,0]

# Compute inverse DFT
inverse_dft_shift = np.fft.ifftshift(dft_shift)
inverse_dft = cv2.idft(inverse_dft_shift)
inverse_dft = cv2.magnitude(inverse_dft[:, :, 0], inverse_dft[:, :, 1])

# Plot the original image, magnitude spectrum, phase spectrum, and the inverse DFT image
plt.figure(figsize=(12, 8))

plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(phase_spectrum_single_channel, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(inverse_dft, cmap='gray')
plt.title('Inverse DFT'), plt.xticks([]), plt.yticks([])

plt.show()
