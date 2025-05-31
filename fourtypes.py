
import matplotlib.pyplot as plt 
import cv2

img1 = cv2.imread('dark.jpg')
img2 = cv2.imread('bright.jpg')
img3 = cv2.imread('lowcontrast.jpg')
img4 = cv2.imread('highcontrast.jpg')
 
  
# calculate frequency of pixels in range 0-255 
hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])  
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])  
hist3 = cv2.calcHist([img3],[0],None,[256],[0,256])  
hist4 = cv2.calcHist([img4],[0],None,[256],[0,256])  
plt.figure(figsize=(8, 4))

plt.subplot(421),plt.imshow(img1);
plt.subplot(422),plt.plot(hist1);
plt.subplot(423),plt.imshow(img2);
plt.subplot(424),plt.plot(hist2);
plt.subplot(425),plt.imshow(img3);
plt.subplot(426),plt.plot(hist3);
plt.subplot(427),plt.imshow(img4);
plt.subplot(428),plt.plot(hist4);
