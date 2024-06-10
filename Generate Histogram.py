import cv2
import matplotlib.pyplot as plt

img=cv2.imread("your_path.jpg",cv2.IMREAD_GRAYSCALE)
hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel value")
plt.xlabel("Frequency")
plt.show()
