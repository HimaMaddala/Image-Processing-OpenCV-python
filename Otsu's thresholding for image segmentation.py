import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread("your_path\\lena.jpg")

# Split the image into its color channels
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal using morphological closing
kernel = np.ones((2,2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

# Threshold to get sure foreground
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

print("The threshold value determined by Otsu's method is:", ret)

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# Display the results
plt.subplot(211)
plt.imshow(rgb_img)
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(212)
plt.imshow(thresh, 'gray')
plt.imsave(r'thresh.png', thresh)
plt.title("Otsu's binary threshold")
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
