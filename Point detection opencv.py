import cv2
import numpy as np

image = cv2.imread("path\\virat.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (3,3), 0)

laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

log = cv2.Laplacian(blurred, cv2.CV_64F)
log =np.uint8(np.absolute(log))

_, points = cv2.threshold(log, 30, 255, cv2.THRESH_BINARY)

cv2.imshow('Original image', image)
cv2.imshow('Detected image', points)
cv2.waitKey(0)
cv2.destroyAllWindows()
