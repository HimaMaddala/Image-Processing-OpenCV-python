import cv2
import numpy as np

img=cv2.imread('your_path\\apple.jpg',0)
kernel=np.ones((3,3),np.uint8)

img_erosion=cv2.erode(img,kernel,iterations=1)
boundary_extracted=img-img_erosion

cv2.imshow('Input image',img)
cv2.imshow('Erosion',img_erosion)
cv2.imshow('Boundary ',boundary_extracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
