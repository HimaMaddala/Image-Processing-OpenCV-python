import cv2 
import matplotlib.pyplot as plt


#reading image
img1 = cv2.imread('path_to_image')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
print(keypoints_1)
img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
#cv2.imshow(img_1)
cv2.imshow("Simpe regions", img_1)
cv2.waitKey(0)
