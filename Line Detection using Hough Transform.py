import cv2
import numpy as np


img=cv2.imread("C:\\Users\\madda\\OneDrive\\Desktop\\CVIP FINAL\\chessboard.jpg")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges=cv2.Canny(gray, 50,150, apertureSize=3)

lines=cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
cv2.imshow('Line detection', img)
cv2.waitKey(0)
