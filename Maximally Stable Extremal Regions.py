import os
import sys
import cv2
import numpy as np

from PIL import Image


vis = cv2.imread("path\\chessboard.jpg")
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(vis)
for p in regions:
    xmax, ymax = np.amax(p, axis=0)
    xmin, ymin = np.amin(p, axis=0)
    cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
cv2.imshow("MSER regions", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
