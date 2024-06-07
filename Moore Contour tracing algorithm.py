import matplotlib.pyplot as plt
import numpy as np
import cv2

chirag = cv2.imread("path\\image.jpeg")
gray_img = cv2.cvtColor(chirag, cv2.COLOR_BGR2GRAY)
_,img = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)
img = cv2.medianBlur(img, 3)
B = []
starting_pixel = list(zip(*np.where(img == 255))) 
B_x = starting_pixel[0][0]
B_y = starting_pixel[0][1]
B.append(starting_pixel[0])

b_x = starting_pixel[0][0]
b_y = starting_pixel[0][1] - 1

d_x = b_x - B_x
d_y = b_y - B_y

prev_x = b_x  # Initialize prev_x
prev_y = b_y  # Initialize prev_y

offsetTable = [(0, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
count = 1
contour = np.zeros((chirag.shape))
while(count <= 2):
    id = 0
    for i in range(9):
        if(d_x == offsetTable[i][0] and d_y == offsetTable[i][1]):
            id = i            
            break
    
    while(True):
        if(id == 8):
            id = 0       
        c_x = B_x + offsetTable[id + 1][0]
        c_y = B_y + offsetTable[id + 1][1]        
        if (img[c_x][c_y] != 0):
            B_x = c_x
            B_y = c_y                       
            temp = (B_x, B_y)           
            if (temp in B) == True:
                count += 1
            B.append((B_x, B_y))
            lenth = len(B)
            if (lenth % 100 == 0):
                for pixel in range(len(B)):
                    contour[B[pixel][0]][B[pixel][1]] = [0, 255, 0]
                cv2.imshow('Image_contour', contour)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            d_x = prev_x - B_x
            d_y = prev_y - B_y
            break
        prev_x = c_x
        prev_y = c_y        
        id += 1

for pixel in range(len(B)):
    contour[B[pixel][0]][B[pixel][1]] = [0, 255, 0]

cv2.imshow('Image_contour', contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
