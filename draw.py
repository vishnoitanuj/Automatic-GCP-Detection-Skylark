import numpy as np
import cv2
# def draw_circle(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y
#         print(x,y)

# filename='contours.jpeg'
# img = cv2.imread(filename)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#     elif k == ord('a'):
#         print(mouseX,mouseY)

img = cv2.imread('plot/DJI_0036.JPG')
cv2.circle(img,(1721, 2633), 63, (0,0,255), 1)

cv2.imwrite('test_images/point.jpeg',img)