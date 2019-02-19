import cv2

image = cv2.imread('ML-Dataset#2/M1_F1.3_0402.JPG')
print(image)
cv2.rectangle(image,(697,236),(747,272),(0,255,255), 5)
cv2.imshow("frame",image)
cv2.waitKey(0)