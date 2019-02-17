import cv2
import numpy as np


filename='ML-Dataset#2/M1_F1.3_0402.JPG'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show(img):
    cv2.resize(img,(500,500))
    cv2.imshow("frame",img)
    cv2.waitKey(0)

def threshold(img,lower_thresh,upper_thresh):
    low = np.array([lower_thresh,lower_thresh,lower_thresh])
    high = np.array([upper_thresh,upper_thresh,upper_thresh])
    mask = cv2.inRange(img,low,high)
    return mask

def differntial_rgb(img):
    blank_rgb = np.zeros(img.shape, np.uint8)
    b = np.array(img[:,:,0], np.int)
    g = np.array(img[:,:,1], np.int)
    r = np.array(img[:,:,2], np.int)
    blank_rgb[:,:,0] = np.absolute(np.subtract(b,r))
    blank_rgb[:,:,1] = np.absolute(np.subtract(g,b))
    blank_rgb[:,:,2] = np.absolute(np.subtract(r,g))
    low = np.array([0,0,0])
    high = np.array([30,30,30])
    mask = cv2.inRange(blank_rgb, low, high)
    return mask


def morphology(img):
    '''
    Dilation: Enlarges bright, white areas by adding pixels
    Erosion: Removes pixels
    '''
    kernel = np.ones([3,3], np.uint8)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) #Useful for closing small holes in image (dilation foloowed by erosion)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel) #Useful in noise reduction (Erosion followed by dilation)
    return opening

mask = threshold(img,200,255)
mask = morphology(mask)
# show(mask)

d_mask = differntial_rgb(img)
d_mask = morphology(d_mask)
# show(d_mask)

thresh = cv2.bitwise_and(mask, d_mask)          #Mask to get minimum  exact white lines
cv2.imwrite("thresh.png",thresh)