import cv2
import numpy as np
import matplotlib.pyplot as plt


# filename='ML-Dataset#1/DJI_0500.JPG'
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# clone = img.copy()

# def show(img):
#     cv2.resize(img,(500,500))
#     cv2.imshow("frame",img)
#     cv2.waitKey(0)


def morphology(img):
    '''
    Dilation: Enlarges bright, white areas by adding pixels
    Erosion: Removes pixels
    '''
    kernel = np.ones([3,3], np.uint8)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) #Useful for closing small holes in image (dilation foloowed by erosion)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel) #Useful in noise reduction (Erosion followed by dilation)
    return opening

def threshold(img):
    '''
    Mask to get minimum  exact white lines
    Range is 200, 255 (hard-coded), can be altered according to dataset
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,mask = cv2.threshold(gray,200,255,0)        
    mask = morphology(mask)
    # show(mask)
    return mask

# thresh = threshold(img)         
# cv2.imwrite("thresh.jpeg",thresh)

'''
Finding Lines
'''

def find_contours(img):
    new_img, contours, h = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return new_img, contours

# img, lines = find_contours(thresh)
# cv2.imwrite("contours.jpeg",img)
# print("Total Contours detected: ",len(lines))

def extra_contour_elimination(lines):
    '''
    Elimination via contour area method
    '''
    contours=[]
    for line in lines:
        a = cv2.contourArea(line)
        if a>0 and a<=850:          # 65*15=975cm**2 (approximated for pixel)
            contours.append(line)

    '''
    Elimination via shape of L, which is concave
    '''
    concave = []
    for line in contours:
        epsilon = 0.01*cv2.arcLength(line,True)
        approx = cv2.approxPolyDP(line,epsilon,True)
        if not cv2.isContourConvex(approx):
            concave.append(line)
    
    return concave

# final_lines = extra_contour_elimination(lines) 
# print("Number of contours after elimination: ",len(final_lines))
# print(final_lines)

'''
Testing for dataset
'''


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = []
    for i in range(len(nodes)):
        dist_2.append(np.linalg.norm(nodes[i]-node))
    dist_2 = np.asarray(dist_2)
    min_dist = np.amin(dist_2)
    index, = np.where(dist_2==min_dist)
    return min_dist,index[0]

def req_contour(final_lines,x,y):
    distances=[]
    indexes=[]
    for line in final_lines:
        dist,index=closest_node([x,y],line)
        indexes.append(index)
        distances.append(dist)

    # print(min(distances))
    # print(distances)
    line=distances.index(min(distances))
    contour=indexes[line]
    # print(final_lines[line][contour])

    required_contour = final_lines[line]
    rejected_contours = np.delete(final_lines,line)
    return required_contour, min(distances), rejected_contours

# required_contour, dist, rejected_contours = req_contour(lines,2701,1590)
# print("req", required_contour)

def crop_contour(required_contour,thresh):
    
    rect = cv2.minAreaRect(np.array(required_contour))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(clone,[box],0,(0,0,255),2)

    # new_img = clone[y:y+h,x:x+h]
    # print(new_img)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(thresh, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW)+2,int(croppedH)+2), (size[0]/2, size[1]/2))
    # cv2.imwrite("req.png",croppedRotated)
    return croppedRotated

# rect = cv2.minAreaRect(required_contour)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(clone,[box],0,(0,0,0),-1)
# cv2.imwrite("req2.jpeg",clone)
# cv2.resize(clone,(500,500))
# cv2.imshow("frame",clone)
# cv2.waitKey(0)

# cv2.imwrite("req.jpeg",crop_contour(required_contour,thresh))