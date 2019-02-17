import cv2
import numpy as np


filename='ML-Dataset#2/M1_F1.3_0402.JPG'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show(img):
    cv2.resize(img,(500,500))
    cv2.imshow("frame",img)
    cv2.waitKey(0)


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

thresh = threshold(img)         
cv2.imwrite("thresh.png",thresh)

'''
Finding Lines
'''

def find_contours(img):
    new_img, contours, h = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return new_img, contours

img, lines = find_contours(thresh)
cv2.imwrite("contours.png",img)
print("Total Contours detected: ",len(lines))

def extra_contour_elimination(lines):
    '''
    Elimination via contour area method
    '''
    contours=[]
    for line in lines:
        a = cv2.contourArea(line)
        if a>0 and a<=975:          # 65*15=975cm**2 (approximated for pixel)
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

final_lines = extra_contour_elimination(lines) 
print("Number of contours after elimincation: ",len(final_lines))
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

distances=[]
indexes=[]
for line in final_lines:
    dist,index=closest_node([721,252],line)
    indexes.append(index)
    distances.append(dist)

print(min(distances))
# print(distances)
line=distances.index(min(distances))
contour=indexes[line]
print(final_lines[line][contour])
