from image_process import (
    threshold,
    morphology,
    find_contours,
    extra_contour_elimination,
    crop_contour,
    req_contour
)

import csv
import os
import cv2
import numpy as np

path = os.path.join(os.getcwd(),'Dataset/L')
path2 = os.path.join(os.getcwd(),'Dataset/nL')

d=1
def make_datatset(filename,x,y,c,d): 
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    thresh = threshold(img)
    img, lines = find_contours(thresh)
    final_lines = extra_contour_elimination(lines)
    required_contour, dist, rejected_contour = req_contour(final_lines,x,y)
    L = crop_contour(required_contour,thresh)
    name=str(c)+'.jpeg'
    cv2.imwrite(os.path.join(path,name),L)
    for i in range(len(rejected_contour)):
        nL = crop_contour(rejected_contour[i],thresh)
        name = str(d)+'.jpeg'
        cv2.imwrite(os.path.join(path2,name),nL)
        d+=1


file = 'data2.csv'
c = []
with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for i in reader:
        filename = i[0]
        x = int(i[1])
        y = int(i[2])
        c.append([filename,x,y])
f.close()

for i,data in enumerate(c):
    make_datatset(data[0],data[1],data[2],i,d)
