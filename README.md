# Skylark Drone Assignment: GCP detection in Drone Imagery

## Problem statement
Manual detection of Ground Control Points is a cubersome task and thus, a Computer Vision and Deep Learning model can help simplify this task by identifying the points and then plotting their co-ordinates.

Problem with using object detection: The GCP points are very small and thus, normal object detection algorithms could not be applied. There is scope of applying <strong>RetinaNet model</strong>, but that has led to bad accuracy, and great computation since it uses detection and here the method can be optimised using <strong>classification</strong> instead. (Classifying whether it is a GCP(1) or not(0)).

## Method Employed

<blockqoute>
    1. <a href='image_process.py'>Detection of contours in image</a>: The contours can be used to detect white "L" shape GCP and crop that from the image for training images.
        * Step 1 : Thresholding the RGB image and mask it with (220,255): the white mask. The value range 220-255 can be altered for images (this suited best for dataset provided).
        * Step 2: Thresholding is followed by morphology. Used opencv function
    ~~~
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)       #Dilation: Useful for closing small holes in image in thresh image
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel)   #Erosion: Useful in noise reduction
    ~~~
    
    
</blockqoute>
    
