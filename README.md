# Skylark Drone Assignment: GCP detection in Drone Imagery

### For Full detailed report please refer <a href='#'>[Skylark GCP Report.pdf]</a>

## Problem statement
Manual detection of Ground Control Points is a cumbersome task and thus, a Computer Vision and Deep Learning model can help simplify this task by identifying the points and then plotting their co-ordinates.

Problem with using object detection: The GCP points are very small and thus, normal object detection algorithms could not be applied. There is scope of applying <strong>RetinaNet model</strong>, but that has led to bad accuracy, and great computation since it uses detection and here the method can be optimised using <strong>classification</strong> instead. (Classifying whether it is a GCP(1) or not(0)).

## Method Employed

### Pre-process Flowchart refer: <a href=''>pre-process flow chart.pdf</a>

1. <a href='image_process.py'>Detection of contours in image</a>: The contours can be used to detect white "L" shape GCP and crop that from the image for training images.

>* Step 1 : Thresholding the RGB image and mask it with (220,255): the white mask. The value range 220-255 can be altered for images (this suited best for dataset provided).

>* Step 2: Thresholding is followed by morphology. Used opencv function

~~~~
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)       #Dilation: Useful for closing small holes in image in thresh image
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel)   #Erosion: Useful in noise reduction
~~~~

>* Step 3: Find contours in the threshold image received from above.

>* Step 4: Eliminate extra contours on basis of area and concativity. Since "L" is a concave structure. Rejected contours also are taken into consideration for developing train module for 0 (not L).

>* Step 5: The required contours are then matched with given GCP for closest node. Here the required bounding box is attatched.

>* Step 6: Crop contour of the required bounding box and prepare  training module for 1 (not L).

<strong>Note:</strong> The coordinates are rounded to nearest integer to get output.

## Training Module
<a href='training.ipynb'>Notebook</a>
The learning model is a Sequential model with 5 convolutional layers build in keras.  To get a single max probability output, softmax is used. The filters kernel size in the convolutional layer are on trial and test basis. The training data has 286 positive samples and 213 negative samples contained in repository 'data/train'. The test samples has 8 samples each of negative and positive L, contained in repository 'data/test'. The model summary is
![model_summary](images/model_summary.png)

>* The images tested from <a href='CV-Assignment-Dataset'>CV-Assignment-Dataset</a> are stored with marked detection in <a href='/plot'>plot</a> directory.

>* The model is saved as <a href='model.hdf5'>model.hdf5</a> file.

## GCP Extraction
<a href='gcp_extraction.ipynb'>Notebook</a>

>* Fill the original directory in orig_dir variable.
>* Fill the destination of plot directory in dest_dir.
>* The pre-processing is same as training.
>* The segregated contours are then passed to model for prediction of required shape.
>* The minAreaRect function gives box co-ordinates of the prediction in the threshold image.
>* Detect <strong>Harris Corners</strong> in the predicted contour and eliminate all corners that lie beyond the boundaries of the above calculated box.
>* The Harris corner points are dilated and then passed to a cv2.cornerSubPix() function to get corners with Sub Pixel Accuracy (Source: <a href='https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html'>Link</a>)
>* Calculate <strong>Center of Mass of the predicted contour</strong>, since the  required point is always close to COM.
>* Calculate the corner (detected from Harris) closest to COM, that's the required co-ordinates.
>* Encapsulate the image name and coordinates to <a href='output.csv'>output.csv</a> file.