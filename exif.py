import cv2
import piexif
from PIL import Image

def rotate_jpeg(filename):
    """
    Rotates and overwrites the image

    :param str filename: Absolute path to the image-file
    """
    image = Image.open(filename)
    

    if 'exif' not in image.info:
        return

    exif_dict = piexif.load(image.info["exif"])
    
    if piexif.ImageIFD.Orientation not in exif_dict["0th"]:
        return

    orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
    exif_bytes = piexif.dump(exif_dict)

    print(orientation)

    if orientation == 2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        image = image.rotate(180)
    elif orientation == 4:
        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 7:
        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        image = image.rotate(90, expand=True)

    # TODO: Write the new orientation of the image into the EXIF metadata

    image.save(filename, exif=exif_bytes)

rotate_jpeg('ML-Dataset#2/M1_F1.3_0402.JPG')