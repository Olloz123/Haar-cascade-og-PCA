import cv2
import numpy as np

faces = {}
image = cv2.imread('grayface.pgm', cv2.IMREAD_GRAYSCALE)

if image is not None:
    faces['grayface.pgm'] = image
else:
    print("Image not loaded.")

