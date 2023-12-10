#kilde: https://geeksforgeeks.org/python-haar-cascades-for-object-detection/

# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the cascade classifiers for face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create a function to detect face
def adjusted_detect_face(img):
    face_img = img.copy()

    face_rect = face_cascade.detectMultiScale(face_img,
                                              scaleFactor=1.2,
                                              minNeighbors=5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 5) # sidste tal er tykkelse af box

    return face_img, face_rect


# Reading in the image and creating copies
img = cv2.imread('mig.jpg')
img_copy1 = img.copy()


# Detecting the face
face, face_rect = adjusted_detect_face(img_copy1)

# Saving the image
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

plt.imshow(face)
plt.show()  # Add this line to display the image
cv2.imwrite('../../Desktop/haarcascade/haarcascade/face.jpg', face)

############################################################################


img = Image.open('../../Desktop/haarcascade/haarcascade/face.jpg')

#crops image to size of face
crop_rectangle = (face_rect[0][0],face_rect[0][1],face_rect[0][0] + face_rect[0][2], face_rect[0][1] + face_rect[0][3])
cropped_im = img.crop(crop_rectangle)

facescale = cropped_im.resize((92,112))

grayface = facescale.convert('L')

grayface.show()

grayface.save('grayface.pgm')
