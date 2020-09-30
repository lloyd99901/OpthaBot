from Katna.image import Image as KatnaImage
from PIL import Image
import os
import matplotlib.pyplot as plt

img = KatnaImage()




import cv2
import numpy as np

import cv2
import numpy as np

'''
# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('wideShot.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Obtain bounding rectangle and extract ROI
x,y,w,h = cv2.boundingRect(thresh)
cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
ROI = original[y:y+h, x:x+w]

# Add alpha channel
b,g,r = cv2.split(ROI)
alpha = np.ones(b.shape, dtype=b.dtype) * 50
ROI = cv2.merge([b,g,r,alpha])

#cv2.imshow('thresh', thresh)
#cv2.imshow('image', image)
plt.imshow(thresh)
plt.show()
'''

file_name = "wideShot.jpg"

def getSmallestSide(filepath):
    width, height = Image.open(filepath).size
    return min([width,height])
x = getSmallestSide('wideShot.jpg')
print(x)

cropList = img.crop_image('wideShot.jpg',x,x,1)

chosenOne = cropList[0]
currentDir = os.getcwd()
ext = '.png'
name = 'new'

'''
src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite("test.png", dst)
'''