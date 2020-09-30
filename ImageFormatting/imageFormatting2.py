import json


import smartcrop
from PIL import Image

def getSmallestSide(filepath):
    width, height = Image.open(filepath).size
    return min([width,height])
x = getSmallestSide('wideShot.jpg')


import numpy as np

image=Image.open('wideShot.jpg')
image.load()

image_data = np.asarray(image)
image_data_bw = image_data.max(axis=2)
non_empty_columns = np.where(image_data_bw.max(axis=0)<255)[0]
non_empty_rows = np.where(image_data_bw.max(axis=1)<255)[0]
cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

new_image = Image.fromarray(image_data_new)
new_image.save('L_2d_cropped.png')
