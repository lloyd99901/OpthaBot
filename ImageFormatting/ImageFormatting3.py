from PIL import Image, ImageChops
import cv2
import numpy as np

# load image
img = cv2.imread('wideShot.jpg')
#rsz_img = cv2.resize(img, None, fx=0.25, fy=0.25) # resize since image is huge
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
imagem = cv2.bitwise_not(gray)
'''
# threshold to get just the signature
retval, thresh_gray = cv2.threshold(imagem, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# find where the signature is and make a cropped region
points = np.argwhere(thresh_gray==0) # find where the black pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image

# get the thresholded crop
retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
'''

cv2.imwrite("f.jpg", gray)

image=Image.open('f.jpg')
image.load()
'''
image_data = np.asarray(imagem)
image_data_bw = image_data.max(axis=2)
non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

new_image = Image.fromarray(image_data_new)
new_image.save('L_2d_cropped.png')

imageBox = image.getbbox()
cropped=image.crop(imageBox)
cropped.save('L_2d_cropped.png')
# display
#cv2.imwrite("f.jpg", imagem)
#cv2.waitKey(0)
'''


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


new = trim(image)
new.show()
