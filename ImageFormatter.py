##IMAGE FORMATTING##

from PIL import Image #Python pillow image library
import PIL.ImageOps  #Additional module from pillow
import cv2 #OpenCV image library
import smartcrop # Image analyis library, allows intelligent cropping
import numpy as np #NumPy lib
import os
from resizeimage import resizeimage

##NOTE NOT ALL OF THESE FUNCTIONS ENDED UP BEING USED - THEY ARE KEPT HERE IN CASE OF USE IN OTHER AREAS##
def convert2Grayscale(img): #converts a cv2 image to grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def invertImage(img): #inverts the image colours for a cv2 image
    return cv2.bitwise_not(img)

def getSmallestSide(img): #gets the smallest side of an image
    width, height = img.size
    print([width,height])
    return min([width,height])

def invertImagePIL(img): #inverts colours of a pillow image
    return PIL.ImageOps.invert(img)

def convert2grayscalePIL(img): #converts colours to greyscale for a pillow image
    return img.convert('LA')

def convertCV2toPIL(opencv_image): #converts a cv2 image to a pillow image
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) #changes colour coding from BGR to RGB
    pil_image = Image.fromarray(color_coverted) #conversion from cv2 to pillow
    return pil_image

def convertPILtoCV2(pil_img): # converts a pillow image into a cv2 image
    numpy_image = np.array(pil_img) # creates a numpy array of rgb values for the image
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) # creates a cv2 BGR image from the RGB array
    return opencv_image

def calculateScore(score):
    return (abs(score["detail"])+abs(score["saturation"])) #generates a score from the smartcrop output by adding detail, saturation and total crop score which gives a pretty good estimate for where the fundus image is

def returnCropCoords(img):
    sc = smartcrop.SmartCrop() #generates a smartcrop object
    result = sc.crop(img, 200, 200) #creates 500 decent crops

    #here we will cycle through the possible crops to see which one is the best
    highestVal = 0 #this will track the highest score
    for i in result["crops"]:
        #print(i)
        x = calculateScore(i["score"]) #calculate score for each crop
        #print(x)
        if x > highestVal: #check if the score is highest or not
            highestVal = x


    #this for loop just grabs the crop that had the best score
    for i in result["crops"]:
        if calculateScore(i["score"]) == highestVal:
            print(highestVal)
            return [i["x"],i["y"],i["width"],i["height"]] #this returns the crop details

def resizeImage(img):
    img = resizeimage.resize_contain(img,[500,500])#resizes images to a width of 500 pixels whilst keeping aspect ratio - eases computing power for crop analysis
    background = Image.new("RGB", img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    #img.show()
    return background

def cropImageByColorDetection(file):  #Credit to jamisonbryant for pycropper
    TOLERANCE = 20 #Tolerance for colour change detection

    # Print out some file information
    image = Image.open(file) #open up the file
    print(image)
    image_width = image.size[0] #get width
    image_height = image.size[1] #get height

    # Sample background color
    def rgb_tuple_to_str(tuple):
        return 'rgb(' + str(tuple[0]) + ', ' + str(tuple[1]) + ', ' + str(tuple[2]) + ')'

    def is_like_bg_color(color):
        #Setting base values
        color_r, color_g, color_b = color[0], color[1], color[2]  # grab rgb value
        bg_r, bg_g, bg_b = bg_color[0], bg_color[1], bg_color[2] # Grab background rgb value
        r_similar, g_similar, b_similar = False, False, False #tracking if the colour is simmilar

        #checking if the colours are simmilar to the background or different
        if color_r in range(bg_r - TOLERANCE, bg_r + TOLERANCE):
            r_similar = True

        if color_g in range(bg_g - TOLERANCE, bg_g + TOLERANCE):
            g_similar = True

        if color_b in range(bg_b - TOLERANCE, bg_b + TOLERANCE):
            b_similar = True

        return r_similar and g_similar and b_similar


    #Sampling the background colour
    pixel_map = image.load() #load in the image


    x_offset = image_width * 0.05
    y_offset = image_height * 0.05

    #sampling the corners of the image for the background colour
    ul_color = pixel_map[x_offset, y_offset]
    ur_color = pixel_map[image_width - x_offset, y_offset]
    ll_color = pixel_map[x_offset, image_height - y_offset]
    lr_color = pixel_map[image_width - x_offset, image_height - y_offset]
    bg_color = ()


    #checking if we have the background colour or not
    #if ul_color == ur_color or ur_color == ll_color or ll_color == lr_color:
    bg_color = ul_color
        #print("Sampled background color: " + rgb_tuple_to_str(ul_color))


    ##Searching for the crop coordinates

    # Search for top edge
    top_edge_coords = []

    for i in range(0, image_width, int(image_width / 10)): #cycle through the cols
        for y in range(0, image_height - 1): #cycle through the rows
            if not is_like_bg_color(pixel_map[i, y]): # check if bg colour is not simmilar
                top_edge_coords.append(y) #add it to the list of top level crop coords
                break

    top_edge_coord = top_edge_coords[0]
    for c in top_edge_coords:
        if c < top_edge_coord:
            top_edge_coord = c


    # Search for bottom edge

    bottom_edge_coords = []

    for i in range(0, image_width, int(image_width / 10)):
        for y in range(image_height - 1, 0, -1):
            if not is_like_bg_color(pixel_map[i, y]):
                bottom_edge_coords.append(y)
                break

    bottom_edge_coord = bottom_edge_coords[0]
    for c in bottom_edge_coords:
        if c > bottom_edge_coord:
            bottom_edge_coord = c


    # Search for left edge


    left_edge_coords = []

    for i in range(0, image_height, int(image_height / 10)):
        for x in range(0, image_width - 1):
            if not is_like_bg_color(pixel_map[x, i]):
                left_edge_coords.append(x)
                break

    left_edge_coord = left_edge_coords[0]
    for c in left_edge_coords:
        if c < left_edge_coord:
            left_edge_coord = c

    # Search for right edge

    right_edge_coords = []

    for i in range(0, image_height, int(image_height / 10)):
        for x in range(image_width - 1, 0, -1):
            try:
                if not is_like_bg_color(pixel_map[x, i]):
                    right_edge_coords.append(x)
                    break
            except IndexError:
                pass

    right_edge_coord = right_edge_coords[0]
    for c in right_edge_coords:
        if c > right_edge_coord:
            right_edge_coord = c


    # Crop the image

    cropped_image = image.crop((left_edge_coord, top_edge_coord, right_edge_coord, bottom_edge_coord))
    #image.show()
    #cropped_image.show()

    return resizeImage(cropped_image)

def cropBySmartcrop(filename): #this version is deprecated - cause it sucks - cause smartcrop sucks
    image = cv2.imread(filename)  # Read the image (As OPENCV)
    image = convertCV2toPIL(image)  # Convert the image to a pillow image
    image = resizeImage(image)
    # image = invertImagePIL(image)
    crops = returnCropCoords(image)  # Grab the crop details
    area = (crops[0], crops[1], crops[0] + crops[2], crops[1] + crops[3])  # variable that stores the crop details
    cropped = image.crop(area)  # Crop the image
    getSmallestSide(image)
    cropped.show()  # Show the image

if __name__ == "__main__":
    os.chdir(r'C:\Users\Public\Documents\PyCharmProjs\LearningTensorflow\ImageFormatting\TestSet')
    cropped = cropImageByColorDetection("9.jpg")
    reduced = resizeImage(cropped)
    reduced.show()