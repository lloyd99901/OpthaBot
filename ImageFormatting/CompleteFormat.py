##COMPLETE IMAGE DIRECTORY FORMATTER###
##THIS FILE WILL TAKE A FULL DIRECTORY OF IMAGES AND PREPARE THEM AND OUTPUT THEM INTO A NEW DIRECTORY##


from ImageFormatting.ImageFormatter import cropImageByColorDetection, resizeImage
import os
from os import listdir
from os.path import isfile, join

##PLACE DIRECTORIES HERE##
INPUT_DIRECTORY = r"D:\FinalSet\Training"
OUTPUT_DIRECTORY = r"D:\FinalSet\Compressed"
###

def formatAllImages():

    global INPUT_DIRECTORY, OUTPUT_DIRECTORY #IMPORT VARS

    os.chdir(INPUT_DIRECTORY) #CHANGE DIR TO INPUT DIRECTORY
    onlyfiles = [f for f in listdir(INPUT_DIRECTORY) if isfile(join(INPUT_DIRECTORY, f))] #GRAB ALL FILENAMES IN THE DIRECTORY
    print(onlyfiles)


    #LOOP THROUGH FILE NAMES
    for i in onlyfiles:
        os.chdir(INPUT_DIRECTORY) #GO TO INPUT DIR
        img = resizeImage(cropImageByColorDetection(i)) #GRAB AND PRE PROCESS IMAGE
        os.chdir(OUTPUT_DIRECTORY)#GO TO OUTPUT DIR
        img.save(i) #SAVE IMAGE
        print("FILE " + i + " COMPLETE")



if __name__ == "__main__":
    formatAllImages()