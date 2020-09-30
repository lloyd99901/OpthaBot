from ImageFormatting.ImageFormatter import cropImageByColorDetection
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
INPUT_DIRECTORY = r"C:\Users\rajib\OneDrive\Desktop\FinalSet\Testing"
OUTPUT_DIRECTORY = r"C:\Users\rajib\OneDrive\Desktop\FinalSet\Cropped"

def main():

    global INPUT_DIRECTORY, OUTPUT_DIRECTORY
    onlyfiles = [f for f in listdir(INPUT_DIRECTORY) if isfile(join(INPUT_DIRECTORY, f))]
    os.chdir(INPUT_DIRECTORY)
    print(os.getcwd())

    croppedImages = [cropImageByColorDetection(onlyfiles[i]) for i in range(len(onlyfiles))]

    os.chdir(OUTPUT_DIRECTORY)

    for i in range(len(croppedImages)):
        image = croppedImages[i]
        filename = str(i) + ".jpg"
        image.save(filename)
        print(filename)


    return

if __name__ == "__main__":
    main()