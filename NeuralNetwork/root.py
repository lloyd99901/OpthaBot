from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from DataGrabbing.DataFormatting import generateStructures
from ImageFormatting.ImageFormatter import cropImageByColorDetection
import win32file
import numpy as np
import gc

#PREPARATION
gc.collect()
win32file._setmaxstdio(2048) #REMOVE WINDOWS RAM ACCESS LIMIT
print(tf.version) #CHECK TFLOW IS WORKING
#

CLASS_NAMES = ["0","1","2","3","4","5","6","7"]

DATA_GEN = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True
)

def twoD2threeD(array): #Converts a 2d flat array into a 3d array
    return np.reshape(list(array.getdata()), (256, 256, 3)).tolist()

def nparray2list(array): #CONVERT 3D NUMPY ARRAY INTO A LIST
    return [np.array(i).tolist() for i in array]

def singleTFlowImage(image): #Simply crops and prepares one image into a suitable array - for testing purposes only
    return twoD2threeD(cropImageByColorDetection(image))


def TFlowDataFormatting(cutOff): #TENSOR FLOW DATA FORMATTER
    print("Obtaining data")
    AllImages, AllLables = generateStructures(r"D:\FinalSet\trainingData.xlsx",r"D:\FinalSet\Compressed",cutOff)

    print("Splitting Data")
    listSlice = round(cutOff*0.9) #Decides how much of our data will be reserved for testing and how much for training.

    ##these just slices our whole set into our training and data sets
    trainData = AllImages[0:listSlice]
    testData = AllImages[listSlice:-1]

    trainLabel = AllLables[0:listSlice]
    testLabel = AllLables[listSlice:-1]

    ##Will add data augumentation later
    #trainSet = DATA_GEN.flow(trainData,trainLabel,batch_size=32)  # this will augument the data - making the training data bigger than it seems
    #testSet =  DATA_GEN.flow(testData,testLabel,batch_size=32)

    print("Converting data into matrices")
    #Converts our images into numpy arrays
    trainData = [tf.keras.preprocessing.image.img_to_array(i) for i in trainData]
    testData = [tf.keras.preprocessing.image.img_to_array(i) for i in testData]
    
    #trainData = np.array(trainData).reshape(-1,256,256,3)
    #testData = np.array(testData).reshape(-1,256,256,3)
    #Converts the numpy arrays into native lists. - I will update the model to take numpy arrays or tensors at another time
    #trainData = [nparray2list(i) for i in trainData]
    #testData = [nparray2list(i) for i in testData]

    print("Done")
    #print(trainSet)

    return trainData,trainLabel,testData,testLabel


def runModel(train1,train2):

    global DATA_GEN

    print("Initiallizing Model")
    model = models.Sequential()
    model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(12, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.summary()

    '''
    model.sequential( #going to rebuild model here
        [

        ]
    )
    '''
    print("Compiling Model - This may take some time")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print("Running Model - This may take a long time")

    history = model.fit(train1,train2, epochs=100)
                        #,validation_data=(test1,test2))
    return model, history


if __name__ == "__main__":
    print("STARTING")
    print("~~GATHERING/PROCESSING DATASETS~~")
    x1,x2,y1,y2 = TFlowDataFormatting(15)
    print("~~STARTING NEURAL NETWORK~~")
    model, history = runModel(x1,x2)
    print("~~TESTING MODEL~~")
    test_loss, test_acc = model.evaluate(y1, y2, verbose=2)
    print(test_acc)