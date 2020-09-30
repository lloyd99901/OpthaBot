from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from DataStructureFormatting.DataFormatting import generateStructures
from ImageFormatting.ImageFormatter import cropImageByColorDetection
import win32file
import numpy as np
import gc

#PREPARATION
gc.collect()
win32file._setmaxstdio(2048) #REMOVE WINDOWS RAM ACCESS LIMIT
#print(tf.version) #CHECK TFLOW IS WORKING
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
    AllImages, AllLables = generateStructures(r"D:\FinalSet\trainingData.xlsx",r"D:\FinalSet\Compressed",cutOff)

    listSlice = round(cutOff*0.9)

    trainData = AllImages[0:listSlice]
    testData = AllImages[listSlice:-1]

    trainLabel = AllLables[0:listSlice]
    testLabel = AllLables[listSlice:-1]

    #trainSet = DATA_GEN.flow(trainData,trainLabel,batch_size=32)  # this will augument the data - making the training data bigger than it seems
    #testSet =  DATA_GEN.flow(testData,testLabel,batch_size=32)

    trainSet = [tf.keras.preprocessing.image.img_to_array(i) for i in trainData]
    testSet= [tf.keras.preprocessing.image.img_to_array(i) for i in testData]

    trainSet = [nparray2list(i) for i in trainSet]
    testSet = [nparray2list(i) for i in testSet]
    print("Done")
    #print(trainSet)

    return trainSet,trainLabel,testSet,testLabel


def runModel(train1,train2,test1,test2):

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

    history = model.fit(train1,train2, epochs=50)
                        #,validation_data=(test1,test2))
    return model, history



##GRAB DATA

#print(tf.version)
print("Grabbing Data")
#trainData, trainLabel = generateStructures(r"G:\Desktop\DataSets\Useful Datasets\FinalSet\trainingData.xlsx",r"G:\Desktop\DataSets\Useful Datasets\FinalSet\Cropped",7269)
#testData, testLabel = generateStructures(r"G:\Desktop\DataSets\Useful Datasets\FinalSet\testingData.xlsx",r"G:\Desktop\DataSets\Useful Datasets\FinalSet\Cropped",23)
print("Data Grabbed")

print("Forming Tensor Structures")
#trainDataset = tf.data.Dataset.from_tensor_slices((trainData,trainLabel))
#testDataset = tf.data.Dataset.from_tensor_slices((testData,testLabel))
print("Done!")

#print(trainData)
'''
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(testData,  testLabel, verbose=2)


print(test_acc)
'''

if __name__ == "__main__":
    x1,x2,y1,y2 = TFlowDataFormatting(2000)
    model, history = runModel(x1,x2,y1,y2)
    test_loss, test_acc = model.evaluate(y1, y2, verbose=2)
    print(test_acc)