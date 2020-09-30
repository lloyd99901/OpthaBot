from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from random import randint


#a sample is a data set
#a label is an outcome
#

print(tf.version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

# import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

#Basic TF algos

##Linear regression

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#dft sets have the features and the ytrain/eval has the labels
#The goal is to predict what sort of passenger would have survived the titanic given passenger manifest


#First we obviously gotta filter qunatitative and qualitative data into categorial and numerical data

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = [] #this is like metadata for each columnm
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#The training process
#We feed data into the model in batches according to the number of epochs
#an epoch is a stream of data in the data set, the number of epochs is the number of times the model will see the dataset , mulitple epochs are used to train the model better


#an input function is a function that defines how the data will be fed in batches and epochs


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


#Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) #passing the column metadata in

#Training the model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

print(result)

#Predictions
result = list(linear_est.predict(eval_input_fn)) #gives a list of dicts of the evalutations
print(result)
