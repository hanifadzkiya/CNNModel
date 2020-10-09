from model import Model
from layers.activation import Activation
from layers.max_pooling import MaxPooling
from layers.convolution_layer import ConvolutionLayer
from layers.dense_layer import DenseLayer
from layers.flatten_layer import FlattenLayer
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array

import pickle

train_ds = image_dataset_from_directory(
        directory='train/',
        labels='inferred',
        label_mode='int',
        batch_size=40,
        shuffle=False,
        image_size=(100, 100))
test = image_dataset_from_directory(
        directory='test/',
        labels='inferred',
        label_mode='int',
        batch_size=40,
        shuffle=False,
        image_size=(100, 100))

train_list_images = []
train_list_labels=[]
for images, labels in train_ds.take(1):
    # print(labels[0].numpy())        
    for i in range(len(images)):
        train_list_images.append(images[i].numpy()*(1/255))
        train_list_labels.append(labels[i].numpy())

test_list_images = []
test_list_labels=[]
for images, labels in test.take(1):
    # print(labels[0].numpy())        
    for i in range(len(images)):
        test_list_images.append(images[i].numpy()*(1/255))
        test_list_labels.append(labels[i].numpy())
model = Model()
model.add(ConvolutionLayer(inputs_size=(100,100,3), padding=0, n_filter=2, filter_size=(3,3), n_stride=1))
model.add(Activation())
model.add(MaxPooling((2,2), 2))
model.add(FlattenLayer())
model.add(DenseLayer(units=8, activation='relu'))
model.add(DenseLayer(units=1, activation='sigmoid'))
# for image in list_images:
#     prediction = model.forward(image)
#     print("prediction : " + str(prediction))
# prediction = model.forward(list_images[0])
from sklearn.metrics import accuracy_score

# split traing 90%
X_train, X_test, y_train, y_test = train_test_split(train_list_images, train_list_labels, test_size=0.1)
model.fit(X_train,y_train,2,3,0.1,0.1)
y_predict = np.zeros(len(y_test))
for i, image in enumerate(X_test):
    predict = model.forward(image)
    if (predict[0] > 0.5):
        y_predict[i] = 1
    else:
        y_predict[i] = 0


accuracy = accuracy_score(y_test,y_predict)
print(accuracy)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_predict = np.zeros(len(y_test))
for i, image in enumerate(X_test):
    predict = loaded_model.forward(image)
    if (predict[0] > 0.5):
        y_predict[i] = 1
    else:
        y_predict[i] = 0

accuracy = accuracy_score(y_test,y_predict)
print(accuracy)

# using data test
y_predict = np.zeros(len(test_list_images))
for i, image in enumerate(test_list_images):
    predict = loaded_model.forward(image)
    if (predict[0] > 0.5):
        y_predict[i] = 1
    else:
        y_predict[i] = 0

accuracy = accuracy_score(test_list_labels,y_predict)
print(accuracy)