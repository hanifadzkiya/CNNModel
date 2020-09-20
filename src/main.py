from model import Model
from layers.activation import Activation
from layers.max_pooling import MaxPooling
from layers.convolution_layer import ConvolutionLayer
from layers.dense_layer import DenseLayer
from layers.flatten_layer import FlattenLayer


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array

train_ds = image_dataset_from_directory(
        directory='test/',
        labels='inferred',
        label_mode='int',
        batch_size=40,
        shuffle=False,
        image_size=(100, 100))

list_images = []
for images, labels in train_ds.take(1):
    print(labels[0].numpy())        
    for i in range(len(images)):
        list_images.append(images[i].numpy())

        
model = Model()
model.add(ConvolutionLayer(inputs_size=(100,100,3), padding=0, n_filter=3, filter_size=(5,5), n_stride=1))
model.add(Activation())
model.add(MaxPooling((3,3), 3))
model.add(FlattenLayer())
model.add(DenseLayer(units=10, activation='relu'))
model.add(DenseLayer(units=2, activation='relu'))
prediction = model.forward(list_images[0])

print(prediction)
if (prediction[0] > prediction[1]):
    print('kucing')
else:
    print('anjing')