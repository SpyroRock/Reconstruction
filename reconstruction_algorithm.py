from model_structure import create_model
from model_structure_256x64 import create_model_256_64
from keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import multi_gpu_model
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
from numpy import load, save
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import pickle

img_height = 256
img_width = 256

img_height_test = 64
img_width_test = 64

speckle_data = load('speckle_array_case0.npy')
#speckle_data = tf.constant(speckle_data, dtype=tf.uint8)
print(tf.constant(speckle_data))
print(speckle_data.shape)

speckle_labels = load('symbol_array_case0.npy')
#speckle_labels = tf.constant(speckle_labels, dtype=tf.uint8)
print(tf.constant(speckle_labels))
print(speckle_labels.shape)

#plt.imshow(speckle_labels[2], cmap='gray')
#plt.show()
#dictionary = {speckle_labels_n: speckle_labels_mn_n for speckle_labels_n, speckle_labels_mn_n in zip(speckle_labels, speckle_labels_mn)}

X_train, X_test, y_train, y_test = train_test_split(speckle_data, speckle_labels, test_size=0.1, random_state=42)

X_train = X_train.reshape(-1, img_height, img_width, 1)
X_test = X_test.reshape(-1, img_height, img_width, 1)
input_shape = (img_height, img_width, 1)

y_train = y_train.reshape(-1, img_height_test, img_width_test, 1)
y_test = y_test.reshape(-1, img_height_test, img_width_test, 1)
input_shape_test = (img_height_test, img_width_test, 1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_input = Input(shape=input_shape)

reconstruction = create_model_256_64(img_input, input_shape, input_shape_test)
print(reconstruction)

reconstruction.summary()

reconstruction.compile(loss='mean_squared_error',
                        optimizer = 'adam',
                        metrics = ['accuracy'])

reconstruction.fit(X_train, y_train, 
                   batch_size = 50, 
                   epochs = 260, 
                   verbose = 1, 
                   validation_data = (X_test, y_test)) # Data on which to evaluate the loss and any model metrics at the end of each epoch. 
                                                       # The model will not be trained on this data. 
                                                       # This could be a list (x_val, y_val) or a list (x_val, y_val, val_sample_weights). 
                                                       # validation_data will override validation_split.

score = reconstruction.evaluate(X_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test acuracy:', score[1]) 

y_predicted = reconstruction.predict(X_test)

extract = Model(reconstruction.inputs, reconstruction.layers[-1].output) 
features = extract.predict(X_test)
print(features.shape)
save('features_data.npy', features)
save('features_predicted.npy', y_predicted)

reconstruction.save('reconstruction_model')

