from distutils.version import LooseVersion
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import multi_gpu_model
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from numpy import load, save
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import pickle

# assert LooseVersion(tf.__version__) >= LooseVersion('1.0')
# print('Tensorflow Version: {}'.format(tf.__version__))

# #Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found')
# else:
#     print('Deafault GPU Device: {}'.format(tf.test.gpu_device_name()))


# model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))

img_height = 256
img_width = 256

img_height_test = 64
img_width_test = 64

speckle_data = load('speckle_array_case0.npy')
print(speckle_data.shape)
#speckle_labels = load('speckle_labels.npy')
speckle_labels = load('symbol_array_case0.npy')
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


reconstruction = Sequential()
reconstruction.add(Conv2D(128, 1, activation = 'relu', input_shape=input_shape))
reconstruction.add(MaxPooling2D(pool_size = (2,2)))
reconstruction.add(Dropout(0.25))
reconstruction.add(Conv2D(128, 1, activation = 'relu'))
reconstruction.add(Reshape((1, 128*128*128), input_shape=(128, 128, 128)))
reconstruction.add(Dense(16384, activation='relu'))
reconstruction.add(Dropout(0.25))
reconstruction.add(Dense(4096, activation='relu'))
reconstruction.add(Dropout(0.25))
reconstruction.add(Dense(4096, activation='relu'))
reconstruction.add(Dropout(0.25))
reconstruction.add(Reshape((64, 64, 1), input_shape=(1, 4096)))
reconstruction.add(Conv2D(64, 1, activation = 'relu', input_shape=input_shape_test))
reconstruction.add(Dropout(0.25))
reconstruction.add(Conv2D(32, 1, activation = 'relu'))
reconstruction.add(Dropout(0.25))
reconstruction.add(Conv2D(16, 1, activation = 'relu'))
reconstruction.add(Dropout(0.05))
reconstruction.add(Conv2D(1, 1, activation = 'relu'))

reconstruction.summary()
#parralel_model = multi_gpu_model(model, gpus = 2)
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
#y_test_class = np.argmax(y_test, axis=1)
#y_predicted_class = np.argmax(y_predicted, axis=1)

#print(confusion_matrix(y_test_class, y_predicted_class))

#plt.imshow(dictionary[y_predicted_class[9]], cmap='gray')
#plt.show()

extract = Model(reconstruction.inputs, reconstruction.layers[-1].output) # Dense(128,...)
features = extract.predict(X_test)
print(features.shape)
save('features_data', features)
save('features_predicted', y_predicted)

pickle_out = open('reconstruction.pkl','wb')
pickle.dump(reconstruction, pickle_out)
pickle_out.close()
reconstruction.predict(X_test)
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")