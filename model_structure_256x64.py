import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import multi_gpu_model
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

def create_model_256_64(img_input, input_shape, input_shape_test):
    print(input_shape[0])
    #power = find_power(input_shape[0])
    #nb_filter = 2**(power-1)
    #print(power)
    #print(nb_filter)
    # x = Sequential()(img_input)
    x = Conv2D(64, 1, activation = 'relu', input_shape=input_shape)(img_input)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, 1, activation = 'relu')(x)
    x = Reshape((1, 128*128*32), input_shape=(128, 128, 32))(x)
    x = Dense(16384, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Reshape((64, 64, 1), input_shape=(1, 4096))(x)
    x = Conv2D(64, 1, activation = 'relu', input_shape=input_shape_test)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, 1, activation = 'relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(16, 1, activation = 'relu')(x)
    x = Dropout(0.05)(x)
    x = Conv2D(1, 1, activation = 'relu')(x)
    model = Model(img_input, x, name='reconstruction')
    
    return model


def find_power(input_shape):
    power = 0
    i = 2**power
    while(i<input_shape):
        power += 1
        i = 2**power
        
    return power



        