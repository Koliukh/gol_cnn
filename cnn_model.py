from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from gol_conf import *
import numpy as np

def cnn_model_fn(width, height, n_samples):
    model = Sequential()
    model.add(Conv2D(20, (3, 3), input_shape=(width+2, height+2, 1), padding='same', activation='relu', name='conv_layer1'))
    model.add(Conv2D(1, (3, 3), padding='valid', activation='relu', name='ouput_layer'))  
    return model