from keras.layers import LSTM, Activation, Reshape
from keras.models import Sequential
import numpy as np


def newmodel(n_steps, shape):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(
        Reshape((n_steps, 1, shape[0], shape[1]), input_shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model
