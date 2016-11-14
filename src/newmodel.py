from keras.layers import LSTM, Activation, Convolution3D, SimpleRNN
from keras.layers import Reshape, BatchNormalization
from keras.models import Sequential
import numpy as np


def dep_newmodel(n_steps, shape):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # reshape input to (timestep, rows, cols, channels)
    model.add(Reshape((n_steps, shape[0], shape[
              1], 1), input_shape=input_shape, name='Add channel dim'))

    # Convolution Block 1
    n = 16
    model.add(Convolution3D(n, 1, 12, 2, name='conv1'))
    model.add(BatchNormalization(axis=4, name='bn1'))
    # model.add(LSTM(256, return_sequences=True))
    # model.add(LSTM(np.prod(shape), return_sequences=True))
    # model.add(Reshape(input_shape))
    # model.add(Activation('sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model


def newmodel(n_steps, shape):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(SimpleRNN(np.prod(shape), return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model
    pass

if __name__ == '__main__':
    nn = newmodel(256, (87, 2))
    nn.summary()
