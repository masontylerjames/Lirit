from keras.layers import LSTM, Flatten, Reshape, Activation
from keras.models import Sequential
from miditransform import shape
import numpy as np

# each time step is a 32nd note, so 8 of those is a quarter note and
# there are 4 quarter notes in a measure in 4/4 time
n_steps = 8 * 4 * 8


def model():
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # effectively flattens all the state matrices
    model.add(Flatten(input_shape=input_shape))
    model.add(Reshape(flat_shape))
    model.add(LSTM(np.prod(shape), input_shape=input_shape,
                   return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('hard_sigmoid'))
    model.compile(loss='mse', optimizer='sgd')
    return model

if __name__ == '__main__':
    model = model()
