from keras.layers import LSTM
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
    input_shape = (n_steps, np.prod(shape))
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=True))
    model.compile(loss='mse', optimizer='sgd')
    return model

if __name__ == '__main__':
    model = model()
