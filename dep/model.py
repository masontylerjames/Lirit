from keras.layers import LSTM, Activation, Reshape
from keras.models import Sequential
from miditransform import shape
import numpy as np

# each time step is a 32nd note, so 8 of those is a quarter note and
# there are 4 quarter notes in a measure in 4/4 time
n_steps = 8 * 4 * 8
input_shape = (n_steps, shape[0], shape[1])


def model():
    '''
    OUTPUT: a compiled model
    '''
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model

if __name__ == '__main__':
    model = model()
