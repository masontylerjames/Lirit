from keras.layers import LSTM, Input
from keras.models import Sequential
from miditransform import shape


def model():
    '''
    OUTPUT: a compiled model
    '''
    model = Sequential()
    # model.add(Input(shape=shape))
    model.add(LSTM(shape, input_shape=shape))
    model.compile(loss='mse', optimizer='sgd')
    return model
