from keras.layers import LSTM, Input
from keras.models import Sequential
from miditransform import shape


def model():
    '''
    OUTPUT: a compiled model
    '''
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(LSTM(output_shape=shape, activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.compile(loss='mse', optimizer='sgd')
    return model
