from keras.models import Sequential
from keras.Layers import LSTM
from miditransform import shape


def model():
    '''
    OUTPUT: a compiled model
    '''
    model = Sequential()
    model.add(LSTM(output_shape=shape, input_shape=shape, activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.compile(loss='mse', optimizer='sgd')
    return model
