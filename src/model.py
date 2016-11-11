from keras.models import Sequential
from keras.Layers import LSTM
from miditransform import shape


def model(dropout=0.5):
    '''
    input_shape: a tuple determining the shape of the input layer
    '''
    model = Sequential()
    model.add(LSTM(output_shape=shape, input_shape=shape, activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    return model
