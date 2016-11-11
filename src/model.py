from keras.model import Model
from keras.Layers import Input, Dense


def model(input_shape=None, dropout=0.5):
    '''
    input_shape: a tuple determining the shape of the input layer
    '''
    inputs = Input(shape=shape(input_shape))
    output = Dense(output_shape=shape(input_shape))(inputs)
    model = Model(input=inputs, output=output)
    return model


def shape(input_shape):
    shape = (119, 2) if input_shape is None else input_shape
    return shape
