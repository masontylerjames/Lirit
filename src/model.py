from keras.model import Model
from keras.Layers import Input, Dense


def model(input_shape=None, dropout=0.5):
    '''
    input_shape: a tuple determining the shape of the input layer
    '''
    model = Model()
    inputs = make_input_layer(input_shape)
    predictions = Dense()
    pass


def make_input_layer(input_shape):
    shape = (119, 2) if input_shape is None else input_shape
    return Input(shape=shape)
