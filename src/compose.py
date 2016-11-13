from model import input_shape
from model_train import offset
import numpy as np


def compose(model, length, seed=None):
    '''
    INPUT: Keras model, int
    '''
    statematrix = None
    if seed is None:
        seed = np.random.random(input_shape)
    return model.predict(seed)
