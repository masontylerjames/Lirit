from model import input_shape
from train_model import offset
from miditransform import noteStateMatrixToMidi
import numpy as np


def compose(model, length, seed=None):
    '''
    INPUT: Keras model, int
    '''
    statematrix = None
    if seed is None:
        seed = np.random.random(input_shape)
        seed = seed[np.newaxis]
    predict = model.predict(seed)
    statematrix = cleanstatematrix(predict)
    noteStateMatrixToMidi(statematrix)


def cleanstatematrix(statematrix):
    sample = np.random.random(statematrix.shape)
    sm = sample < statematrix
    return sm
