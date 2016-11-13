from model import input_shape
from train_model import offset
from miditransform import noteStateMatrixToMidi
import numpy as np


def compose(model, length, filename='example', seed=None):
    '''
    INPUT: Keras model, int
    '''
    statematrix = None
    if seed is None:
        seed = np.random.random(input_shape)
        seed = seed[np.newaxis]
    predict = cleanstatematrix(model.predict(seed))
    statematrix = predict[0]
    while len(statematrix) < length:
        predict = cleanstatematrix(model.predict(predict))
        statematrix += predict[0][-offset:]
        break
    noteStateMatrixToMidi(statematrix, filename)


def cleanstatematrix(statematrix):
    sample = np.random.random(statematrix.shape)
    sm = sample < statematrix
    return sm
