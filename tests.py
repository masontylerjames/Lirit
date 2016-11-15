from Lirit import Lirit
from src.miditransform import midiToStateMatrix
from src.fit import generateXY, cleanstatematrix
import numpy as np


def train_one(lirit, **kwargs):
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    X, Y = generateXY(sm, 256, 128)
    lirit.model.fit(np.asarray(X), np.asarray(Y), **kwargs)


def predict(lirit, n=0):
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    X, Y = generateXY(sm, 256, 128)
    return lirit.model.predict(np.asarray([X[n]]))

if __name__ == '__main__':
    lirit = Lirit()
