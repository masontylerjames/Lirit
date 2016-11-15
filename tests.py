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


def notes_on(lirit, n=0, **kwargs):
    train_one(lirit)
    probs = predict(lirit)
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    X, Y = generateXY(sm, 256, 128)
    X = np.array(X)
    Y = np.array(Y)
    pred = cleanstatematrix(probs)
    notecount1 = np.array([np.sum(state.T[1]) for state in X[0]])
    notecount2 = np.array([np.sum(state.T[1]) for state in pred[0]])
    print notecount1.mean(), notecount2.mean()

if __name__ == '__main__':
    lirit = Lirit()
