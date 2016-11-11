from miditransform import midiToStateMatrix
from model import model, n_steps
import numpy as np


def train(midifile):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix)


def _train(statematrix):
    neuralnet = model()
    train = np.array([[[entry for note in state for entry in note]
                       for state in statematrix[0:n_steps]]])
    test = np.array([[[entry for state in state for entry in state]
                      for state in statematrix[1: n_steps + 1]]])
    neuralnet.fit(train, test)
    return neuralnet

if __name__ == '__main__':
    neuralnet = train('../data/train/mozart/mz_311_1_format0.mid')
