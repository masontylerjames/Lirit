from miditransform import midiToStateMatrix
from model import model, n_steps
import numpy as np


def train(midifile):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix)


def _train(statematrix):
    neuralnet = model()
    train = [np.flatten(state) for state in statematrix[0:n_steps]]
    test = [np.flatten(state) for state in statematrix[1:n_steps + 1]]
    neuralnet.fit(train, test)
    return neuralnet

if __name__ == '__main__':
    model = train('../data/train/mozart/mz_311_1_format0.mid')
