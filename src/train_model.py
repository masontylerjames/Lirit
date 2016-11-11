from miditransform import midiToStateMatrix
from model import model, n_steps


def train(midifile):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix)


def _train(statematrix):
    neuralnet = model()
    train = statematrix[0:n_steps]
    test = statematrix[1:n_steps + 1]
    neuralnet.fit(train, test)
    return neuralnet
