from miditransform import midiToStateMatrix
from model import model


def train_one_step(midifile):
    statematrix = midiToStateMatrix(midifile)
    model().fit(statematrix[100], statematrix[101])


def train(midifile, sample_length=None, samples=None):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix, sample_length, samples)


def _train(statematrix, sample_length, samples):
    neuralnet = model()
    return neuralnet
