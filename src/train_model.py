from miditransform import midiToStateMatrix
from model import model


def train(midifile, sample_length=None, samples=None):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix, sample_length, samples)


def _train(statematrix, sample_length, samples):
    neuralnet = model()
    return neuralnet
