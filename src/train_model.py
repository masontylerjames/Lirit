from miditransform import midiToStateMatrix
from model import model


def train(midifile):
    statematrix = midiToStateMatrix(midifile)
    neuralnet = model()
    pass
