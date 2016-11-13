from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.model import model, input_shape
from src.train_model import getfiles, offset, generateXY
import numpy as np


class Lirit(object):

    def __init__(self):
        self.model = model()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def fitmidi(self, filename, **kwargs):
        statematrix = midiToStateMatrix(filename)
        X, Y = generateXY(statematrix)
        self.model.fit(X, Y, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        X, Y = generateXY(midiToStateMatrix(files[0]))
        for f in files[1:]:
            statematrix = midiToStateMatrix(f)
            X_f, Y_f = generateXY(statematrix)
            X += X_f
            Y += Y_f
        self.model.fit(X, Y, **kwargs)

    def compose(self, length, filename='example', seed=None):
        '''
        INPUT: int

        length: length of the resulting music piece in number of 32nd notes
        filename: the name of the file where the result is saved
        seed: a single input entry for the neural network to start with. If None
        it's seeded with random numbers
        '''
        statematrix = None
        if seed is None:
            seed = np.random.random(input_shape)
            seed = seed[np.newaxis]
        predict = cleanstatematrix(self.model.predict(seed))
        statematrix = predict[0]
        while len(statematrix) < length:
            predict = cleanstatematrix(self.model.predict(predict))
            statematrix += predict[0][-offset:]
            break
        noteStateMatrixToMidi(statematrix[:length], filename)


def cleanstatematrix(statematrix):
    sample = np.random.random(statematrix.shape)
    sm = sample < statematrix
    return sm

if __name__ == '__main__':
    pass
