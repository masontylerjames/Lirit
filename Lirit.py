from miditransform import noteStateMatrixToMidi
from src.model import model, input_shape
from src.train_model import multitrain, offset
import numpy as np


class Lirit(object):

    def __init__(self):
        self.model = model()

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
