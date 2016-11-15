from keras.layers import LSTM, Reshape
from keras.models import Sequential
from os.path import abspath
from src.fit import getfiles, generateXY, cleanstatematrix
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape
import cPickle as pickle
import numpy as np

shape = state_shape


class Lirit(object):

    def __init__(self, n_steps=256):
        self.n_steps = n_steps
        self.offset = 1
        self.model = model(self.n_steps)
        self.input_shape = (n_steps, shape[0], shape[1])

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidis(self, filenames, **kwargs):
        X, Y = [], []
        if isinstance(filenames, list):
            statematrix = midiToStateMatrix(filenames[0])
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
            for f in filenames[1:]:
                statematrix = midiToStateMatrix(f)
                X_f, Y_f = generateXY(
                    statematrix, self.n_steps, self.offset)
                X += X_f
                Y += Y_f
        else:
            statematrix = midiToStateMatrix(filenames)
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
        self.model.fit(X, Y, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        print '{} in pipeline'.format(files[0].split('/')[-1])
        X, Y = generateXY(midiToStateMatrix(
            files[0]), self.n_steps, self.offset)
        for f in files[1:]:
            print '{} in pipeline'.format(f.split('/')[-1])
            statematrix = midiToStateMatrix(f)
            X_f, Y_f = generateXY(
                statematrix, self.n_steps, self.offset)
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
            seed = np.random.random(self.input_shape)
            seed = seed[np.newaxis]
            seed = (seed > .85) * 1
        predict = cleanstatematrix(self.model.predict(seed))
        statematrix = np.append(seed[0], predict[0])
        while len(statematrix) < length + self.n_steps:
            predict = cleanstatematrix(
                self.model.predict(statematrix[-self.n_steps:]))
            statematrix = np.append(statematrix, predict[
                                    0][-self.offset:], axis=0)
        noteStateMatrixToMidi(statematrix[n_steps:], filename)

    def save(self, filename):
        with open(abspath(filename), 'w') as f:
            pickle.dump(self, f)


def model(n_steps):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(512))
    model.add(LSTM(256))
    model.add(LSTM(np.prod(state_shape)))
    model.add(Reshape(state_shape))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

if __name__ == '__main__':
    pass
