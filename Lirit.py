from keras.layers import LSTM, Reshape
from keras.models import Sequential, load_model
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
        it's seeded with random numbers and then set to 1s and 0s based on a threshold
        '''
        statematrix = None
        offset = 0
        if seed is None:
            offset = self.n_steps
            seed = np.random.random(self.input_shape)
            seed = seed[np.newaxis]
            seed = (seed > .85) * 1
        predict = cleanstatematrix(self.model.predict(seed))
        statematrix = np.append(seed[0], predict, axis=0)
        while len(statematrix) < length + offset:
            perc10 = (length - self.n_steps + offset) / 10
            if len(statematrix) % perc10 == 0:
                print '{}% done'.format(len(statematrix) / perc10 * 10)
            predict = cleanstatematrix(
                self.model.predict(statematrix[-self.n_steps:][np.newaxis]))
            statematrix = np.append(statematrix, predict, axis=0)
        noteStateMatrixToMidi(statematrix[offset:], filename)

    def save(self, filename):
        self.model.save(abspath(filename))

    def load(self, filename):
        self.model = load_model(filename)


def model(n_steps):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(np.prod(state_shape)))
    model.add(Reshape(state_shape))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

if __name__ == '__main__':
    pass
