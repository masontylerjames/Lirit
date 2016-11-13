from keras.layers import LSTM, Activation, Reshape
from keras.models import Sequential
from os import listdir
from os.path import abspath
from src.fit import getfiles, generateXY
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import shape
import cPickle as pickle
import numpy as np


class Lirit(object):

    def __init__(self, n_steps=256, offset=128):
        self.n_steps = n_steps
        self.input_shape = (n_steps, shape[0], shape[1])
        self.offset = offset
        self.model = model(self.n_steps, self.input_shape)

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidi(self, filename, **kwargs):
        statematrix = midiToStateMatrix(filename)
        X, Y = generateXY(statematrix, self.n_steps, self.offset)
        self.model.fit(X, Y, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        print '{} in pipeline'.format(files[0].split('/')[-1])
        X, Y = generateXY(midiToStateMatrix(files[0]))
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
            seed = seed > .9
        predict = cleanstatematrix(self.model.predict(seed))
        statematrix = predict[0]
        while len(statematrix) < length:
            predict = cleanstatematrix(self.model.predict(predict))
            statematrix = np.append(statematrix, predict[
                                    0][-self.offset:], axis=0)
            break
        noteStateMatrixToMidi(statematrix[:length], filename)

    def save(self, filename):
        with open(abspath(filename), 'w') as f:
            pickle.dump(self, f)


def model(n_steps, input_shape):
    '''
    OUTPUT: a compiled model
    '''
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model


def cleanstatematrix(statematrix):
    sample = np.random.random(statematrix.shape)
    sm = sample < statematrix
    return sm

if __name__ == '__main__':
    lirit = Lirit()
    collection = [abspath('data/train') + '/' +
                  d for d in listdir('data/train')]
    lirit.fitcollection(collection)
    l = 128 * 40
    for i in range(10):
        filename = 'test{}'.format(i)
        lirit.compose(l, filename)
