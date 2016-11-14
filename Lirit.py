from keras.layers import LSTM, Activation, Reshape
from keras.models import Sequential
from os import listdir
from os.path import abspath
from fit import getfiles, generateXY, cleanstatematrix
from miditransform import noteStateMatrixToMidi, midiToStateMatrix
import cPickle as pickle
import numpy as np


class Lirit(object):

    def __init__(self, n_steps=256, offset=128):
        self.n_steps = n_steps
        self.lowerBound = 21
        self.upperBound = 108
        self.shape = (self.upperBound - self.lowerBound, 2)
        self.offset = offset
        self.model = model(self.n_steps, self.shape)

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidis(self, filenames, **kwargs):
        X, Y = [], []
        if isinstance(filenames, list):
            statematrix = midiToStateMatrix(
                filenames[0], self.lowerBound, self.upperBound)
            X, Y = generateXY(
                statematrix, self.n_steps, self.offset, self.shape)
            for f in filenames[1:]:
                statematrix = midiToStateMatrix(
                    f, self.lowerBound, self.upperBound)
                X_f, Y_f = generateXY(
                    statematrix, self.n_steps, self.offset, self.shape)
                X += X_f
                Y += Y_f
        else:
            statematrix = midiToStateMatrix(
                filename, self.lowerBound, self.upperBound)
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
        self.model.fit(X, Y, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        print '{} in pipeline'.format(files[0].split('/')[-1])
        X, Y = generateXY(midiToStateMatrix(
            files[0], self.lowerBound, self.upperBound), self.n_steps, self.offset)
        for f in files[1:]:
            print '{} in pipeline'.format(f.split('/')[-1])
            statematrix = midiToStateMatrix(
                f, self.lowerBound, self.upperBound)
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
        noteStateMatrixToMidi(
            statematrix[:length], self.lowerBound, self.upperBound, filename)

    def save(self, filename):
        with open(abspath(filename), 'w') as f:
            pickle.dump(self, f)


def model(n_steps, shape):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(np.prod(shape), return_sequences=True))
    model.add(Reshape(input_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    return model


if __name__ == '__main__':
    lirit = Lirit()
    midis = [abspath('data/train/mozart/mz_311_1_format0.mid')] * 20
    # collection = [abspath('data/train') + '/' +
    #               d for d in listdir('data/train')]
    # lirit.fitcollection(collection)
    lirit.fitmidis(midis)
    lirit.save('lirit.pkl')
    l = 128 * 55
    for i in range(1):
        filename = 'test{}'.format(i)
        seed = [midiToStateMatrix(
            abspath('data/train/mozart/mz_311_1_format0.mid'), 21, 108)[:256]]
        lirit.compose(l, filename, seed=seed)
