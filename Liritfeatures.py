from keras.layers import LSTM, Reshape, Activation, Dense
from keras.models import Sequential, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateXY
from src.features import noteStateMatrixToInputForm
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape
import numpy as np

shape = (87, 25)


class Lirit(object):

    def __init__(self, n_steps=128):
        self.n_steps = n_steps
        self.offset = 1
        self.model = model(self.n_steps)
        self.input_shape = (n_steps, shape[0], shape[1])

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidis(self, filenames, **kwargs):
        X, Y = [], []
        if isinstance(filenames, list):
            print '{} in pipeline'.format(filenames[0].split('/')[-1])
            statematrix = midiToStateMatrix(filenames[0])
            statematrix = noteStateMatrixToInputForm(statematrix)
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
            for f in filenames[1:]:
                print '{} in pipeline'.format(filenames[0].split('/')[-1])
                statematrix = midiToStateMatrix(f)
                statematrix = noteStateMatrixToInputForm(statematrix)
                X_f, Y_f = generateXY(
                    statematrix, self.n_steps, self.offset)
                X += X_f
                Y += Y_f
        else:
            statematrix = midiToStateMatrix(filenames)
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
            X = self._reshapeInput(X)
            Y = self._reshapeInput(Y)
        self.model.fit(X, Y, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        self.fitmidis(files, **kwargs)

    def compose(self, length, filename='example', seed=None, verbose=False):
        '''
        INPUT: int

        length: length of the resulting music piece in number of 32nd notes
        filename: the name of the file where the result is saved
        seed: a single input entry for the neural network to start with. If
        None a seed is randomly generated
        '''
        statematrix = None
        sm_offset = 0
        if seed is None:
            sm_offset = self.n_steps
            seed = generateSeed(self.input_shape)
        if len(seed.shape) == 4:
            seed = self._reshapeInput(seed)

        probas = self.model.predict(seed)  # generate probabilites
        # turn probas into predictions with dimensions 87xn
        predict = outputToState(self._reshapeOutput(
            probas), self._reshapeOutput(seed[0]))
        # append flattened  predictions to statematrix
        statematrix = np.append(
            seed[0], self._reshapeInput(predict), axis=0)

        while len(statematrix) < length + sm_offset:
            if verbose:
                print 'Created {} of {} steps'.format(len(statematrix), length - self.n_steps + sm_offset)
            # generate probabilites
            probas = self.model.predict(
                statematrix[-self.n_steps:][np.newaxis])
            # turn probas into predictions with dimensions 87xn
            predict = outputToState(self._reshapeOutput(
                probas), self._reshapeOutput(statematrix))
            # append flattened  predictions to statematrix
            statematrix = np.append(
                statematrix, self._reshapeInput(predict), axis=0)

        statematrix = self._reshapeOutput(statematrix)
        noteStateMatrixToMidi(statematrix[sm_offset:], filename)

    def save(self, filename):
        self.model.save(abspath(filename))

    def load(self, filename):
        self.model = load_model(filename)

    def _reshapeInput(self, inputdata):
        inputdata = np.asarray(inputdata)
        inputshape = inputdata.shape
        newshape = np.append(
            inputshape[:-2], np.prod(inputshape[-2:]))
        return np.reshape(inputdata, newshape)

    def _reshapeOutput(self, outputdata):
        outputdata = np.asarray(outputdata)
        outputshape = outputdata.shape
        newshape = np.append(outputshape[:-1], state_shape)
        return np.reshape(outputdata, newshape)


def model(n_steps):
    '''
    OUTPUT: a compiled model
    '''
    input_shape = (n_steps, shape[0], shape[1])
    flat_shape = (n_steps, np.prod(shape))
    model = Sequential()
    # flattens the state matrix for LSTM
    # model.add(Reshape(flat_shape, input_shape=input_shape))
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(flat_shape)))
    model.add(LSTM(256))
    model.add(Dense(np.prod(shape)))
    model.add(Activation('sigmoid'))
    # model.add(Reshape(state_shape))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

if __name__ == '__main__':
    pass
