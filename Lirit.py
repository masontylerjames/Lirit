from keras.layers import LSTM, Lambda, Activation, TimeDistributed, Convolution2D, Convolution3D, Dense
from keras.layers import Reshape, Input, merge, Permute
from keras.models import Model, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateXY, splitX
from src.features import features_shape
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape
import numpy as np


class Lirit(object):

    def __init__(self, n_steps=128, split=False):
        self.n_steps = n_steps
        self.offset = 1
        self.model = model(self.n_steps)
        self.input_shape = (n_steps, features_shape[
                            0], features_shape[1])
        self.split = split

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidis(self, filenames, **kwargs):
        X, Y = [], []
        if isinstance(filenames, list):
            print '{} in pipeline'.format(filenames[0].split('/')[-1])
            statematrix = midiToStateMatrix(filenames[0])
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
            for f in filenames[1:]:
                print '{} in pipeline'.format(filenames[0].split('/')[-1])
                statematrix = midiToStateMatrix(f)
                X_f, Y_f = generateXY(
                    statematrix, self.n_steps, self.offset)
                X += X_f
                Y += Y_f
        else:
            statematrix = midiToStateMatrix(filenames)
            X, Y = generateXY(statematrix, self.n_steps, self.offset)
            # X = self._reshapeInput(X)
            # Y = self._reshapeInput(Y)
        if self.split:
            X = splitX(X)

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
        # turn probas into predictions with dimensions 87x2
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
            # turn probas into predictions with dimensions 87x2
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
    # survival of the fittest BABY
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape)
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(features_shape[1])]
    lstm_1 = LSTM(128)
    layer_1 = [lstm_1(layer) for layer in features_input]
    dense_1 = Dense(features_shape[0])
    layer_2 = [dense_1(layer) for layer in layer_1]
    reshape_1 = Reshape((features_shape[0], 1))
    layer_3 = [reshape_1(layer)
               for layer in layer_2]
    stitch = merge(layer_3, mode='concat', concat_axis=-1)
    out = Activation('sigmoid')(stitch)
    model = Model(input=base_input, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_new_1(n_steps):
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape)

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(128, activation='linear')
    layer_1 = [lstm_1(layer) for layer in features_input]
    dense_1 = Dense(features_shape[0])
    layer_2 = [dense_1(layer) for layer in layer_1]
    reshape_1 = Reshape((features_shape[0], 1))
    layer_3 = [reshape_1(layer) for layer in layer_2]
    stitch_shared = merge(layer_3, mode='concat', concat_axis=-1)

    # accept beat input
    beat_input = Input(shape=(n_steps, 4))
    beat_lstm = LSTM(174, activation='linear')(beat_input)
    beat_reshape = Reshape((87, 2))

    sum_silos = merge([stitch_shared, beat_reshape], mode='sum')
    out = Activation('sigmoid')(sum_silos)
    model = Model(input=base_input, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model

if __name__ == '__main__':
    pass
