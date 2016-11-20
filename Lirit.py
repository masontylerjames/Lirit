from keras.layers import LSTM, Lambda, Activation, Dense, Convolution1D
from keras.layers import Reshape, Input, TimeDistributed, merge
from keras.models import Model, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateInputsAndTargets
from src.features import features_shape
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape, lowerBound, upperBound
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
        X, Y = None, None
        if not isinstance(filenames, list):
            filenames = [filenames]

        for f in filenames:
            print '{} in pipeline'.format(f.split('/')[-1])
            statematrix = midiToStateMatrix(f)
            if statematrix is not None:
                X_f, Y_f = generateInputsAndTargets(
                    statematrix, self.n_steps)
                if X is None or Y is None:
                    X, Y = X_f, Y_f
                else:
                    X = [np.append(X[i], X_f[i], axis=0)
                         for i in range(len(X))]
                    Y = np.append(Y, Y_f, axis=0)

        if X is None or Y is None:
            print 'No viable training data'
        else:
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
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(128, name='sw_lstm')
    layer_1 = [lstm_1(layer) for layer in features_input]
    dense_1 = Dense(features_shape[0], name='sw_dense')
    layer_2 = [dense_1(layer) for layer in layer_1]
    reshape_1 = Reshape((features_shape[0], 1), name='concat_prepare')
    layer_3 = [reshape_1(layer) for layer in layer_2]
    stitch_shared = merge(layer_3, mode='concat',
                          concat_axis=-1, name='sw_out_prepare')

    # accept beat input
    beat_input = Input(shape=(n_steps, 4), name='beat')
    beat_lstm = LSTM(16, name='beat_lstm')(beat_input)
    beat_dense = Dense(174, name='beat_dense')(beat_lstm)
    beat_reshape = Reshape(
        (87, 2), name='beat_out_prepare')(beat_dense)

    sum_silos = merge([stitch_shared, beat_reshape],
                      mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)
    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_new_2(n_steps):
    silos = []

    # adding beat info seemed successful
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(128, name='sw_lstm')
    layer_1 = [lstm_1(layer) for layer in features_input]
    dense_1 = Dense(features_shape[0], name='sw_dense')
    layer_2 = [dense_1(layer) for layer in layer_1]
    reshape_1 = Reshape((features_shape[0], 1), name='concat_prepare')
    layer_3 = [reshape_1(layer) for layer in layer_2]
    stitch_shared = merge(layer_3, mode='concat',
                          concat_axis=-1, name='sw_out_prepare')
    silos.append(stitch_shared)

    # accept beat input and neural net that
    beat_input = Input(shape=(n_steps, 4), name='beat')
    beat_lstm = LSTM(16, name='beat_lstm')(beat_input)
    beat_dense = Dense(
        features_shape[0] * 2, name='beat_dense')(beat_lstm)
    beat_reshape = Reshape(
        (features_shape[0], 2), name='beat_out_prepare')(beat_dense)
    silos.append(beat_reshape)

    # add convolution to try and bootstrap an understanding of key
    # through setting initial weights
    l = features_shape[0] - 11
    key_reshape_1 = Reshape(
        (n_steps, features_shape[0], 1))(features_input[0])
    key_weights = _genKeyWeights(l)
    key_convolution = TimeDistributed(
        Convolution1D(2, l, weights=key_weights, name='key_convolution'))(key_reshape_1)
    key_reshape_2 = TimeDistributed(Reshape((24,)))(key_convolution)
    key_lstm = LSTM(32, name='key_lstm')(key_reshape_2)
    key_dense = Dense(features_shape[0] * 2)(key_lstm)
    key_reshape_3 = Reshape((features_shape[0], 2))(key_dense)
    silos.append(key_reshape_3)

    # sum silos and then put through sigmoid activation
    sum_silos = merge(silos, mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)

    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_new_3(n_steps):
    silos = []

    # adding beat info seemed successful
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(256, name='sw_lstm')
    layer_1 = [lstm_1(layer) for layer in features_input]
    dense_1 = Dense(features_shape[0], name='sw_dense')
    layer_2 = [dense_1(layer) for layer in layer_1]
    reshape_1 = Reshape((features_shape[0], 1), name='concat_prepare')
    layer_3 = [reshape_1(layer) for layer in layer_2]
    stitch_shared = merge(layer_3, mode='concat',
                          concat_axis=-1, name='sw_out_prepare')
    silos.append(stitch_shared)

    # accept beat input and neural net that
    beat_input = Input(shape=(n_steps, 4), name='beat')
    beat_lstm = LSTM(32, name='beat_lstm')(beat_input)
    beat_dense = Dense(
        features_shape[0] * 2, name='beat_dense')(beat_lstm)
    beat_reshape = Reshape(
        (features_shape[0], 2), name='beat_out_prepare')(beat_dense)
    silos.append(beat_reshape)

    # add convolution to try and bootstrap an understanding of key
    # through setting initial weights
    l = features_shape[0] - 11
    key_reshape_1 = Reshape(
        (n_steps, features_shape[0], 1))(features_input[0])
    key_weights = _genKeyWeights(l)
    key_convolution = TimeDistributed(
        Convolution1D(2, l, weights=key_weights, name='key_convolution'))(key_reshape_1)
    key_reshape_2 = TimeDistributed(
        Reshape((76 * 2,)))(key_convolution)
    key_lstm = LSTM(256, name='key_lstm')(key_reshape_2)
    key_dense = Dense(features_shape[0] * 2)(key_lstm)
    key_reshape_3 = Reshape((features_shape[0], 2))(key_dense)
    silos.append(key_reshape_3)

    # sum silos and then put through sigmoid activation
    sum_silos = merge(silos, mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)

    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def _genKeyWeights(l):
    major = np.array(
        [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1]).reshape(12, 1)
    minor = np.array(
        [1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1]).reshape(12, 1)
    while len(major) < upperBound:
        major = np.append(major, major, axis=0)
        minor = np.append(minor, minor, axis=0)
    keys = np.append(major, minor, axis=1)[lowerBound:lowerBound + l]
    keys = keys[:, np.newaxis, np.newaxis]
    return [keys, np.zeros(2)]


def _genKeyWeights2():
    major = np.array(
        [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1]).reshape(12, 1)
    minor = np.array(
        [1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1]).reshape(12, 1)
    keys = np.append(major, minor, axis=1)
    keys = keys[:, np.newaxis, np.newaxis]
    return [keys, np.zeros(2)]

if __name__ == '__main__':
    pass
