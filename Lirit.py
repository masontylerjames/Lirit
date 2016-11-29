from keras.layers import LSTM, Lambda, Activation, Dense, Convolution1D, Embedding
from keras.layers import Reshape, Input, TimeDistributed, merge, Dropout, Flatten
from keras.models import Model, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateInputsAndTargets
from src.features import features_shape, noteStateMatrixToInputForm
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
import numpy as np
from featuresmodel import addfeatures


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
            pass
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
        sm_offset = self.n_steps
        if seed is None:
            sm_offset = self.n_steps
            seed = generateSeed(self.input_shape)

        conservatism = 1.
        for state in seed:
            N_notes = state[:, 0].sum()
            conservatism = calcConservatism(N_notes, conservatism)

        inputs = addfeatures(seed)
        probas = self.model.predict(
            inputs[np.newaxis])  # generate probabilites
        # turn probas into predictions
        predict = outputToState(probas, conservatism=conservatism)
        # append predictions to statematrix
        statematrix = np.append(seed, predict, axis=0)

        while len(statematrix) < length + sm_offset:
            N_notes = predict[:, 0].sum()
            conservatism = calcConservatism(N_notes, conservatism)

            if verbose:
                print 'Created {} of {} steps'.format(len(statematrix) - self.n_steps, length - self.n_steps + sm_offset)
            inputs = addfeatures(
                statematrix[-self.n_steps:])[np.newaxis]
            # generate probabilites
            probas = self.model.predict(inputs)
            # turn probas into predictions
            predict = outputToState(probas, conservatism)
            # append predictions to statematrix
            statematrix = np.append(statematrix, predict, axis=0)

        noteStateMatrixToMidi(statematrix[sm_offset:], filename)

    def save(self, filename):
        self.model.save(abspath(filename))

    def load(self, filename):
        self.model = load_model(filename)


def calcConservatism(n, conservatism):
    if n < 2:
        conservatism -= .02
    conservatism += (1. - conservatism) * .3
    return conservatism


def model_shortdropout(n_steps):
    # short onoff convolution
    silos = []

    # adding beat info seemed successful
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(256, name='sw_lstm_1')
    layer_1 = [lstm_1(layer) for layer in features_input]
    dropout = Dropout(.25)
    layer_drop_1 = [dropout(layer) for layer in layer_1]
    dense_2 = Dense(features_shape[0], name='sw_dense_2')
    layer_4 = [dense_2(layer) for layer in layer_drop_1]
    reshape_1 = Reshape(
        (features_shape[0], 1), name='sw_concat_prepare')
    layer_5 = [reshape_1(layer) for layer in layer_4]
    stitch_shared = merge(layer_5, mode='concat',
                          concat_axis=-1, name='sw_out_prepare')
    silos.append(stitch_shared)

    # accept beat input and neural net that
    beat_input = Input(shape=(n_steps, 4), name='beat')
    beat_lstm = LSTM(16, name='beat_lstm')(beat_input)
    beat_dense = Dense(
        features_shape[0] * 2, name='beat_dense_2')(beat_lstm)
    beat_reshape = Reshape(
        (features_shape[0], 2), name='beat_out_prepare')(beat_dense)
    silos.append(beat_reshape)

    # add convolution to try and bootstrap an understanding of key
    # through setting initial weights
    key_reshape_1 = Reshape(
        (n_steps, features_shape[0], 1), name='key_add_dim')(features_input[0])
    key_weights = _genKeyWeights3()
    key_convolution = TimeDistributed(
        Convolution1D(4, 12, weights=key_weights, name='key_convolution'), name='key_convolutions')(key_reshape_1)
    key_reshape_2 = TimeDistributed(
        Reshape((76 * 4,)), name='key_flatten')(key_convolution)
    key_dropout_1 = Dropout(.25)(key_reshape_2)
    key_lstm = LSTM(512, name='key_lstm')(key_dropout_1)
    key_dropout_2 = Dropout(.25)(key_lstm)
    key_dense_2 = Dense(
        features_shape[0] * 2, name='key_dense_2')(key_dropout_2)
    key_reshape_3 = Reshape(
        (features_shape[0], 2), name='key_prepare')(key_dense_2)
    silos.append(key_reshape_3)

    # sum silos and then put through sigmoid activation
    sum_silos = merge(silos, mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)

    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_words(n_steps):
    embed_dim = 16
    base_input = Input(shape=(n_steps, 87))
    embedding_td = TimeDistributed(
        Embedding(261, embed_dim, input_length=87))(base_input)
    reshape_1 = Reshape((n_steps, 87, embed_dim, 1))(embedding_td)
    n = 1
    conv_td_td = TimeDistributed(
        TimeDistributed(Convolution1D(n, embed_dim)))(reshape_1)
    flatten_1 = TimeDistributed(Flatten())(conv_td_td)
    lstm = LSTM(128 * n)(flatten_1)
    dense = Dense(174)(lstm)
    reshape_2 = Reshape((87, 2))(dense)

    model = Model(input=base_input, output=reshape_2)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model(n_steps):
    # short onoff convolution
    silos = []

    # adding beat info seemed successful
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(256, return_sequences=True, name='sw_lstm_1')
    layer_1 = [lstm_1(layer) for layer in features_input]
    lstm_2 = LSTM(256, name='sw_lstm_2')  # deeper
    layer_2 = [lstm_2(layer) for layer in layer_1]
    dense_1 = Dense(256, name='sw_dense_1')  # deeper
    layer_3 = [dense_1(layer) for layer in layer_2]
    dense_2 = Dense(features_shape[0], name='sw_dense_2')
    layer_4 = [dense_2(layer) for layer in layer_3]
    reshape_1 = Reshape(
        (features_shape[0], 1), name='sw_concat_prepare')
    layer_5 = [reshape_1(layer) for layer in layer_4]
    stitch_shared = merge(layer_5, mode='concat',
                          concat_axis=-1, name='sw_out_prepare')
    silos.append(stitch_shared)

    # accept beat input and neural net that
    beat_input = Input(shape=(n_steps, 4), name='beat')
    beat_lstm = LSTM(16, name='beat_lstm')(beat_input)
    beat_dense_1 = Dense(256, name='beat_dense_1')(beat_lstm)
    beat_dense = Dense(
        features_shape[0] * 2, name='beat_dense_2')(beat_dense_1)
    beat_reshape = Reshape(
        (features_shape[0], 2), name='beat_out_prepare')(beat_dense)
    silos.append(beat_reshape)

    # add convolution to try and bootstrap an understanding of key
    # through setting initial weights
    key_reshape_1 = Reshape(
        (n_steps, features_shape[0], 1), name='key_add_dim')(features_input[0])
    key_weights = _genKeyWeights3()
    key_convolution = TimeDistributed(
        Convolution1D(4, 12, weights=key_weights, name='key_convolution'), name='key_convolutions')(key_reshape_1)
    key_reshape_2 = TimeDistributed(
        Reshape((76 * 4,)), name='key_flatten')(key_convolution)
    key_lstm = LSTM(128, name='key_lstm')(key_reshape_2)
    key_keys = Dense(48, name='key_regularization')(key_lstm)
    key_activation = Activation(
        'softmax', name='key_softmax')(key_keys)
    key_dense_1 = Dense(256, name='key_dense_1')(key_activation)
    key_dense_2 = Dense(
        features_shape[0] * 2, name='key_dense_2')(key_dense_1)
    key_reshape_3 = Reshape(
        (features_shape[0], 2), name='key_prepare')(key_dense_2)
    silos.append(key_reshape_3)

    # sum silos and then put through sigmoid activation
    sum_silos = merge(silos, mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)

    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def _genKeyWeights3():
    major = np.array([1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1])
    nat_minor = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1])
    har_minor = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1])
    mel_minor = np.array([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1])
    keys = [major, nat_minor, har_minor, mel_minor]
    keys = [key.reshape(12, 1, 1, 1) for key in keys]
    return [np.concatenate(keys, axis=3), -1 * np.ones(4)]

if __name__ == '__main__':
    pass
