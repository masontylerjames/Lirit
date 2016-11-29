from keras.layers import *
from keras.models import Model, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateInputsAndTargets
from src.features import features_shape, noteStateMatrixToInputForm
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
import numpy as np
from src.featuresmodel import addfeatures


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


def model():
    n_steps = 128
    n = 173
    dropout = 0.5
    input_layer = Input(shape=(n_steps, 87, n))
    permute_1 = Permute((2, 1, 3))(input_layer)
    with tf.device('/gpu:0'):
        per_pitch_lstm_1 = TimeDistributed(
            LSTM(256, return_sequences=True), name='pitch_lstms_1')(permute_1)
        dropout_1 = Dropout(dropout)(per_pitch_lstm_1)
    with tf.device('/gpu:1'):
        per_pitch_lstm_2 = TimeDistributed(
            LSTM(256), name='pitch_lstms_2')(dropout_1)
        dropout_2 = Dropout(dropout)(per_pitch_lstm_2)
    with tf.device('/gpu:2'):
        pitch_lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(
            dropout_2)
        dropout_3 = Dropout(dropout)(pitch_lstm_1)
    with tf.device('/gpu:3'):
        pitch_lstm_2 = Bidirectional(
            LSTM(64, return_sequences=True))(dropout_3)
        dropout_4 = Dropout(dropout)(pitch_lstm_2)
    with tf.device('/gpu:4'):
        dense_1 = TimeDistributed(Dense(32))(dropout_4)
        dense_2 = TimeDistributed(Dense(2))(dense_1)
        out = Activation('sigmoid')(dense_2)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

if __name__ == '__main__':
    pass
