from keras.layers import LSTM, Lambda, Activation, TimeDistributed, Convolution2D, Convolution3D, Dense
from keras.layers import Reshape, Input, merge, Permute
from keras.models import Model, load_model
from os.path import abspath
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateXY
from src.features import features_shape
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape
import numpy as np


class Lirit(object):

    def __init__(self, n_steps=128):
        self.n_steps = n_steps
        self.offset = 1
        self.model = model(self.n_steps)
        self.input_shape = (n_steps, features_shape[
                            0], features_shape[1])

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


def model_old(n_steps):
    '''
    OUTPUT: a compiled model
    '''
    # This model is a failure, it takes greater than an hour to build
    # the fit function
    inputs = Input(shape=(n_steps, state_shape[0], features_shape[1]))
    slices_1 = [Lambda(lambda x: x[:, :, i, :], output_shape=(
        n_steps, features_shape[1]))(inputs) for i in
        range(state_shape[0])]
    time_lstm_1 = [LSTM(features_shape[1], return_sequences=True)(layer)
                   for layer in slices_1]
    reshape_time = [Reshape((n_steps, 1, features_shape[1]))(layer)
                    for layer in time_lstm_1]
    cohesive_1 = merge(reshape_time, mode='concat', concat_axis=-2)
    permute_1 = Permute((2, 1, 3))(cohesive_1)
    slices_2 = [Lambda(lambda x: x[:, :, i, :], output_shape=(
        state_shape[0], features_shape[1]))(permute_1) for i in range(n_steps)]
    pitch_lstm_1 = [LSTM(features_shape[1], return_sequences=True)(layer)
                    for layer in slices_2]
    reshape_pitch = [Reshape((state_shape[0], 1, features_shape[1]))(
        layer) for layer in pitch_lstm_1]
    cohesive_2 = merge(reshape_pitch, mode='concat', concat_axis=-2)
    permute_2 = Permute((2, 1, 3))(cohesive_2)
    reshape_1 = Reshape(
        (n_steps, state_shape[0] * features_shape[1]))(permute_2)
    time_lstm_2 = LSTM(state_shape[0] * state_shape[1])(reshape_1)
    reshape_2 = Reshape((state_shape[0], state_shape[1]))(time_lstm_2)
    model = Model(input=inputs, output=reshape_2)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


def model_old_biax(n_steps):
    '''
    different biaxial model
    '''
    n = [64, 64, 64, 64]

    inputs = Input(shape=(n_steps, state_shape[0], features_shape[1]))
    permute_1 = Permute((2, 1, 3))(inputs)
    distributedLSTM_1 = TimeDistributed(
        LSTM(n[0], activation='relu', return_sequences=True))(permute_1)
    distributedLSTM_2 = TimeDistributed(
        LSTM(n[1], activation='relu', return_sequences=True))(distributedLSTM_1)
    permute_2 = Permute((2, 1, 3))(distributedLSTM_2)
    distributedLSTM_3 = TimeDistributed(
        LSTM(n[2], activation='relu', return_sequences=True))(permute_2)
    distributedLSTM_4 = TimeDistributed(
        LSTM(n[3], activation='relu', return_sequences=True))(distributedLSTM_3)
    permute_3 = Permute((2, 1, 3))(distributedLSTM_4)
    LSTM_1 = TimeDistributed(LSTM(2, activation='linear'))(permute_3)
    out = Activation('sigmoid')(LSTM_1)
    model = Model(input=inputs, output=out)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


def model_1(n_steps):
    inputs = Input(shape=(n_steps, state_shape[0], features_shape[1]))
    reshape_1 = Reshape(
        (n_steps, state_shape[0], features_shape[1], 1))(inputs)
    convolution_2d_1 = TimeDistributed(
        Convolution2D(2, 1, 2))(reshape_1)
    reshape_2 = Reshape(
        (n_steps, state_shape[0], 2, 1))(convolution_2d_1)
    convolution_2d_2 = TimeDistributed(
        Convolution2D(48, 24, 2))(reshape_2)
    reshape_3 = Reshape((n_steps, 64, 48, 1))(convolution_2d_2)
    convolution_3d_1 = Convolution3D(
        1, 2, 33, 25, subsample=(2, 1, 1))(reshape_3)
    reshape_4 = Reshape((64, 32 * 24))(convolution_3d_1)
    lstm_1 = LSTM(256, activation='linear',
                  return_sequences=True)(reshape_4)
    lstm_2 = LSTM(256, activation='linear')(lstm_1)
    dense_1 = Dense(87 * 2)(lstm_2)
    reshape_5 = Reshape((87, 2))(dense_1)
    activation = Activation('sigmoid')(reshape_5)
    model = Model(input=inputs, output=activation)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model(n_steps):
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape)
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(features_shape[1])]
    # n_input = [Lambda(lambda x: x[:, :, i, :], output_shape=(n_steps, features_shape[1]))(base_input)
    #                for i in range(features_shape[0])]
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


def model_3(n_steps):
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape)

    # do shared weights between on/off and actuation
    f_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
               for i in range(features_shape[1])]
    f_lstm_1 = LSTM(128, activation='linear')
    f_layer_1 = [f_lstm_1(layer) for layer in f_input]
    f_dense_1 = Dense(features_shape[0])
    f_layer_2 = [f_dense_1(layer) for layer in f_layer_1]
    f_reshape_1 = Reshape((features_shape[0], 1))
    f_layer_3 = [f_reshape_1(layer) for layer in f_layer_2]
    f_stitch = merge(f_layer_3, mode='concat', concat_axis=-1)

    # do shared weights between notes
    n_input = [Lambda(lambda x: x[:, :, i, :], output_shape=(n_steps, features_shape[1]))(base_input)
               for i in range(features_shape[0])]
    n_lstm_1 = LSTM(8, activation='linear')
    n_layer_1 = [n_lstm_1(layer) for layer in n_input]
    n_dense_1 = Dense(features_shape[1])
    n_layer_2 = [n_dense_1(layer) for layer in n_layer_1]
    n_reshape_1 = Reshape((1, features_shape[1]))
    n_layer_3 = [n_reshape_1(layer) for layer in n_layer_2]
    n_stitch = merge(n_layer_3, mode='concat', concat_axis=-2)

    activation = Activation('sigmoid')
    probabilites = [activation(f_stitch), activation(n_stitch)]
    out = merge(probabilites, mode='ave')
    model = Model(input=base_input, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

if __name__ == '__main__':
    pass
