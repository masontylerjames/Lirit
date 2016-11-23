from keras.layers import LSTM, Dense, Activation, Convolution2D, MaxPooling1D
from keras.layers import Input, Permute, TimeDistributed, merge, Reshape, Flatten
from keras.models import Model
from src.miditransform import midiToStateMatrix
from src.fit import generateInputsAndTargets, generateInputs


def model(n_steps):
    silos = []

    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(256, return_sequences=True), name='sw_lstm_1')(shared_setup)
    shared_lstm_2 = TimeDistributed(
        LSTM(256, return_sequences=True), name='sw_lstm_2')(shared_lstm_1)
    shared_lstm_3 = TimeDistributed(
        LSTM(256), name='sw_lstm_3')(shared_lstm_2)
    shared_dense_1 = TimeDistributed(
        Dense(128), name='sw_dense_1')(shared_lstm_3)
    shared_dense_2 = TimeDistributed(
        Dense(87), name='sw_dense_2')(shared_dense_1)
    shared_out = Permute((2, 1), name='sw_out')(shared_dense_2)
    silos.append(shared_out)

    # beats
    beat_input = Input(shape=(n_steps, 4))
    beat_lstm = LSTM(16, name='beat_lstm')(beat_input)
    beat_dense_1 = Dense(128, name='beat_dense_1')(beat_lstm)
    beat_dense_2 = Dense(174, name='beat_dense_2')(beat_dense_1)
    beat_reshape = Reshape((87, 2), name='beat_out')(beat_dense_2)
    silos.append(beat_reshape)

    # convolution
    n = 1
    conv_reshape_1 = TimeDistributed(
        Reshape((87, 2, 1)), name='conv_reshape_1')(input_layer)
    convolution = TimeDistributed(
        Convolution2D(n, 12, 2), name='conv_2d')(conv_reshape_1)
    conv_reshape_2 = Reshape(
        (n_steps, 76, n), name='conv_reshape_2')(convolution)
    conv_pooling = TimeDistributed(
        MaxPooling1D(n, 12), name='conv_pooling')(conv_reshape_2)
    conv_flatten = TimeDistributed(
        Flatten(), name='conv_flatten')(conv_pooling)
    conv_lstm = LSTM(256, name='conv_lstm')(conv_flatten)
    conv_dense = Dense(174, name='conv_dense')(conv_lstm)
    conv_reshape_3 = Reshape((87, 2), name='conv_out')(conv_dense)
    silos.append(conv_reshape_3)

    merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(merge_layer)
    model = Model(input=[input_layer, beat_input], output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

if __name__ == '__main__':
    nn = model(128)
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    X, Y = generateInputsAndTargets(sm, 128)
    # sm_neg = (sm * 2) - 1
    # X = generateInputs(sm_neg, 128)
    nn.fit(X, Y, batch_size=128)
