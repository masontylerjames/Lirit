from keras.layers import LSTM, Lambda, Activation, TimeDistributed, Convolution2D, Convolution3D, Dense
from keras.layers import Reshape, Input, merge, Permute
from keras.models import Model
from src.compose import outputToState, generateSeed
from src.fit import getfiles, generateXY, splitX
from src.features import features_shape
from src.miditransform import noteStateMatrixToMidi, midiToStateMatrix
from src.miditransform import state_shape
import numpy as np


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


def model_1(n_steps):
    # tried some convolutional stuff
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


def model_3_old(n_steps):
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
    # the number of splits for notes makes this intractable
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


def model_3(n_steps):
    # same as model_3_old but split the inputs manually
    in_shape = (n_steps, features_shape[0], features_shape[1])

    # do shared weights between on/off and actuation
    f_inputs = [Input(shape=(n_steps, features_shape[0]))
                for i in range(features_shape[1])]
    f_lstm_1 = LSTM(128, activation='linear')
    f_layer_1 = [f_lstm_1(layer) for layer in f_inputs]
    f_dense_1 = Dense(features_shape[0])
    f_layer_2 = [f_dense_1(layer) for layer in f_layer_1]
    f_reshape_1 = Reshape((features_shape[0], 1))
    f_layer_3 = [f_reshape_1(layer) for layer in f_layer_2]
    f_stitch = merge(f_layer_3, mode='concat', concat_axis=-1)

    # do shared weights between notes
    n_inputs = [Input(shape=(n_steps, features_shape[1]))
                for i in range(features_shape[0])]
    n_lstm_1 = LSTM(8, activation='linear')
    n_layer_1 = [n_lstm_1(layer) for layer in n_inputs]
    n_dense_1 = Dense(features_shape[1])
    n_layer_2 = [n_dense_1(layer) for layer in n_layer_1]
    n_reshape_1 = Reshape((1, features_shape[1]))
    n_layer_3 = [n_reshape_1(layer) for layer in n_layer_2]
    n_stitch = merge(n_layer_3, mode='concat', concat_axis=-2)

    activation = Activation('sigmoid')
    probabilites = [activation(f_stitch), activation(n_stitch)]
    out = merge(probabilites, mode='ave')
    model = Model(input=f_inputs + n_inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_new_6(n_steps):
    # 3D average pooling
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

    # average pooling for each pitch
    pool_reshape_channel = Reshape(in_shape + (1,))(base_input)
    pooling = AveragePooling3D((16, 1, 2))(pool_reshape_channel)
    pool_reshape_lstm = Reshape((128 / 16, 87))(pooling)
    pool_lstm = LSTM(256)(pool_reshape_lstm)
    pool_dense = Dense(features_shape[0] * 2)(pool_lstm)
    pool_reshape_out = Reshape((features_shape[0], 2))(pool_dense)

    sum_silos = merge([stitch_shared, beat_reshape, pool_reshape_out],
                      mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)
    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_new_7(n_steps):
    # Deep ReLU lstm, failure
    in_shape = (n_steps, features_shape[0], features_shape[1])
    base_input = Input(shape=in_shape, name='sm_slice')

    # shared weights between on/off and actuation features
    features_input = [Lambda(lambda x: x[:, :, :, i], output_shape=(n_steps, features_shape[0]))(base_input)
                      for i in range(2)]
    lstm_1 = LSTM(256, activation='relu',
                  return_sequences=True, name='sw_lstm')
    lstm_2 = LSTM(256, activation='relu', return_sequences=True)
    lstm_3 = LSTM(256, activation='relu', return_sequences=True)
    lstm_4 = LSTM(256, activation='relu')
    layer_1 = [lstm_1(layer) for layer in features_input]
    layer_1_2 = [lstm_2(layer) for layer in layer_1]
    layer_1_3 = [lstm_3(layer) for layer in layer_1_2]
    layer_1_4 = [lstm_4(layer) for layer in layer_1_3]
    dense_1 = Dense(features_shape[0], name='sw_dense')
    layer_2 = [dense_1(layer) for layer in layer_1_4]
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
    out = Activation('sigmoid', name='constrain_out')(stitch_shared)
    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_new_8(n_steps):
    # 3D average pooling and convolution
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

    # average pooling for each pitch
    pool_reshape_channel = Reshape(in_shape + (1,))(base_input)
    pooling = AveragePooling3D((16, 1, 2))(pool_reshape_channel)
    pool_reshape_lstm = Reshape((128 / 16, 87))(pooling)

    # add convolution to try and bootstrap an understanding of key
    # through setting initial weights
    key_reshape_1 = Reshape(
        (8, features_shape[0], 1))(pool_reshape_lstm)
    key_weights = _genKeyWeights2()
    key_convolution = TimeDistributed(
        Convolution1D(2, 12, weights=key_weights, name='key_convolution'))(key_reshape_1)
    key_reshape_2 = TimeDistributed(
        Reshape((76 * 2,)))(key_convolution)

    pool_lstm = LSTM(256)(key_reshape_2)
    pool_dense = Dense(features_shape[0] * 2)(pool_lstm)
    pool_reshape_out = Reshape((features_shape[0], 2))(pool_dense)

    sum_silos = merge([stitch_shared, beat_reshape, pool_reshape_out],
                      mode='sum', name='sum_silos')
    out = Activation('sigmoid', name='constrain_out')(sum_silos)
    inputs = [base_input, beat_input]
    model = Model(input=inputs, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def model_new_2(n_steps):
    # long onoff convolution
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

    def model(n_steps):
        # basic shared weights model
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
