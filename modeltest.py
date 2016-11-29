from keras.layers import *
from keras.models import Model
from src.miditransform import midiToStateMatrix
from src.fit import generateInputsAndTargets


def run_a_model(model_name, n_steps, **kwargs):
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    if model_name == 'simple_model':
        nn = simple_model(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_2':
        nn = model_2(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_3':
        nn = model_3(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_beat':
        nn = model_beat(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
    if model_name == 'model_deep_1':
        nn = model_deep_1(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_convolution':
        nn = model_convolution(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_bidirectional':
        nn = model_bidirectional(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_biaxial_words':
        nn = model_biaxial_words(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]
    if model_name == 'model_biaxial':
        nn = model_biaxial(n_steps)
        X, Y = generateInputsAndTargets(sm, n_steps)
        X = X[0]

    nn.fit(X, Y, **kwargs)
    nn.summary()


def model_biaxial(n_steps):
    input_layer = Input(shape=(n_steps, 87, 2))
    reshape_1 = TimeDistributed(
        Reshape((87, 2, 1)), name='reshape_1')(input_layer)
    padding_1 = TimeDistributed(
        ZeroPadding2D(padding=(12, 0)), name='padding')(reshape_1)
    reshape_2 = TimeDistributed(
        Reshape((87 + 24, 2)), name='reshape_2')(padding_1)
    conv = TimeDistributed(Convolution1D(
        64, 25), name='conv')(reshape_2)
    permute_1 = Permute((2, 1, 3))(conv)
    per_pitch_lstm = TimeDistributed(
        LSTM(128), name='pitch_lstms')(permute_1)
    pitch_lstm_1 = LSTM(64, return_sequences=True)(per_pitch_lstm)
    dense_1 = TimeDistributed(Dense(2))(pitch_lstm_1)
    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(dense_1)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_biaxial_words(n_steps):
    # neat but too slow
    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87))
    embed = TimeDistributed(Embedding(87 * 3, 87))(input_layer)
    pitch_lstm = TimeDistributed(Bidirectional(
        LSTM(128, return_sequences=True)))(embed)
    permute = Permute((2, 1, 3))(pitch_lstm)
    time_lstm = TimeDistributed(LSTM(128))(permute)
    final_dense = TimeDistributed(Dense(2))(time_lstm)

    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(final_dense)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_bidirectional(n_steps):

    base_input = Input(shape=(n_steps, 87, 2))

    out = Activation('sigmoid')(raw_out)
    model = Model(input=base_input, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_convolution(n_steps):
    # add 2d convolution
    # .0176 loss after 100 epochs, .0165 with 3 convolutions
    silos = []
    base_input = Input(shape=(n_steps, 87, 2))
    # shared weight LSTM
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(base_input)
    shared_lstm_1 = TimeDistributed(
        LSTM(256, return_sequences=True), name='sw_lstm_1')(shared_setup)
    shared_lstm_2 = TimeDistributed(
        LSTM(256), name='sw_lstm_2')(shared_lstm_1)
    shared_dense_1 = TimeDistributed(
        Dense(128), name='sw_dense_1')(shared_lstm_2)
    shared_dense_2 = TimeDistributed(
        Dense(87), name='sw_dense_2')(shared_dense_1)
    raw_out = Permute((2, 1))(shared_dense_2)
    silos.append(raw_out)

    # convolution
    n = 6
    cv_reshape_1 = TimeDistributed(Reshape((87, 2, 1)))(base_input)
    cv_cv = TimeDistributed(Convolution2D(n, 25, 2))(cv_reshape_1)
    cv_flatten = TimeDistributed(Flatten())(cv_cv)
    cv_lstm_1 = LSTM(64 * n)(cv_flatten)
    cv_lstm_2 = LSTM(32 * n)(cv_lstm_1)
    cv_dense = Dense(174)(cv_lstm_2)
    cv_reshape_2 = Reshape((87, 2))(cv_dense)
    silos.append(cv_reshape_2)

    merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(merge_layer)
    model = Model(input=base_input, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_deep_1(n_steps):
    # just go deep
    # depth did not help
    # shared weights
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(128, return_sequences=True), name='sw_lstm_1')(shared_setup)
    shared_lstm_2 = TimeDistributed(
        LSTM(128, return_sequences=True), name='sw_lstm_2')(shared_lstm_1)
    shared_lstm_3 = TimeDistributed(
        LSTM(128, return_sequences=True), name='sw_lstm_3')(shared_lstm_2)
    shared_lstm_4 = TimeDistributed(
        LSTM(128), name='sw_lstm_4')(shared_lstm_3)
    shared_dense_1 = TimeDistributed(
        Dense(87), name='sw_dense_1')(shared_lstm_4)
    raw_out = Permute((2, 1))(shared_dense_1)

    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(raw_out)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model
    pass


def model_beat(n_steps):
    # simple_model with beats
    # .0518 loss after 100 epochs
    silos = []
    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(128), name='sw_lstm_1')(shared_setup)
    shared_dense_1 = TimeDistributed(
        Dense(87), name='sw_dense_1')(shared_lstm_1)
    raw_out = Permute((2, 1))(shared_dense_1)
    silos.append(raw_out)

    beat_input = Input((n_steps, 4))
    beat_lstm = LSTM(16)(beat_input)
    beat_dense = Dense(174)(beat_lstm)
    beat_reshape = Reshape((87, 2))(beat_dense)
    silos.append(beat_reshape)

    merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(merge_layer)
    model = Model(input=[input_layer, beat_input], output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_3(n_steps):
    # simple model but with another lstm layer
    # .0427 loss after 100 epochs
    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(128, return_sequences=True), name='sw_lstm_1')(shared_setup)
    shared_lstm_2 = TimeDistributed(
        LSTM(128), name='sw_lstm_2')(shared_lstm_1)
    shared_dense_1 = TimeDistributed(
        Dense(87), name='sw_dense_1')(shared_lstm_2)
    raw_out = Permute((2, 1))(shared_dense_1)

    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(raw_out)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def model_2(n_steps):
    # simple model but with a wider lstm layer
    # .0403 after 100 epochs
    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(256), name='sw_lstm_1')(shared_setup)
    shared_dense_1 = TimeDistributed(
        Dense(87), name='sw_dense_1')(shared_lstm_1)
    raw_out = Permute((2, 1))(shared_dense_1)

    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(raw_out)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def simple_model(n_steps):
    # 0.0480 after 100 epochs
    # shared weight LSTM
    input_layer = Input(shape=(n_steps, 87, 2))
    shared_setup = Permute(
        (3, 1, 2), name='sw_permute_1')(input_layer)
    shared_lstm_1 = TimeDistributed(
        LSTM(128), name='sw_lstm_1')(shared_setup)
    shared_dense_1 = TimeDistributed(
        Dense(87), name='sw_dense_1')(shared_lstm_1)
    raw_out = Permute((2, 1))(shared_dense_1)

    # merge_layer = merge(silos, mode='ave')
    out = Activation('sigmoid')(raw_out)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model
